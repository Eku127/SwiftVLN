from __future__ import annotations

import copy
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.modeling_utils import no_init_weights

from action_formats import ORIGINAL, ORIGINAL_UNNORM_KEY, get_original_norm_stats
from openfly_core import OpenVLAForActionPrediction, register_openfly_auto_classes


_STEP_RE = re.compile(r"step-(\d+)")


def _sort_key(path: Path) -> tuple[int, str]:
    match = _STEP_RE.search(path.name)
    step = int(match.group(1)) if match else -1
    return step, path.name


def resolve_native_checkpoint_path(model_name_or_path: str) -> tuple[str, str]:
    path = Path(model_name_or_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Native OpenFly checkpoint must be a .pt file: {path}")
        if path.parent.name != "checkpoints":
            raise ValueError(f"Expected native checkpoint under a checkpoints/ directory: {path}")
        return str(path), str(path.parents[1])

    if not path.is_dir():
        raise FileNotFoundError(f"Native OpenFly model path not found: {path}")

    checkpoints_dir = path / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Native OpenFly run dir missing checkpoints/: {path}")

    checkpoints = sorted(checkpoints_dir.glob("*.pt"), key=_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No native OpenFly checkpoints found under: {checkpoints_dir}")

    return str(checkpoints[-1]), str(path)


def resolve_native_processor_source(model_name_or_path: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    model_path = Path(model_name_or_path).expanduser().resolve()
    candidates = []
    if model_path.is_dir():
        candidates.extend([model_path, model_path.parent, model_path.parent.parent])
    elif model_path.is_file():
        candidates.extend([model_path.parent, model_path.parent.parent, model_path.parent.parent.parent])

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root / "baseline" / "openfly" / "model" / "openfly-agent-7b")

    for candidate in candidates:
        if (candidate / "preprocessor_config.json").exists() and (candidate / "tokenizer_config.json").exists():
            return str(candidate)

    raise FileNotFoundError(
        "Unable to resolve a processor/tokenizer source for native OpenFly backend. "
        "Set PROCESSOR_PATH or ensure baseline/openfly/model/openfly-agent-7b exists."
    )


def _rename_layer_scale(key: str) -> str:
    key = key.replace(".ls1.gamma", ".ls1.scale_factor")
    key = key.replace(".ls2.gamma", ".ls2.scale_factor")
    return key


def _map_native_vision_key(key: str) -> str:
    if key.startswith("dino_featurizer."):
        mapped = "vision_backbone.featurizer." + key.removeprefix("dino_featurizer.")
    elif key.startswith("siglip_featurizer."):
        mapped = "vision_backbone.fused_featurizer." + key.removeprefix("siglip_featurizer.")
    else:
        raise KeyError(f"Unsupported native vision key: {key}")
    return _rename_layer_scale(mapped)


def _map_native_llm_key(key: str) -> str:
    if not key.startswith("llm."):
        raise KeyError(f"Unsupported native llm key: {key}")
    return "language_model." + key.removeprefix("llm.")


def _map_native_projector_key(key: str) -> str:
    mapping = {
        "projector.0.weight": "projector.fc1.weight",
        "projector.0.bias": "projector.fc1.bias",
        "projector.2.weight": "projector.fc2.weight",
        "projector.2.bias": "projector.fc2.bias",
        "projector.4.weight": "projector.fc3.weight",
        "projector.4.bias": "projector.fc3.bias",
    }
    if key not in mapping:
        raise KeyError(f"Unsupported native projector key: {key}")
    return mapping[key]


def convert_native_state_dict_to_hf(native_model_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}

    vision_state = native_model_state.get("vision_backbone")
    llm_state = native_model_state.get("llm_backbone")
    projector_state = native_model_state.get("projector")

    if not isinstance(vision_state, dict) or not isinstance(llm_state, dict) or not isinstance(projector_state, dict):
        raise ValueError("Native OpenFly checkpoint must contain vision_backbone, llm_backbone, and projector blocks")

    for key, value in vision_state.items():
        mapped[_map_native_vision_key(key)] = value

    for key, value in llm_state.items():
        mapped[_map_native_llm_key(key)] = value

    for key, value in projector_state.items():
        mapped[_map_native_projector_key(key)] = value

    return mapped


def _filter_expected_missing_keys(missing_keys: list[str]) -> list[str]:
    filtered = []
    for key in missing_keys:
        if key.endswith(".rotary_emb.inv_freq"):
            continue
        filtered.append(key)
    return filtered


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype | None):
    if dtype is None:
        yield
        return
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def build_native_hf_model(
    *,
    model_name_or_path: str,
    processor_source: str,
    cache_dir: str | None,
    grid_size: int,
    unnorm_key: str = ORIGINAL_UNNORM_KEY,
    use_flash_attention_2: bool = False,
    torch_dtype: torch.dtype | None = None,
) -> tuple[OpenVLAForActionPrediction, Any, dict[str, Any]]:
    register_openfly_auto_classes()
    checkpoint_path, run_dir = resolve_native_checkpoint_path(model_name_or_path)
    processor = AutoProcessor.from_pretrained(processor_source, cache_dir=cache_dir)

    config = copy.deepcopy(AutoConfig.from_pretrained(processor_source, cache_dir=cache_dir))
    config.grid_size = int(grid_size)
    config.action_format = ORIGINAL
    config.satnav_unnorm_key = unnorm_key
    config.norm_stats = get_original_norm_stats(unnorm_key)
    if use_flash_attention_2:
        setattr(config, "_attn_implementation", "flash_attention_2")

    with _temporary_default_dtype(torch_dtype):
        with no_init_weights():
            model = OpenVLAForActionPrediction(config)
    native_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    native_model_state = native_checkpoint.get("model")
    if not isinstance(native_model_state, dict):
        raise ValueError(f"Native checkpoint missing 'model' state: {checkpoint_path}")

    hf_state_dict = convert_native_state_dict_to_hf(native_model_state)
    load_result = model.load_state_dict(hf_state_dict, strict=False)
    missing_keys = _filter_expected_missing_keys(list(load_result.missing_keys))
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to map native OpenFly checkpoint into HF model.\n"
            f"checkpoint={checkpoint_path}\n"
            f"missing_keys={missing_keys}\n"
            f"unexpected_keys={unexpected_keys}"
        )

    backend_meta = {
        "backend": "scratch",
        "native_checkpoint_path": checkpoint_path,
        "native_run_dir": run_dir,
        "processor_source": os.path.abspath(processor_source),
        "action_format": ORIGINAL,
        "satnav_unnorm_key": unnorm_key,
        "grid_size": int(grid_size),
        "torch_dtype": str(torch_dtype).replace("torch.", "") if torch_dtype is not None else "default",
    }
    return model, processor, backend_meta
