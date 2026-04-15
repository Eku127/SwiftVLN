import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoModelForVision2Seq, AutoProcessor

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

_REPO_ROOT = os.path.abspath(os.path.join(_BASELINE_SRC, "..", "..", ".."))
_REPORTING_PATH = os.path.join(_REPO_ROOT, "src", "swiftvln", "common", "eval", "reporting.py")


def _load_reporting_module():
    spec = importlib.util.spec_from_file_location("swiftvln_reporting", _REPORTING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reporting helpers from {_REPORTING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reporting = _load_reporting_module()

from action_formats import (
    AUTO,
    COMPACT,
    ORIGINAL,
    ACTION_TO_NAME,
    ORIGINAL_UNNORM_KEY,
    convert_original_action_vector_to_action,
    parse_action_text,
    resolve_action_format,
)
from openfly_core import register_openfly_auto_classes
from prompting import build_openfly_prompt

from satnav.core.env import Env as SatNavEnv
from satnav.dataset.satnav_dataset import SatNavDataset


compute_weighted_trajectory_type_metrics = _reporting.compute_weighted_trajectory_type_metrics
load_satnav_reference_distribution = _reporting.load_satnav_reference_distribution


ERROR_NE_PENALTY = 500.0


class SatNavEnvWrapper:
    def __init__(self, satnav_env: SatNavEnv, config=None):
        self._env = satnav_env
        self._last_info = {}
        self._config = config

    def reset(self, episode) -> dict:
        return self._env.reset_to_episode(episode)

    def step(self, action: int):
        observations, done, info = self._env.step(action)
        self._last_info = info
        return observations, done

    def get_metrics(self) -> dict:
        return self._env.get_metrics()

    def get_last_step_info(self) -> dict:
        return self._last_info

    def get_rgb(self, obs: dict) -> np.ndarray:
        return obs["rgb"]

    def get_instruction(self, episode) -> str:
        instruction = episode.instruction
        if isinstance(instruction, dict):
            return instruction.get("text", instruction.get("instruction_text", ""))
        if hasattr(instruction, "text"):
            return instruction.text
        if hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        return str(instruction)

    @property
    def episode_over(self) -> bool:
        return self._env.episode_over

    @property
    def max_steps(self) -> int:
        return self._env.max_episode_steps

    def close(self):
        pass


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _resolve_processor_source(model_path: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    candidates = [
        Path(model_path),
        Path(model_path).parent,
        Path(model_path).parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "tokenizer_config.json").exists() or (candidate / "preprocessor_config.json").exists():
            return str(candidate)
    return model_path


def get_rank_sync_dir(output_path: str, run_id: str) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id or "default")
    return Path(output_path) / "_rank_sync" / safe_run_id


def mark_rank_complete(output_path: str, run_id: str, rank: int, assigned_episodes: int, finished_episodes: int) -> None:
    sync_dir = get_rank_sync_dir(output_path, run_id)
    sync_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "assigned_episodes": assigned_episodes,
        "finished_episodes": finished_episodes,
        "timestamp": time.time(),
    }
    tmp_path = sync_dir / f"rank_{rank}.json.tmp"
    done_path = sync_dir / f"rank_{rank}.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, done_path)


def wait_for_all_rank_markers(output_path: str, run_id: str, world_size: int, timeout_sec: int) -> None:
    sync_dir = get_rank_sync_dir(output_path, run_id)
    deadline = time.time() + max(timeout_sec, 1)
    while time.time() < deadline:
        if len(list(sync_dir.glob("rank_*.json"))) >= world_size:
            return
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for rank markers under {sync_dir}")


def build_episode_key(episode_id, scene_id) -> str:
    ep_id = str(episode_id) if episode_id is not None else ""
    scene = str(scene_id) if scene_id is not None else ""
    if not ep_id:
        return ""
    return f"{scene}::{ep_id}" if scene else ep_id


def load_dedup_results(result_file: str) -> list[dict[str, Any]]:
    if not os.path.exists(result_file):
        return []

    results_by_ep = {}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep_key = build_episode_key(result.get("episode_id"), result.get("scene_id"))
            if ep_key:
                results_by_ep[ep_key] = result
    return list(results_by_ep.values())


def save_summary(results: list[dict[str, Any]], output_path: str, args) -> None:
    if not results:
        summary = {"total_episodes": 0}
    else:
        summary = {
            "SR": float(np.mean([r["success"] for r in results])),
            "SPL": float(np.mean([r["spl"] for r in results])),
            "OS": float(np.mean([r["oracle_success"] for r in results])),
            "NE": float(np.mean([r["distance_to_goal"] for r in results if r["distance_to_goal"] < 1e6] or [0.0])),
            "avg_steps": float(np.mean([r["steps"] for r in results])),
            "total_episodes": len(results),
        }

        type_stats: dict[str, dict[str, list[float]]] = {}
        for result in results:
            ttype = result.get("trajectory_type", "unknown")
            type_stats.setdefault(ttype, {"sucs": [], "spls": [], "oss": [], "nes": [], "steps": []})
            type_stats[ttype]["sucs"].append(result["success"])
            type_stats[ttype]["spls"].append(result["spl"])
            type_stats[ttype]["oss"].append(result["oracle_success"])
            if result["distance_to_goal"] < 1e6:
                type_stats[ttype]["nes"].append(result["distance_to_goal"])
            type_stats[ttype]["steps"].append(result["steps"])

        if len(type_stats) > 1 or "unknown" not in type_stats:
            summary["by_trajectory_type"] = {}
            for ttype, stats in sorted(type_stats.items()):
                summary["by_trajectory_type"][ttype] = {
                    "SR": float(np.mean(stats["sucs"])),
                    "SPL": float(np.mean(stats["spls"])),
                    "OS": float(np.mean(stats["oss"])),
                    "NE": float(np.mean(stats["nes"])) if stats["nes"] else 0.0,
                    "avg_steps": float(np.mean(stats["steps"])),
                    "count": len(stats["sucs"]),
                }

            reference_distribution = load_satnav_reference_distribution(args.satnav_config_path)
            if reference_distribution:
                weighted = compute_weighted_trajectory_type_metrics(
                    summary["by_trajectory_type"],
                    reference_distribution,
                    metric_keys={
                        "SR": "SR",
                        "SPL": "SPL",
                        "OS": "OS",
                        "NE": "NE",
                        "avg_steps": "avg_steps",
                    },
                )
                if weighted:
                    summary["weighted_by_seen_unseen_distribution"] = weighted

    with open(os.path.join(output_path, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


class OpenFlySatNavEvaluator:
    def __init__(self, args, model, processor, action_format: str):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_format = action_format

    def _build_pixel_values(self, history_frames: deque[np.ndarray], current_rgb: np.ndarray) -> torch.Tensor:
        images = [Image.fromarray(current_rgb).convert("RGB")]
        history_list = list(history_frames)
        while len(history_list) < 2:
            history_list.append(history_list[-1] if history_list else current_rgb)
        images.append(Image.fromarray(history_list[-1]).convert("RGB"))
        images.append(Image.fromarray(history_list[-2]).convert("RGB"))
        return self.image_processor(images=images, return_tensors="pt")["pixel_values"]

    @torch.inference_mode()
    def _predict_compact_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[int, str]:
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=8,
            do_sample=False,
        )
        generated_text = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        action = parse_action_text(generated_text)
        if action is None:
            action = 0
        return action, generated_text

    @torch.inference_mode()
    def _predict_original_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[int, str]:
        action_vector = self.model.predict_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            unnorm_key=self.args.unnorm_key,
            do_sample=False,
        )
        action, rounded_action, distance = convert_original_action_vector_to_action(action_vector)
        generated_text = json.dumps(
            {
                "action_vector": np.asarray(action_vector, dtype=np.float32).round(4).tolist(),
                "rounded_action_vector": rounded_action.tolist(),
                "nearest_satnav_action": action,
                "nearest_distance": round(float(distance), 4),
            },
            ensure_ascii=False,
        )
        return action, generated_text

    @torch.inference_mode()
    def predict_action(self, instruction: str, history_frames: deque[np.ndarray], current_rgb: np.ndarray) -> tuple[int, str]:
        prompt_text = build_openfly_prompt(instruction)
        input_ids = self.tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        pixel_values = self._build_pixel_values(history_frames, current_rgb).unsqueeze(0).to(self.device, dtype=self.model.dtype)

        if self.action_format == ORIGINAL:
            return self._predict_original_action(input_ids, attention_mask, pixel_values)
        return self._predict_compact_action(input_ids, attention_mask, pixel_values)

    def evaluate_episode(self, env_wrapper: SatNavEnvWrapper, episode) -> dict[str, Any]:
        obs = env_wrapper.reset(episode)
        instruction = env_wrapper.get_instruction(episode)
        history_frames: deque[np.ndarray] = deque(maxlen=2)
        done = False
        step = 0
        raw_outputs: list[str] = []
        action_trace: list[dict[str, Any]] = []
        runtime_error = None

        while not done and step < env_wrapper.max_steps:
            current_rgb = env_wrapper.get_rgb(obs)
            try:
                action, generated_text = self.predict_action(instruction, history_frames, current_rgb)
            except Exception as exc:
                runtime_error = repr(exc)
                action = 0
                generated_text = f"[ERROR] {runtime_error}"

            raw_outputs.append(generated_text)
            action_trace.append(
                {
                    "step": step,
                    "action_id": int(action),
                    "action_name": ACTION_TO_NAME.get(int(action), "unknown"),
                    "raw_output": generated_text,
                }
            )
            obs, done = env_wrapper.step(action)
            history_frames.append(current_rgb)
            step += 1
            if action == 0:
                break

        metrics = env_wrapper.get_metrics()
        result = {
            "episode_id": getattr(episode, "episode_id", getattr(episode, "id", "")),
            "scene_id": getattr(episode, "scene_id", ""),
            "trajectory_type": getattr(episode, "trajectory_type", "unknown"),
            "instruction": instruction,
            "success": float(metrics.get("success", 0.0)),
            "spl": float(metrics.get("spl", 0.0)),
            "oracle_success": float(metrics.get("oracle_success", 0.0)),
            "distance_to_goal": float(metrics.get("distance_to_goal", ERROR_NE_PENALTY if runtime_error else 0.0)),
            "steps": int(step),
            "action": int(action_trace[-1]["action_id"]) if action_trace else 0,
            "parsed_action": action_trace[-1]["action_name"] if action_trace else ACTION_TO_NAME[0],
            "generated_text": action_trace[-1]["raw_output"] if action_trace else "",
            "action_trace": action_trace,
            "raw_outputs": raw_outputs,
        }
        if runtime_error:
            result["runtime_error"] = runtime_error
            result["distance_to_goal"] = ERROR_NE_PENALTY
        return result


def init_distributed(args) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")


def main():
    parser = argparse.ArgumentParser(description="OpenFly SatNav Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--satnav_config_path", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--action_format", type=str, default=AUTO, choices=[AUTO, COMPACT, ORIGINAL])
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=os.getenv("OPENFLY_EVAL_RUN_ID", "default"))
    parser.add_argument("--sync_timeout_sec", type=int, default=int(os.getenv("OPENFLY_EVAL_SYNC_TIMEOUT_SEC", "7200")))
    args = parser.parse_args()

    init_distributed(args)
    register_openfly_auto_classes()

    processor_source = _resolve_processor_source(args.model_path, args.processor_path)
    processor = AutoProcessor.from_pretrained(processor_source)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token or processor.tokenizer.unk_token
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=_resolve_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
    ).to(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    model.eval()
    setattr(model, "grid_size", getattr(model, "grid_size", 16))
    config_action_format = getattr(model.config, "action_format", None)
    args.action_format = resolve_action_format(args.action_format, fallback=config_action_format)
    if args.action_format == ORIGINAL and args.unnorm_key is None:
        args.unnorm_key = getattr(model.config, "satnav_unnorm_key", ORIGINAL_UNNORM_KEY)

    os.makedirs(args.output_path, exist_ok=True)
    result_file = os.path.join(args.output_path, "result.jsonl")

    config = OmegaConf.load(args.satnav_config_path)
    satnav_env = SatNavEnv(config)
    env_wrapper = SatNavEnvWrapper(satnav_env, config=config)
    dataset_config = config.DATASET if hasattr(config, "DATASET") else config
    dataset = SatNavDataset(dataset_config)
    episodes = dataset.episodes
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    my_episodes = episodes[args.rank :: args.world_size]
    evaluator = OpenFlySatNavEvaluator(args, model, processor, action_format=args.action_format)
    local_results = []

    for episode in my_episodes:
        result = evaluator.evaluate_episode(env_wrapper, episode)
        local_results.append(result)
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    env_wrapper.close()

    if args.world_size > 1:
        mark_rank_complete(args.output_path, args.run_id, args.rank, len(my_episodes), len(local_results))
        if args.rank == 0:
            wait_for_all_rank_markers(args.output_path, args.run_id, args.world_size, args.sync_timeout_sec)

    if args.rank == 0:
        merged_results = load_dedup_results(result_file)
        save_summary(merged_results, args.output_path, args)

    if args.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
