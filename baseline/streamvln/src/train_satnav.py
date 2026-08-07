# Training entry point for StreamVLN fine-tuning on SatNav data.
#
# Rewritten from StreamVLN's streamvln/streamvln_train.py, keeping all
# model / training logic identical.  The only change is that
# make_supervised_data_module() uses SatNavActionDataset instead of
# VLNActionDataset.
#
# Supports two modes (controlled by --model_name_or_path):
#   Continue-training : pass the official StreamVLN checkpoint path
#   From-scratch      : pass lmms-lab/LLaVA-Video-7B-Qwen2
#
# Run via scripts/train_satnav.sh (torchrun + DeepSpeed ZeRO-2).

import sys
import os

if os.environ.get("STREAMVLN_OFFLINE", "true").lower() in ("1", "true", "yes", "on"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------
# Make StreamVLN packages importable.
# The streamvln-baseline conda env already has a .pth pointing here,
# but we add it explicitly for safety when running outside that env.
# ---------------------------------------------------------------
_STREAMVLN_ROOT = os.path.abspath(
    os.path.expanduser(os.environ.get("STREAMVLN_REPO", "../StreamVLN"))
)
if _STREAMVLN_ROOT not in sys.path:
    sys.path.insert(0, _STREAMVLN_ROOT)

# streamvln/model/stream_video_vln.py imports `from utils.utils import ...`
# which resolves to StreamVLN/streamvln/utils/utils.py, so that dir must be
# on sys.path as well.
_STREAMVLN_SUBPKG = os.path.join(_STREAMVLN_ROOT, "streamvln")
if _STREAMVLN_SUBPKG not in sys.path:
    sys.path.insert(0, _STREAMVLN_SUBPKG)

# Also ensure this baseline directory is on the path so our dataset module
# is importable as `dataset.satnav_action_dataset`.
_BASELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_ROOT not in sys.path:
    sys.path.insert(0, _BASELINE_ROOT)

import ast
import copy
import json
import logging
import pathlib
import re
import math
import time
import random
import yaml

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from functools import partial

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image, ImageFile

import transformers
import tokenizers
from packaging import version
from transformers import AutoConfig

from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import (
    process_highres_image,
    process_anyres_image,
    process_highres_image_crop_split,
    tokenizer_image_token,
)
from llava.utils import rank0_print

from streamvln.model.stream_video_vln import StreamVLNForCausalLM
from streamvln.utils.utils import (
    ANSWER_LIST,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_MEMORY_TOKEN,
    MEMORY_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN,
)
from streamvln.args import ModelArguments, DataArguments, TrainingArguments

# Our SatNav dataset
from dataset.satnav_action_dataset import SatNavActionDataset, make_satnav_collate_fn

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None
IS_TOKENIZER_GREATER_THAN_0_14 = (
    version.parse(tokenizers.__version__) >= version.parse("0.14")
)

try:
    from petrel_client.client import Client
    client = Client("~/petreloss.conf")
except ImportError:
    pass


# ---------------------------------------------------------------
# Utility functions (copied verbatim from streamvln_train.py)
# ---------------------------------------------------------------

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = [
        "vision_tower",
        "mm_projector",
        "mem_projector",
        "point_projector",
        "vision_resampler",
        "mem_resampler",
        "pointnet",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dumps to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    elif hasattr(trainer.args, "mm_tunable_parts") and (
        len(trainer.args.mm_tunable_parts.split(",")) == 1
        and (
            "mm_mlp_adapter" in trainer.args.mm_tunable_parts
            or "mm_vision_resampler" in trainer.args.mm_tunable_parts
        )
    ):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")

    if check_only_save_mm_adapter_tunnable:
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, "mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def safe_save_model_for_hf_trainer_fsdp(
    trainer: transformers.Trainer, output_dir: str
):
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.state_dict_type = "FULL_STATE_DICT"
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# ---------------------------------------------------------------
# Data module — uses SatNavActionDataset
# ---------------------------------------------------------------

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    vision_tower,
    data_args,
) -> Dict:
    """Create SatNav dataset and collator for supervised fine-tuning."""
    nav_dataset = SatNavActionDataset(
        tokenizer=tokenizer, data_args=data_args, task_id=0
    )
    train_dataset = nav_dataset
    rank0_print(f"len train_dataset: {len(train_dataset)}")

    data_collator = make_satnav_collate_fn(tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


# ---------------------------------------------------------------
# Model loader (copied verbatim from streamvln_train.py)
# ---------------------------------------------------------------

def get_model(model_args, training_args, data_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if (
        training_args.attn_implementation == "sdpa"
        and torch.__version__ < "2.1.2"
    ):
        raise ValueError(
            "The 'sdpa' attention implementation requires torch version 2.1.2 or higher."
        )

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if (
        model_args.use_pos_skipping is not None
        and model_args.pos_skipping_range is not None
    ):
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if (
        model_args.rope_scaling_factor is not None
        and model_args.rope_scaling_type is not None
    ):
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = (
                cfg_pretrained.max_position_embeddings
                * model_args.rope_scaling_factor
            )
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(
            cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
        )

    if (
        model_args.mm_spatial_pool_stride is not None
        and model_args.mm_spatial_pool_out_channels is not None
        and model_args.mm_spatial_pool_mode is not None
        and model_args.mm_resampler_type is not None
    ):
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config[
            "mm_spatial_pool_out_channels"
        ] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode
    if model_args.mm_spatial_pool_size is not None:
        overwrite_config["mm_spatial_pool_size"] = model_args.mm_spatial_pool_size

    if data_args.num_future_steps:
        overwrite_config["num_future_steps"] = data_args.num_future_steps
    if data_args.num_history:
        overwrite_config["num_history"] = data_args.num_history

    if model_args.mm_tunable_parts:
        overwrite_config["mm_tunable_parts"] = model_args.mm_tunable_parts

    # Ensure local vision tower path from launcher is honored (offline-safe).
    if model_args.vision_tower:
        overwrite_config["mm_vision_tower"] = model_args.vision_tower
        overwrite_config["vision_tower"] = model_args.vision_tower

    overwrite_config["mm_newline_position"] = model_args.mm_newline_position
    overwrite_config["mm_patch_merge_type"] = model_args.mm_patch_merge_type

    if overwrite_config:
        if cfg_pretrained is None:
            cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    model = StreamVLNForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs,
    )
    return model


# ---------------------------------------------------------------
# Main training function (copied verbatim from streamvln_train.py,
# only make_supervised_data_module call is replaced)
# ---------------------------------------------------------------

def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print("Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    model = get_model(
        model_args, training_args, data_args, bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    if (
        model_args.rope_scaling_factor is not None
        and model_args.rope_scaling_type is not None
    ):
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # ---- Tokenizer ----
    if (
        "mistral" in model_args.model_name_or_path.lower()
        or "mixtral" in model_args.model_name_or_path.lower()
        or "zephyr" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
        )
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        # Fallback: try to load anyway
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    # ---- Vision modules ----
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=None
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if (
                isinstance(data_args.image_grid_pinpoints, str)
                and "x" in data_args.image_grid_pinpoints
            ):
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception:
                    patch_size = data_args.image_processor.size["shortest_edge"]
                assert patch_size in [
                    224,
                    336,
                    384,
                    448,
                    512,
                ], "patch_size should be in [224, 336, 384, 448, 512]"
                matches = re.findall(
                    r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints
                )
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                grid_pinpoints = [
                    (i, j)
                    for i in range(range_start[0], range_end[0] + 1)
                    for j in range(range_start[1], range_end[1] + 1)
                ]
                data_args.image_grid_pinpoints = [
                    [dim * patch_size for dim in pair] for pair in grid_pinpoints
                ]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(
                    data_args.image_grid_pinpoints
                )

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

        # ---- Decide which parts to train ----
        if model_args.mm_tunable_parts is None:
            model.config.tune_mm_mlp_adapter = (
                training_args.tune_mm_mlp_adapter
            ) = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = (
                training_args.tune_mm_vision_resampler
            ) = model_args.tune_mm_vision_resampler
            if (
                model_args.tune_mm_mlp_adapter
                or model_args.tune_mm_vision_resampler
            ):
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = (
                training_args.freeze_mm_mlp_adapter
            )
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = (
                training_args.freeze_mm_vision_resampler
            )
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = (
                model_args.unfreeze_mm_vision_tower
            )
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)
        else:
            rank0_print(
                f"Using mm_tunable_parts: {model_args.mm_tunable_parts}"
            )
            model.config.mm_tunable_parts = (
                training_args.mm_tunable_parts
            ) = model_args.mm_tunable_parts
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if (
                "mm_vision_resampler" in tunable_parts
                and training_args.token_compression == "resampler"
            ):
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if (
                        "vision_tower" not in name
                        and "mm_projector" not in name
                        and "vision_resampler" not in name
                    ):
                        param.requires_grad_(True)
            if "mm_lora_layer" in tunable_parts:
                for name, param in model.named_parameters():
                    if "lora" in name:
                        param.requires_grad_(True)

        for name, param in model.named_parameters():
            if param.requires_grad:
                rank0_print(name)
        total_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in model.parameters()
        )
        trainable_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        rank0_print(f"Total parameters: ~{total_params / 1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params / 1e6:.2f} MB)")

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = (
            data_args.mm_use_im_start_end
        ) = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # ---- Data augmentation ----
    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.2, saturation=0.2),
                v2.RandomPosterize(bits=4),
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomAutocontrast(),
                v2.ToPILImage(),
            ]
        )
    else:
        data_args.transform_train = None

    # ---- Dataset ----
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, vision_tower=vision_tower, data_args=data_args
    )

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    "[WARNING] Attempting to use FSDP while {} parameters do not "
                    "require gradients: {}".format(
                        len(params_no_grad), params_no_grad
                    )
                )
            else:
                print(
                    "[WARNING] Attempting to use FSDP while {} parameters do not "
                    "require gradients: {}...(omitted)".format(
                        len(params_no_grad),
                        ", ".join(params_no_grad[:10]),
                    )
                )

    # ---- Trainer ----
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        if training_args.fsdp:
            safe_save_model_for_hf_trainer_fsdp(
                trainer=trainer, output_dir=training_args.output_dir
            )
        else:
            safe_save_model_for_hf_trainer(
                trainer=trainer, output_dir=training_args.output_dir
            )

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
