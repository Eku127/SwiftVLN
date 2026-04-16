from __future__ import annotations

import os

from transformers import AutoModelForVision2Seq, AutoProcessor

from action_formats import (
    COMPACT,
    ORIGINAL,
    get_original_action_templates,
    get_original_norm_stats,
)
from backends.base import TrainBackendArtifacts, build_default_backend_meta, maybe_enable_gradient_checkpointing, resolve_dtype
from dataset.satnav_dataset import OpenFlyDataCollator
from openfly_core import register_openfly_auto_classes


def build_hf_train_backend(model_args, data_args, training_args) -> TrainBackendArtifacts:
    register_openfly_auto_classes()

    processor_source = model_args.processor_name_or_path or model_args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_source, cache_dir=model_args.cache_dir)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "low_cpu_mem_usage": True,
        "torch_dtype": resolve_dtype(model_args.torch_dtype),
    }
    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    setattr(model, "grid_size", model_args.grid_size)
    setattr(model.config, "grid_size", model_args.grid_size)
    setattr(model.config, "action_format", data_args.action_format)
    setattr(model.config, "satnav_unnorm_key", data_args.unnorm_key)
    setattr(model.config, "satnav_action_templates", get_original_action_templates())

    if data_args.action_format == ORIGINAL:
        norm_stats = get_original_norm_stats(data_args.unnorm_key)
        setattr(model.config, "norm_stats", norm_stats)
        setattr(model, "norm_stats", norm_stats)

    if training_args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model)

    data_collator = OpenFlyDataCollator(
        processor=processor,
        model_max_length=min(int(tokenizer.model_max_length), 2048),
        pad_token_id=tokenizer.pad_token_id,
        action_format=data_args.action_format,
    )

    if data_args.action_format == ORIGINAL:
        metadata = get_original_norm_stats(data_args.unnorm_key)
    else:
        metadata = {
            "format": "satnav_openfly_compact_actions",
            "action_format": COMPACT,
            "action_names": {
                "0": "stop",
                "1": "forward",
                "2": "left",
                "3": "right",
            },
            "grid_size": model_args.grid_size,
            "model_name_or_path": os.path.abspath(model_args.model_name_or_path),
        }

    backend_meta = build_default_backend_meta(
        backend_name="hf",
        model_name_or_path=model_args.model_name_or_path,
        processor_source=processor_source,
        action_format=data_args.action_format,
        grid_size=model_args.grid_size,
    )
    return TrainBackendArtifacts(
        backend_name="hf",
        processor=processor,
        model=model,
        data_collator=data_collator,
        metadata=metadata,
        backend_meta=backend_meta,
    )
