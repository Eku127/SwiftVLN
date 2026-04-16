from __future__ import annotations

import os

from action_formats import ORIGINAL, get_original_action_templates, get_original_norm_stats
from backends.base import TrainBackendArtifacts, maybe_enable_gradient_checkpointing, resolve_dtype
from dataset.satnav_dataset import OpenFlyDataCollator
from native_core import build_native_hf_model, resolve_native_checkpoint_path, resolve_native_processor_source


def build_native_train_backend(model_args, data_args, training_args) -> TrainBackendArtifacts:
    if data_args.action_format != ORIGINAL:
        raise ValueError("OpenFly native backend currently supports only action_format=original")

    processor_source = resolve_native_processor_source(model_args.model_name_or_path, model_args.processor_name_or_path)
    native_checkpoint_path, native_run_dir = resolve_native_checkpoint_path(model_args.model_name_or_path)

    model, processor, backend_meta = build_native_hf_model(
        model_name_or_path=model_args.model_name_or_path,
        processor_source=processor_source,
        cache_dir=model_args.cache_dir,
        grid_size=model_args.grid_size,
        unnorm_key=data_args.unnorm_key,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=resolve_dtype(model_args.torch_dtype),
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model.config.use_cache = False
    setattr(model, "grid_size", model_args.grid_size)
    setattr(model.config, "grid_size", model_args.grid_size)
    setattr(model.config, "action_format", data_args.action_format)
    setattr(model.config, "satnav_unnorm_key", data_args.unnorm_key)
    setattr(model.config, "satnav_action_templates", get_original_action_templates())
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

    metadata = get_original_norm_stats(data_args.unnorm_key)
    backend_meta.update(
        {
            "backend": "native",
            "native_checkpoint_path": os.path.abspath(native_checkpoint_path),
            "native_run_dir": os.path.abspath(native_run_dir),
            "model_name_or_path": os.path.abspath(model_args.model_name_or_path),
            "processor_source": os.path.abspath(processor_source),
        }
    )
    return TrainBackendArtifacts(
        backend_name="native",
        processor=processor,
        model=model,
        data_collator=data_collator,
        metadata=metadata,
        backend_meta=backend_meta,
    )
