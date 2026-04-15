import json
import os
from dataclasses import dataclass, field
from typing import Optional

import accelerate.optimizer
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, HfArgumentParser, Trainer, TrainerCallback, TrainingArguments

from action_formats import (
    COMPACT,
    ORIGINAL,
    ORIGINAL_UNNORM_KEY,
    get_original_action_templates,
    get_original_norm_stats,
    resolve_action_format,
)
from dataset.satnav_dataset import OpenFlyDataCollator, SatNavOpenFlyDataset
from openfly_core import register_openfly_auto_classes


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def _maybe_enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            return
        except Exception:
            pass
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "gradient_checkpointing_enable"):
        language_model.gradient_checkpointing_enable()


def _patch_accelerate_optimizer_train_eval() -> None:
    def _safe_train(self):
        train_fn = getattr(self.optimizer, "train", None)
        if callable(train_fn):
            return train_fn()
        return None

    def _safe_eval(self):
        eval_fn = getattr(self.optimizer, "eval", None)
        if callable(eval_fn):
            return eval_fn()
        return None

    accelerate.optimizer.AcceleratedOptimizer.train = _safe_train
    accelerate.optimizer.AcceleratedOptimizer.eval = _safe_eval


class ProcessorSaveCallback(TrainerCallback):
    def __init__(self, processor, metadata: dict) -> None:
        self.processor = processor
        self.metadata = metadata

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.processor.save_pretrained(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def on_train_begin(self, args, state, control, **kwargs):
        self.processor.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "OpenFly base model dir or HF snapshot dir"})
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional processor/tokenizer source dir; defaults to model_name_or_path"},
    )
    cache_dir: Optional[str] = None
    torch_dtype: str = "auto"
    grid_size: int = 16
    use_flash_attention_2: bool = False


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "SatNav trajectory_data/annotations.json"})
    image_folder: str = field(metadata={"help": "SatNav trajectory_data root"})
    action_format: str = field(default=COMPACT, metadata={"help": "OpenFly action format: compact or original"})
    unnorm_key: str = field(default=ORIGINAL_UNNORM_KEY, metadata={"help": "Un-normalization key for original action mode"})
    max_episodes: Optional[int] = None
    max_samples: Optional[int] = None


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _patch_accelerate_optimizer_train_eval()
    register_openfly_auto_classes()

    if training_args.remove_unused_columns:
        training_args.remove_unused_columns = False

    action_format = resolve_action_format(data_args.action_format)

    processor_source = model_args.processor_name_or_path or model_args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_source, cache_dir=model_args.cache_dir)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "low_cpu_mem_usage": True,
        "torch_dtype": _resolve_dtype(model_args.torch_dtype),
    }
    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    setattr(model, "grid_size", model_args.grid_size)
    setattr(model.config, "grid_size", model_args.grid_size)
    setattr(model.config, "action_format", action_format)
    setattr(model.config, "satnav_unnorm_key", data_args.unnorm_key)

    if action_format == ORIGINAL:
        norm_stats = get_original_norm_stats(data_args.unnorm_key)
        setattr(model.config, "norm_stats", norm_stats)
        setattr(model, "norm_stats", norm_stats)

    if training_args.gradient_checkpointing:
        _maybe_enable_gradient_checkpointing(model)

    train_dataset = SatNavOpenFlyDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        action_format=action_format,
        max_episodes=data_args.max_episodes,
        max_samples=data_args.max_samples,
    )
    data_collator = OpenFlyDataCollator(
        processor=processor,
        model_max_length=min(int(tokenizer.model_max_length), 2048),
        pad_token_id=tokenizer.pad_token_id,
        action_format=action_format,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    if action_format == ORIGINAL:
        metadata = get_original_norm_stats(data_args.unnorm_key, num_transitions=len(train_dataset))
    else:
        metadata = {
            "format": "satnav_openfly_compact_actions",
            "action_format": action_format,
            "action_names": {
                "0": "stop",
                "1": "forward",
                "2": "left",
                "3": "right",
            },
            "action_counts": train_dataset.action_counts,
            "num_samples": len(train_dataset),
            "grid_size": model_args.grid_size,
            "model_name_or_path": model_args.model_name_or_path,
        }

    setattr(model.config, "satnav_action_templates", get_original_action_templates())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.add_callback(ProcessorSaveCallback(processor, metadata))
    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    if not training_args.deepspeed:
        trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
