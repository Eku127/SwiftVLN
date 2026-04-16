import json
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, Trainer, TrainerCallback, TrainingArguments

from action_formats import (
    COMPACT,
    ORIGINAL,
    ORIGINAL_UNNORM_KEY,
    get_original_norm_stats,
    resolve_action_format,
)
from backends import HF_BACKEND, build_train_backend, patch_accelerate_optimizer_train_eval
from dataset.satnav_dataset import SatNavOpenFlyDataset


class ArtifactSaveCallback(TrainerCallback):
    def __init__(self, processor, metadata: dict, backend_meta: dict) -> None:
        self.processor = processor
        self.metadata = metadata
        self.backend_meta = backend_meta

    def _write_artifacts(self, output_dir: str) -> None:
        self.processor.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "backend_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.backend_meta, f, ensure_ascii=False, indent=2)

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._write_artifacts(checkpoint_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        self._write_artifacts(args.output_dir)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "OpenFly base model dir or HF snapshot dir"})
    backend: str = field(default=HF_BACKEND, metadata={"help": "OpenFly backend: hf or native"})
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

    patch_accelerate_optimizer_train_eval()

    if training_args.remove_unused_columns:
        training_args.remove_unused_columns = False

    action_format = resolve_action_format(data_args.action_format)
    data_args.action_format = action_format

    train_dataset = SatNavOpenFlyDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        action_format=action_format,
        max_episodes=data_args.max_episodes,
        max_samples=data_args.max_samples,
    )
    backend_artifacts = build_train_backend(model_args, data_args, training_args)

    os.makedirs(training_args.output_dir, exist_ok=True)
    if action_format == ORIGINAL:
        metadata = get_original_norm_stats(data_args.unnorm_key, num_transitions=len(train_dataset))
    else:
        metadata = backend_artifacts.metadata

    metadata["action_counts"] = train_dataset.action_counts
    metadata["num_samples"] = len(train_dataset)

    trainer = Trainer(
        model=backend_artifacts.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=backend_artifacts.data_collator,
        tokenizer=backend_artifacts.processor.tokenizer,
    )
    trainer.add_callback(ArtifactSaveCallback(backend_artifacts.processor, metadata, backend_artifacts.backend_meta))
    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    if not training_args.deepspeed:
        trainer.save_model(training_args.output_dir)
    backend_artifacts.processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
