# Copyright (c) Alibaba, Inc. and its affiliates.
"""
SwiftVLN Training Entry Point

Usage:
    python src/swiftvln/model/trainer.py --custom_register_path src/swiftvln/model ...
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

from swift.dataset import LazyLLMDataset
from swift.pipelines.train.sft import SwiftSft
from swift.utils import get_logger

from swiftvln.model.arguments import SwiftVLNTrainArguments
from swiftvln.model.dataset import SwiftVLNDataset

logger = get_logger()


class SwiftVLNSft(SwiftSft):
    """SwiftVLN SFT trainer with overlap context."""

    args_class = SwiftVLNTrainArguments
    args: SwiftVLNTrainArguments

    def _log(self, message: str) -> None:
        logger.info(f"[SwiftVLN] {message}")

    def _dataset_paths(self) -> tuple[Optional[str], List[str]]:
        raw_dataset = getattr(self.args, "dataset", None)
        if not raw_dataset:
            return None, []
        data_path = (
            ",".join(raw_dataset) if isinstance(raw_dataset, list) else raw_dataset
        )
        paths = [path.strip() for path in data_path.split(",") if path.strip()]
        return data_path, paths

    def _get_dataset(self):
        data_path, paths = self._dataset_paths()
        has_vln_annotations = any(
            os.path.isdir(path)
            and os.path.exists(os.path.join(path, "annotations.json"))
            for path in paths
        )
        if not (data_path and has_vln_annotations):
            return super()._get_dataset()

        self._log(f"Detected VLN dataset(s), creating SwiftVLNDataset (paths={paths})")
        dataset = SwiftVLNDataset(**self._build_dataset_kwargs(data_path))
        self._log_dataset_created(dataset)
        return dataset, None

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        if isinstance(train_dataset, SwiftVLNDataset):
            self._log("Skipping HuggingFace preprocessing for SwiftVLNDataset")
            return train_dataset, val_dataset
        return super()._encode_dataset(
            train_dataset,
            val_dataset,
            pre_process=pre_process,
        )

    def _post_process_datasets(self, datasets):
        for index, dataset in enumerate(datasets):
            if not isinstance(dataset, SwiftVLNDataset):
                continue
            self._log("Wrapping SwiftVLNDataset with LazyLLMDataset")
            datasets[index] = LazyLLMDataset(
                dataset,
                self.template.encode,
                strict=self.args.strict,
                random_state=self.args.data_seed,
            )

        has_other_dataset = any(
            dataset is not None
            and not isinstance(dataset, (SwiftVLNDataset, LazyLLMDataset))
            for dataset in datasets
        )
        if has_other_dataset:
            datasets = super()._post_process_datasets(datasets)
        return datasets

    def _show_dataset(self, train_dataset, val_dataset):
        inner_dataset = (
            train_dataset.dataset
            if isinstance(train_dataset, LazyLLMDataset)
            else train_dataset
        )
        if not isinstance(inner_dataset, SwiftVLNDataset):
            return super()._show_dataset(train_dataset, val_dataset)

        self._log(f"Dataset: {len(inner_dataset)} samples")
        self._log_dataset_summary(inner_dataset)
        if len(inner_dataset) > 0:
            self._log_sample_details(inner_dataset[0], inner_dataset)

    def _prepare_template(self):
        """Prepare template and set compression/history processor parameters."""
        super()._prepare_template()

        # Critical: Update history_processor_type and recreate history_processor
        # because get_template() cannot pass custom parameters, so template uses defaults
        if hasattr(self.template, "history_processor"):
            from swiftvln.common.history_processors import create_history_processor

            # Get parameters from args
            history_processor_type = self.args.history_processor_type
            num_history = self.args.num_history
            log_base = self.args.log_base
            compress_stride = self.args.compress_stride
            use_tome = self.args.use_tome
            gtc_output_tokens = self.args.gtc_output_tokens
            gtc_temperature = self.args.gtc_temperature
            gtc_num_iterations = self.args.gtc_num_iterations

            # Log the configuration
            logger.info("[SwiftVLN] Configuring history processor:")
            logger.info(f"  - history_processor_type: {history_processor_type}")
            logger.info(
                f"  - memory_method: {getattr(self.args, 'memory_method', 'history')}"
            )

            # Recreate history_processor with correct parameters
            self.template.history_processor_type = history_processor_type
            self.template.num_history = num_history
            self.template.log_base = log_base
            self.template.compress_stride = compress_stride
            self.template.use_tome = use_tome
            self.template.gtc_output_tokens = gtc_output_tokens
            self.template.gtc_temperature = gtc_temperature
            self.template.gtc_num_iterations = gtc_num_iterations
            self.template.memory_method = getattr(self.args, "memory_method", "history")
            self.template.map_global_side_m = getattr(
                self.args, "map_global_side_m", 1000.0
            )
            self.template.map_local_side_m = getattr(
                self.args, "map_local_side_m", 400.0
            )
            self.template.map_render_px = getattr(self.args, "map_render_px", 448)
            self.template.map_mask_method = getattr(
                self.args, "map_mask_method", "dilate20"
            )

            self.template.history_processor = create_history_processor(
                processor_type=history_processor_type,
                compress_stride=compress_stride,
                compress_method="tome" if use_tome else "pooling",
                num_history=num_history,
                log_base=log_base,
                output_tokens=gtc_output_tokens,
                temperature=gtc_temperature,
                num_iterations=gtc_num_iterations,
            )

            logger.info(f"  - Created: {self.template.history_processor.name}")

            if history_processor_type == "per_frame":
                compress_method = "tome" if use_tome else "pool"
                logger.info(f"  - num_history: {num_history}, log_base: {log_base}")
                logger.info(
                    f"  - compress: {compress_method}, stride: {compress_stride}"
                )
            elif history_processor_type in ("gtc", "segment_gtc"):
                logger.info(f"  - output_tokens: {gtc_output_tokens}")
                logger.info(f"  - temperature: {gtc_temperature}")
                logger.info(f"  - num_iterations: {gtc_num_iterations}")
        else:
            logger.warning(
                f"[SwiftVLN] Template {type(self.template).__name__} does not have history_processor"
            )

        # Configure embedding enhancement pipeline
        use_pose_embed = getattr(self.args, "use_pose_embed", False)
        use_uav_adapter = getattr(self.args, "use_uav_adapter", False)
        uav_adapter_path = getattr(self.args, "uav_adapter_path", "")
        uav_adapter_type = getattr(self.args, "uav_adapter_type", "transformer_v1")
        uav_adapter_apply_scope = getattr(
            self.args, "uav_adapter_apply_scope", "all_images"
        )
        pose_fusion_method = getattr(self.args, "pose_fusion_method", "additive")
        pose_norm_scale = getattr(self.args, "pose_norm_scale", 100.0)
        self.template.use_pose_embed = use_pose_embed
        self.template.use_uav_adapter = use_uav_adapter

        model = getattr(self, "model", None)
        if model is not None:
            from swiftvln.common.embedding_enhancement.runtime import (
                configure_embedding_enhancement,
            )

            configure_embedding_enhancement(
                model,
                use_pose_embed=use_pose_embed,
                use_uav_adapter=use_uav_adapter,
                uav_adapter_path=uav_adapter_path,
                uav_adapter_type=uav_adapter_type,
                uav_adapter_apply_scope=uav_adapter_apply_scope,
                pose_fusion_method=pose_fusion_method,
                pose_norm_scale=pose_norm_scale,
                clear_disabled_aliases=False,
                logger=logger,
            )

    def _build_dataset_kwargs(self, data_path: str):
        return {
            "data_path": data_path,
            "num_frames": self.args.num_frames,
            "num_history": self.args.num_history,
            "num_future_steps": self.args.num_future_steps,
            "use_random": self.args.use_random,
            "max_samples": self.args.vln_max_samples,
            "num_overlap": self.args.num_overlap,
            "env_type": self.args.vln_env_type,
            "history_processor_type": self.args.history_processor_type,
            "log_base": self.args.log_base,
            "system_prompt_setting": self.args.system_prompt_setting,
            "need_frame_poses": self.args.use_pose_embed,
            "memory_method": self.args.memory_method,
            "map_global_side_m": self.args.map_global_side_m,
            "map_local_side_m": self.args.map_local_side_m,
            "map_render_px": self.args.map_render_px,
            "map_mask_method": self.args.map_mask_method,
        }

    def _log_dataset_created(self, dataset):
        self._log(f"VLN Dataset: {len(dataset)} samples")
        self._log(f"memory_method={self.args.memory_method}")
        self._log(f"history_processor_type={self.args.history_processor_type}")
        if self.args.history_processor_type == "per_frame":
            compress_method = "tome" if self.args.use_tome else "pool"
            self._log(
                f"Per-frame: h={self.args.num_history}, b={self.args.log_base}, "
                f"{compress_method}, s={self.args.compress_stride}"
            )
        elif self.args.history_processor_type in ("gtc", "segment_gtc"):
            self._log(
                f"GTC: output_tokens={self.args.gtc_output_tokens}, "
                f"temperature={self.args.gtc_temperature}, "
                f"num_iterations={self.args.gtc_num_iterations}"
            )
        self._log(
            f"num_overlap={self.args.num_overlap}, "
            f"stride={self.args.num_frames - self.args.num_overlap}"
        )
        self._log(f"system_prompt_setting={self.args.system_prompt_setting}")
        if self.args.memory_method == "map":
            self._log(
                f"map: global={self.args.map_global_side_m}m, "
                f"local={self.args.map_local_side_m}m, "
                f"render={self.args.map_render_px}px, "
                f"mask={self.args.map_mask_method}"
            )

    def _log_dataset_summary(self, dataset):
        self._log(f"system_prompt_setting: {dataset.system_prompt_setting}")
        self._log(f"memory_method: {dataset.memory_method}")
        if dataset.system_prompt_setting == "initial":
            self._log(
                "[INITIAL] Initial view ENABLED: first frame (uncompressed) in system prompt"
            )

    def _log_sample_details(self, sample, dataset):
        self._log(f"Sample keys: {sample.keys()}")
        self._log(
            f"Messages: {len(sample.get('messages', []))}, "
            f"Images: {len(sample.get('images', []))}"
        )
        if sample.get("messages"):
            first_msg = sample["messages"][0]
            sys_content = first_msg.get("content", "")
            has_history = (
                "<history_memory>" in sys_content
                or "historical observations" in sys_content
                or "explored map memories" in sys_content
            )
            self._log(f"Has history images: {has_history}")
            self._log(f"num_history_images: {sample.get('num_history_images', 0)}")
            num_initial = sample.get("num_initial_images", 0)
            self._log(f"num_initial_images: {num_initial}")
            self._log(f"memory_method: {sample.get('memory_method', 'history')}")
            self._log(
                f"system_prompt tags: <history_memory>={sys_content.count('<history_memory>')}, "
                f"<image>={sys_content.count('<image>')}, map_phrase={'explored map memories' in sys_content}"
            )
            self._log(f"images total: {len(sample.get('images', []))}")
            self._log(f"frame_poses total: {len(sample.get('frame_poses', []))}")
            if num_initial > 0:
                has_initial_tag = "initial observation" in sys_content
                self._log(
                    f"[INITIAL] System prompt contains 'initial observation': {has_initial_tag}"
                )
                image_count_in_sys = sys_content.count("<image>")
                self._log(
                    f"[INITIAL] <image> tags in system prompt: {image_count_in_sys}"
                )


def train_main(args: Optional[Union[List[str], SwiftVLNTrainArguments]] = None):
    """Main entry point for SwiftVLN training."""
    return SwiftVLNSft(args).main()


if __name__ == "__main__":
    train_main()
