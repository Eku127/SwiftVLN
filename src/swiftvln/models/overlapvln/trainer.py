# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Training Entry Point

Usage:
    python src/swiftvln/models/overlapvln/trainer.py --custom_register_path src/swiftvln/models/overlapvln ...
"""

import os
from typing import List, Optional, Union

import torch
from swift.utils import get_logger

from swiftvln.common.training.base_sft import BaseVLNSft
from swiftvln.models.overlapvln.arguments import OverlapVLNTrainArguments
from swiftvln.models.overlapvln.dataset import OverlapVLNDataset

logger = get_logger()


class OverlapVLNSft(BaseVLNSft):
    """OverlapVLN SFT trainer with overlap context and mixed training."""

    args_class = OverlapVLNTrainArguments
    args: OverlapVLNTrainArguments
    dataset_class = OverlapVLNDataset
    model_name = "OverlapVLN"

    def _prepare_template(self):
        """Prepare template and set compression/history processor parameters."""
        super()._prepare_template()

        # Critical: Update history_processor_type and recreate history_processor
        # because get_template() cannot pass custom parameters, so template uses defaults
        if hasattr(self.template, 'history_processor'):
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
            logger.info(f"[OverlapVLN] Configuring history processor:")
            logger.info(f"  - history_processor_type: {history_processor_type}")

            # Recreate history_processor with correct parameters
            self.template.history_processor_type = history_processor_type
            self.template.num_history = num_history
            self.template.log_base = log_base
            self.template.compress_stride = compress_stride
            self.template.use_tome = use_tome
            self.template.gtc_output_tokens = gtc_output_tokens
            self.template.gtc_temperature = gtc_temperature
            self.template.gtc_num_iterations = gtc_num_iterations

            self.template.history_processor = create_history_processor(
                processor_type=history_processor_type,
                compress_stride=compress_stride,
                compress_method='tome' if use_tome else 'pooling',
                num_history=num_history,
                log_base=log_base,
                output_tokens=gtc_output_tokens,
                temperature=gtc_temperature,
                num_iterations=gtc_num_iterations,
            )

            logger.info(f"  - Created: {self.template.history_processor.name}")

            if history_processor_type == 'per_frame':
                compress_method = "tome" if use_tome else "pool"
                logger.info(f"  - num_history: {num_history}, log_base: {log_base}")
                logger.info(f"  - compress: {compress_method}, stride: {compress_stride}")
                # Debug output
                if os.environ.get('OVERLAPVLN_DEBUG'):
                    logger.info(f"  [DEBUG] Per-frame configuration verified:")
                    logger.info(f"    -> num_history={num_history} (frames to sample)")
                    logger.info(
                        f"    -> log_base={log_base} "
                        f"({'uniform' if log_base == 1.0 else 'logarithmic'} sampling)"
                    )
                    logger.info(f"    -> compress_method={compress_method}")
                    logger.info(f"    -> compress_stride={compress_stride} ({compress_stride**2}x compression)")
            elif history_processor_type in ('gtc', 'segment_gtc'):
                logger.info(f"  - output_tokens: {gtc_output_tokens}")
                logger.info(f"  - temperature: {gtc_temperature}")
                logger.info(f"  - num_iterations: {gtc_num_iterations}")
        else:
            logger.warning(
                f"[OverlapVLN] Template {type(self.template).__name__} does not have history_processor"
            )

        # Configure embedding enhancement pipeline
        use_pixel_embed = getattr(self.args, 'use_pixel_embed', False)
        use_pose_embed = getattr(self.args, 'use_pose_embed', False)
        use_uav_adapter = getattr(self.args, 'use_uav_adapter', False)
        uav_adapter_path = getattr(self.args, 'uav_adapter_path', '')
        uav_adapter_type = getattr(self.args, 'uav_adapter_type', 'transformer_v1')
        uav_adapter_apply_scope = getattr(self.args, 'uav_adapter_apply_scope', 'all_images')
        pose_fusion_method = getattr(self.args, 'pose_fusion_method', 'additive')
        pose_norm_scale = getattr(self.args, 'pose_norm_scale', 100.0)
        self.template.use_pixel_embed = use_pixel_embed
        self.template.use_pose_embed = use_pose_embed
        self.template.use_uav_adapter = use_uav_adapter

        model = getattr(self, 'model', None)
        if model is not None:
            desired_enhancements = []
            if use_pixel_embed:
                desired_enhancements.append('pixel')
            if use_pose_embed:
                desired_enhancements.append('pose')
            if use_uav_adapter:
                desired_enhancements.append('uav')

            def _needs_rebuild_pipeline() -> bool:
                if not hasattr(model, 'embed_enhance') or model.embed_enhance is None:
                    return True
                if len(desired_enhancements) == 0:
                    return False
                if model.embed_enhance.is_empty:
                    return True
                return any(
                    name not in getattr(model.embed_enhance, 'enhancements', {})
                    for name in desired_enhancements
                )

            # Ensure embed_enhance pipeline exists on the model
            if _needs_rebuild_pipeline():
                from swiftvln.common.embedding_enhancement import create_embedding_pipeline

                embed_dim = model.config.hidden_size
                model.embed_enhance = create_embedding_pipeline(
                    embed_dim=embed_dim,
                    use_pixel_embed=use_pixel_embed,
                    use_pose_embed=use_pose_embed,
                    use_uav_adapter=use_uav_adapter,
                    pose_fusion=pose_fusion_method,
                    pose_norm_scale=pose_norm_scale,
                    uav_adapter_path=uav_adapter_path,
                    uav_adapter_type=uav_adapter_type,
                    uav_adapter_apply_scope=uav_adapter_apply_scope,
                )
                logger.info(f"[OverlapVLN] Rebuilt embed_enhance pipeline in trainer: {model.embed_enhance}")

            # Move to matching device/dtype
            if not model.embed_enhance.is_empty:
                target_dtype = model.visual.dtype if hasattr(model, 'visual') and hasattr(model.visual, 'dtype') else None
                try:
                    target_device = next(model.parameters()).device
                except (StopIteration, AttributeError, TypeError):
                    target_device = getattr(model, 'device', torch.device('cpu'))
                to_kwargs = {}
                if target_dtype is not None:
                    to_kwargs['dtype'] = target_dtype
                if target_device is not None:
                    to_kwargs['device'] = target_device
                if to_kwargs:
                    model.embed_enhance = model.embed_enhance.to(**to_kwargs)

                logger.info(f"[OverlapVLN] Embedding enhancement pipeline: {model.embed_enhance}")
                logger.info(f"  - Enhancements: {model.embed_enhance.enhancement_names}")
                if use_uav_adapter and uav_adapter_path and 'uav' in model.embed_enhance.enhancements:
                    resolved_path = model.embed_enhance.enhancements['uav'].load_external_checkpoint(
                        uav_adapter_path,
                        strict=True,
                    )
                    logger.info(f"[OverlapVLN] Loaded external UAV adapter from: {resolved_path}")

            # Backward compatibility: aliases without duplicate module registration.
            def _set_alias(alias_name: str, value) -> None:
                if hasattr(model, '_modules'):
                    model._modules.pop(alias_name, None)
                model.__dict__[alias_name] = value

            if use_pixel_embed and hasattr(model.embed_enhance, 'enhancements') and 'pixel' in model.embed_enhance.enhancements:
                _set_alias('pixel_embed', model.embed_enhance.enhancements['pixel'])
            elif not hasattr(model, 'pixel_embed'):
                _set_alias('pixel_embed', None)
            if use_pose_embed and hasattr(model.embed_enhance, 'enhancements') and 'pose' in model.embed_enhance.enhancements:
                _set_alias('pose_embed', model.embed_enhance.enhancements['pose'])
            elif not hasattr(model, 'pose_embed'):
                _set_alias('pose_embed', None)
            if use_uav_adapter and hasattr(model.embed_enhance, 'enhancements') and 'uav' in model.embed_enhance.enhancements:
                _set_alias('uav_adapter', model.embed_enhance.enhancements['uav'])
            elif not hasattr(model, 'uav_adapter'):
                _set_alias('uav_adapter', None)

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
        }

    def _log_dataset_created(self, dataset):
        super()._log_dataset_created(dataset)
        self._log(f"history_processor_type={self.args.history_processor_type}")
        if self.args.history_processor_type == 'per_frame':
            compress_method = "tome" if self.args.use_tome else "pool"
            self._log(
                f"Per-frame: h={self.args.num_history}, b={self.args.log_base}, "
                f"{compress_method}, s={self.args.compress_stride}"
            )
        elif self.args.history_processor_type in ('gtc', 'segment_gtc'):
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

    def _log_dataset_summary(self, dataset):
        self._log(f"system_prompt_setting: {dataset.system_prompt_setting}")
        if dataset.system_prompt_setting == "initial":
            self._log("[INITIAL] Initial view ENABLED: first frame (uncompressed) in system prompt")

    def _log_sample_details(self, sample, dataset):
        super()._log_sample_details(sample, dataset)
        if sample.get('messages'):
            first_msg = sample['messages'][0]
            has_history = '<image>' in first_msg.get('content', '')
            self._log(f"Has history images: {has_history}")
            self._log(f"num_history_images: {sample.get('num_history_images', 0)}")
            num_initial = sample.get('num_initial_images', 0)
            self._log(f"num_initial_images: {num_initial}")
            if num_initial > 0:
                sys_content = first_msg.get('content', '')
                has_initial_tag = 'initial observation' in sys_content
                self._log(f"[INITIAL] System prompt contains 'initial observation': {has_initial_tag}")
                image_count_in_sys = sys_content.count('<image>')
                self._log(f"[INITIAL] <image> tags in system prompt: {image_count_in_sys}")


def train_main(args: Optional[Union[List[str], OverlapVLNTrainArguments]] = None):
    """Main entry point for OverlapVLN training."""
    return OverlapVLNSft(args).main()


if __name__ == '__main__':
    train_main()
