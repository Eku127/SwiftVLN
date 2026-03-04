# Copyright (c) Alibaba, Inc. and its affiliates.
"""
OverlapVLN Training Entry Point

Custom SFT trainer for OverlapVLN that:
1. Uses OverlapVLNTrainArguments for compression-specific parameters
2. Creates OverlapVLNDataset when detecting VLN data path
3. Sets compress_stride on the template
4. Supports mixed training with VLN + QA data
5. Integrates with ms-swift's standard training pipeline

Usage:
    python src/swiftvln/models/overlapvln/trainer.py --custom_register_path src/swiftvln/models/overlapvln ...
"""

import os
from typing import Optional, List, Union

import torch

from swift.llm.train.sft import SwiftSft
from swift.llm.dataset import LazyLLMDataset
from swift.utils import get_logger

from swiftvln.models.overlapvln.arguments import OverlapVLNTrainArguments
from swiftvln.models.overlapvln.dataset import OverlapVLNDataset
from swiftvln.common import VLNMixedTrainingMixin, MixedVLNQADataset

logger = get_logger()


class OverlapVLNSft(VLNMixedTrainingMixin, SwiftSft):
    """
    OverlapVLN SFT trainer with history frame compression and mixed training support.
    
    Key features:
    - Uses OverlapVLNDataset with <history_image>/<current_image> tokens
    - Sets compress_stride on the template
    - Supports mixed training with VLN + QA data
    - Compression applies in both training and inference
    """
    args_class = OverlapVLNTrainArguments
    args: OverlapVLNTrainArguments

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
                    logger.info(f"    -> log_base={log_base} ({'uniform' if log_base == 1.0 else 'logarithmic'} sampling)")
                    logger.info(f"    -> compress_method={compress_method}")
                    logger.info(f"    -> compress_stride={compress_stride} ({compress_stride**2}x compression)")
            elif history_processor_type in ('gtc', 'segment_gtc'):
                logger.info(f"  - output_tokens: {gtc_output_tokens}")
                logger.info(f"  - temperature: {gtc_temperature}")
                logger.info(f"  - num_iterations: {gtc_num_iterations}")
        else:
            logger.warning(f"[OverlapVLN] Template {type(self.template).__name__} does not have history_processor")
        
        # Configure embedding enhancement pipeline
        # The pipeline is primarily created in model.py. However, during training
        # the swift framework may not forward use_pixel_embed to the model loader.
        # So we have a fallback here to populate the pipeline if needed.
        use_pixel_embed = getattr(self.args, 'use_pixel_embed', False)
        use_pose_embed = getattr(self.args, 'use_pose_embed', False)
        pose_fusion_method = getattr(self.args, 'pose_fusion_method', 'additive')
        pose_norm_scale = getattr(self.args, 'pose_norm_scale', 100.0)
        self.template.use_pixel_embed = use_pixel_embed
        self.template.use_pose_embed = use_pose_embed
        
        model = getattr(self, 'model', None)
        if model is not None:
            desired_enhancements = []
            if use_pixel_embed:
                desired_enhancements.append('pixel')
            if use_pose_embed:
                desired_enhancements.append('pose')

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
                    pose_fusion=pose_fusion_method,
                    pose_norm_scale=pose_norm_scale,
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

    def _get_dataset(self):
        """
        Get dataset, creating OverlapVLNDataset for VLN data paths.
        Also loads QA dataset if configured for mixed training.
        """
        vln_dataset = None
        
        if self.args.dataset:
            # Handle multiple paths
            if isinstance(self.args.dataset, list):
                data_path = ','.join(self.args.dataset)
            else:
                data_path = self.args.dataset
            
            # Check if at least one path contains annotations.json
            paths = [p.strip() for p in data_path.split(',')]
            has_vln_data = any(
                os.path.isdir(p) and os.path.exists(os.path.join(p, 'annotations.json'))
                for p in paths
            )
            
            if has_vln_data:
                logger.info(f"[OverlapVLN] Detected VLN dataset(s), creating OverlapVLNDataset (paths={paths})")
                vln_dataset = OverlapVLNDataset(
                    data_path=data_path,
                    num_frames=self.args.num_frames,
                    num_history=self.args.num_history,
                    num_future_steps=self.args.num_future_steps,
                    use_random=self.args.use_random,
                    max_samples=self.args.vln_max_samples,
                    num_overlap=self.args.num_overlap,
                    env_type=self.args.vln_env_type,
                    history_processor_type=self.args.history_processor_type,
                    log_base=self.args.log_base,
                    system_prompt_setting=self.args.system_prompt_setting,
                )
                logger.info(f"[OverlapVLN] VLN Dataset: {len(vln_dataset)} samples")
                logger.info(f"[OverlapVLN] history_processor_type={self.args.history_processor_type}")
                if self.args.history_processor_type == 'per_frame':
                    compress_method = "tome" if self.args.use_tome else "pool"
                    logger.info(f"[OverlapVLN] Per-frame: h={self.args.num_history}, b={self.args.log_base}, "
                               f"{compress_method}, s={self.args.compress_stride}")
                elif self.args.history_processor_type in ('gtc', 'segment_gtc'):
                    logger.info(f"[OverlapVLN] GTC: output_tokens={self.args.gtc_output_tokens}, "
                               f"temperature={self.args.gtc_temperature}, "
                               f"num_iterations={self.args.gtc_num_iterations}")
                logger.info(f"[OverlapVLN] num_overlap={self.args.num_overlap}, "
                           f"stride={self.args.num_frames - self.args.num_overlap}")
                logger.info(f"[OverlapVLN] system_prompt_setting={self.args.system_prompt_setting}")
        
        # Load QA dataset using mixin method
        self._qa_dataset = self._load_qa_dataset()
        
        if vln_dataset is not None:
            if self._qa_dataset is not None:
                logger.info(f"[OverlapVLN] Mixed training enabled: VLN + QA")
            return vln_dataset, None
        elif self._qa_dataset is not None:
            return self._qa_dataset, None
        
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        """Skip HuggingFace preprocessing for OverlapVLNDataset."""
        if isinstance(train_dataset, OverlapVLNDataset):
            logger.info("[OverlapVLN] Skipping HuggingFace preprocessing for OverlapVLNDataset")
            return train_dataset, val_dataset
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        """Wrap OverlapVLNDataset with LazyLLMDataset and handle mixed training."""
        args = self.args
        template = self.template
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            
            if isinstance(dataset, OverlapVLNDataset):
                logger.info("[OverlapVLN] Wrapping OverlapVLNDataset with LazyLLMDataset")
                vln_lazy = LazyLLMDataset(
                    dataset, 
                    template.encode, 
                    strict=args.strict, 
                    random_state=args.data_seed
                )
                
                # Try to wrap with QA using mixin method
                final_dataset, is_mixed = self._wrap_vln_with_qa(vln_lazy)
                datasets[i] = final_dataset
        
        # Handle non-OverlapVLN datasets with parent logic
        has_other = any(
            d is not None and not isinstance(d, (OverlapVLNDataset, LazyLLMDataset, MixedVLNQADataset)) 
            for d in datasets
        )
        if has_other:
            datasets = super()._post_process_datasets(datasets)
        
        return datasets

    def _show_dataset(self, train_dataset, val_dataset):
        """Show dataset info."""
        # Handle MixedVLNQADataset
        if isinstance(train_dataset, MixedVLNQADataset):
            self._show_mixed_dataset_info(train_dataset)
            return
        
        inner_dataset = train_dataset
        if isinstance(train_dataset, LazyLLMDataset):
            inner_dataset = train_dataset.dataset
        
        if isinstance(inner_dataset, OverlapVLNDataset):
            logger.info(f"[OverlapVLN] Dataset: {len(inner_dataset)} samples")
            logger.info(f"[OverlapVLN] system_prompt_setting: {inner_dataset.system_prompt_setting}")
            if inner_dataset.system_prompt_setting == "initial":
                logger.info(f"[OverlapVLN] [INITIAL] Initial view ENABLED: first frame (uncompressed) in system prompt")
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                logger.info(f"[OverlapVLN] Sample keys: {sample.keys()}")
                logger.info(f"[OverlapVLN] Messages: {len(sample.get('messages', []))}, "
                           f"Images: {len(sample.get('images', []))}")
                if sample.get('messages'):
                    first_msg = sample['messages'][0]
                    has_history = '<image>' in first_msg.get('content', '')
                    logger.info(f"[OverlapVLN] Has history images: {has_history}")
                    logger.info(f"[OverlapVLN] num_history_images: {sample.get('num_history_images', 0)}")
                    num_initial = sample.get('num_initial_images', 0)
                    logger.info(f"[OverlapVLN] num_initial_images: {num_initial}")
                    if num_initial > 0:
                        sys_content = first_msg.get('content', '')
                        has_initial_tag = 'initial observation' in sys_content
                        logger.info(f"[OverlapVLN] [INITIAL] System prompt contains 'initial observation': {has_initial_tag}")
                        # Count <image> tags in system prompt
                        image_count_in_sys = sys_content.count('<image>')
                        logger.info(f"[OverlapVLN] [INITIAL] <image> tags in system prompt: {image_count_in_sys}")
            return
        
        super()._show_dataset(train_dataset, val_dataset)


def train_main(args: Optional[Union[List[str], OverlapVLNTrainArguments]] = None):
    """Main entry point for OverlapVLN training."""
    return OverlapVLNSft(args).main()


if __name__ == '__main__':
    train_main()
