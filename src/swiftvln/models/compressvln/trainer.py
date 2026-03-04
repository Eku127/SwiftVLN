# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Training Entry Point

Custom SFT trainer for CompressVLN that:
1. Uses CompressVLNTrainArguments for compression-specific parameters
2. Creates CompressVLNDataset when detecting VLN data path
3. Sets compress_stride on the template
4. Supports mixed training with VLN + QA data
5. Integrates with ms-swift's standard training pipeline

Usage:
    python src/swiftvln/models/compressvln/trainer.py --custom_register_path src/swiftvln/models/compressvln ...
"""

import os
from typing import Optional, List, Union

from swift.llm.train.sft import SwiftSft
from swift.llm.dataset import LazyLLMDataset
from swift.utils import get_logger

from swiftvln.models.compressvln.arguments import CompressVLNTrainArguments
from swiftvln.models.compressvln.dataset import CompressVLNDataset
from swiftvln.common import VLNMixedTrainingMixin, MixedVLNQADataset

logger = get_logger()


class CompressVLNSft(VLNMixedTrainingMixin, SwiftSft):
    """
    CompressVLN SFT trainer with history frame compression and mixed training support.
    
    Key features:
    - Uses CompressVLNDataset with <history_image>/<current_image> tokens
    - Sets compress_stride on the template
    - Supports mixed training with VLN + QA data
    - Compression applies in both training and inference
    """
    args_class = CompressVLNTrainArguments
    args: CompressVLNTrainArguments

    def _prepare_template(self):
        """Prepare template and set compression parameters."""
        super()._prepare_template()
        
        # Set compress_stride on the template
        if hasattr(self.template, 'compress_stride'):
            logger.info(f"[CompressVLN] Setting compress_stride={self.args.compress_stride} on template")
            self.template.compress_stride = self.args.compress_stride
            if hasattr(self.template, 'compressor'):
                self.template.compressor.stride = self.args.compress_stride
        else:
            logger.warning(f"[CompressVLN] Template {type(self.template).__name__} does not have compress_stride")

    def _get_dataset(self):
        """
        Get dataset, creating CompressVLNDataset for VLN data paths.
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
                logger.info(f"[CompressVLN] Detected VLN dataset(s), creating CompressVLNDataset (paths={paths})")
                vln_dataset = CompressVLNDataset(
                    data_path=data_path,
                    num_frames=self.args.num_frames,
                    num_history=self.args.num_history,
                    num_future_steps=self.args.num_future_steps,
                    use_random=self.args.use_random,
                    max_samples=self.args.vln_max_samples,
                    use_precomputed_features=self.args.use_precomputed_features,
                    feature_cache_dir=self.args.feature_cache_dir,
                    feature_cache_size=self.args.feature_cache_size,
                    env_type=self.args.vln_env_type,
                )
                logger.info(f"[CompressVLN] VLN Dataset: {len(vln_dataset)} samples")
                logger.info(f"[CompressVLN] compress_stride={self.args.compress_stride}")
                if self.args.use_precomputed_features:
                    logger.info(f"[CompressVLN] use_precomputed_features=True")
        
        # Load QA dataset using mixin method
        self._qa_dataset = self._load_qa_dataset()
        
        if vln_dataset is not None:
            if self._qa_dataset is not None:
                logger.info(f"[CompressVLN] Mixed training enabled: VLN + QA")
            return vln_dataset, None
        elif self._qa_dataset is not None:
            return self._qa_dataset, None
        
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        """Skip HuggingFace preprocessing for CompressVLNDataset."""
        if isinstance(train_dataset, CompressVLNDataset):
            logger.info("[CompressVLN] Skipping HuggingFace preprocessing for CompressVLNDataset")
            return train_dataset, val_dataset
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        """Wrap CompressVLNDataset with LazyLLMDataset and handle mixed training."""
        args = self.args
        template = self.template
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            
            if isinstance(dataset, CompressVLNDataset):
                logger.info("[CompressVLN] Wrapping CompressVLNDataset with LazyLLMDataset")
                vln_lazy = LazyLLMDataset(
                    dataset, 
                    template.encode, 
                    strict=args.strict, 
                    random_state=args.data_seed
                )
                
                # Try to wrap with QA using mixin method
                final_dataset, is_mixed = self._wrap_vln_with_qa(vln_lazy)
                datasets[i] = final_dataset
        
        # Handle non-CompressVLN datasets with parent logic
        has_other = any(
            d is not None and not isinstance(d, (CompressVLNDataset, LazyLLMDataset, MixedVLNQADataset)) 
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
        
        if isinstance(inner_dataset, CompressVLNDataset):
            logger.info(f"[CompressVLN] Dataset: {len(inner_dataset)} samples")
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                logger.info(f"[CompressVLN] Sample keys: {sample.keys()}")
                logger.info(f"[CompressVLN] Messages: {len(sample.get('messages', []))}, Images: {len(sample.get('images', []))}")
                if sample.get('messages'):
                    first_msg = sample['messages'][0]
                    has_history = '<image>' in first_msg.get('content', '')
                    logger.info(f"[CompressVLN] Has history images: {has_history}")
                    logger.info(f"[CompressVLN] num_history_images: {sample.get('num_history_images', 0)}")
            return
        
        super()._show_dataset(train_dataset, val_dataset)


def train_main(args: Optional[Union[List[str], CompressVLNTrainArguments]] = None):
    """Main entry point for CompressVLN training."""
    return CompressVLNSft(args).main()


if __name__ == '__main__':
    train_main()
