# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MonoVLN Training Entry Point

Custom SFT trainer for MonoVLN that:
1. Uses MonoVLNTrainArguments for compression-specific and sampling parameters
2. Creates MonoVLNDataset when detecting VLN data path
3. Sets compress_stride on the template
4. Integrates with ms-swift's standard training pipeline

Usage:
    python src/swiftvln/models/monovln/trainer.py --custom_register_path src/swiftvln/models/monovln ...
"""

import os
import sys
from typing import Optional, List, Union

# Add paths for direct script execution
if __name__ == '__main__':
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add ms-swift root to path
    _msswift_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
    if _msswift_root not in sys.path:
        sys.path.insert(0, _msswift_root)
    # Add vln directory to path
    _vln_dir = os.path.dirname(_current_dir)
    if _vln_dir not in sys.path:
        sys.path.insert(0, _vln_dir)
    # Add current directory to path
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)

from swift.llm.train.sft import SwiftSft
from swift.llm.dataset import LazyLLMDataset
from swift.utils import get_logger

# Use imports that work both as module and direct script
try:
    from .arguments import MonoVLNTrainArguments
    from .dataset import MonoVLNDataset
except ImportError:
    from arguments import MonoVLNTrainArguments
    from dataset import MonoVLNDataset

logger = get_logger()


class MonoVLNSft(SwiftSft):
    """
    MonoVLN SFT trainer with history frame compression.
    
    Key features:
    - Uses MonoVLNDataset with single-turn dialogue format
    - Uses <history_image>/<current_image> tokens
    - Sets compress_stride on the template
    - Compression applies in both training and inference
    """
    args_class = MonoVLNTrainArguments
    args: MonoVLNTrainArguments

    def _prepare_template(self):
        """
        Prepare template and set compression parameters.
        """
        super()._prepare_template()
        
        # Set compress_stride on the template
        if hasattr(self.template, 'compress_stride'):
            logger.info(f"[MonoVLN] Setting compress_stride={self.args.compress_stride} on template")
            self.template.compress_stride = self.args.compress_stride
            if hasattr(self.template, 'compressor'):
                self.template.compressor.stride = self.args.compress_stride
        else:
            logger.warning(f"[MonoVLN] Template {type(self.template).__name__} does not have compress_stride")

    def _get_dataset(self):
        """
        Get dataset, creating MonoVLNDataset for VLN data paths.
        
        Supports:
        - Single path: --dataset /path/to/dataset
        - Multiple paths (list): --dataset /path/to/dataset1 /path/to/dataset2
        - Multiple paths (comma): --dataset /path/to/dataset1,/path/to/dataset2
        """
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
                logger.info(f"[MonoVLN] Detected VLN dataset(s), creating MonoVLNDataset (paths={paths})")
                train_dataset = MonoVLNDataset(
                    data_path=data_path,
                    num_history=self.args.num_history,
                    num_future_steps=self.args.num_future_steps,
                    samples_per_episode=self.args.samples_per_episode,
                    max_samples=self.args.vln_max_samples,
                )
                logger.info(f"[MonoVLN] Dataset created: {len(train_dataset)} samples")
                logger.info(f"[MonoVLN] compress_stride={self.args.compress_stride}, samples_per_episode={self.args.samples_per_episode}")
                return train_dataset, None
        
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        """
        Skip HuggingFace preprocessing for MonoVLNDataset.
        """
        if isinstance(train_dataset, MonoVLNDataset):
            logger.info("[MonoVLN] Skipping HuggingFace preprocessing for MonoVLNDataset")
            return train_dataset, val_dataset
        
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        """
        Wrap MonoVLNDataset with LazyLLMDataset.
        """
        args = self.args
        template = self.template
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            
            if isinstance(dataset, MonoVLNDataset):
                logger.info("[MonoVLN] Wrapping MonoVLNDataset with LazyLLMDataset")
                datasets[i] = LazyLLMDataset(
                    dataset, 
                    template.encode, 
                    strict=args.strict, 
                    random_state=args.data_seed
                )
        
        # Handle non-MonoVLN datasets with parent logic
        has_other = any(
            d is not None and not isinstance(d, (MonoVLNDataset, LazyLLMDataset)) 
            for d in datasets
        )
        if has_other:
            datasets = super()._post_process_datasets(datasets)
        
        return datasets

    def _show_dataset(self, train_dataset, val_dataset):
        """
        Show dataset info.
        """
        inner_dataset = train_dataset
        if isinstance(train_dataset, LazyLLMDataset):
            inner_dataset = train_dataset.dataset
        
        if isinstance(inner_dataset, MonoVLNDataset):
            logger.info(f"[MonoVLN] Dataset: {len(inner_dataset)} samples")
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                logger.info(f"[MonoVLN] Sample keys: {sample.keys()}")
                logger.info(f"[MonoVLN] Number of messages: {len(sample.get('messages', []))}")
                logger.info(f"[MonoVLN] Number of images: {len(sample.get('images', []))}")
                logger.info(f"[MonoVLN] num_history_images: {sample.get('num_history_images', 0)}")
                # Show first message to verify format
                if sample.get('messages'):
                    first_msg = sample['messages'][0]
                    logger.info(f"[MonoVLN] User message preview: {first_msg.get('content', '')[:100]}...")
            return
        
        super()._show_dataset(train_dataset, val_dataset)


def train_main(args: Optional[Union[List[str], MonoVLNTrainArguments]] = None):
    """Main entry point for MonoVLN training."""
    return MonoVLNSft(args).main()


if __name__ == '__main__':
    train_main()
