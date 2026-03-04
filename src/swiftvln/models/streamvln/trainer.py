# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Training Entry Point

Custom SFT trainer for StreamVLN that:
1. Uses StreamVLNTrainArguments for VLN-specific parameters
2. Creates StreamVLNDataset when detecting VLN data path
3. Supports mixed training with VLN + QA data
4. Integrates with ms-swift's standard training pipeline

Usage:
    python src/swiftvln/models/streamvln/trainer.py --custom_register_path src/swiftvln/models/streamvln ...
"""

import os
from typing import Optional, List, Union

from swift.llm.train.sft import SwiftSft
from swift.llm.dataset import LazyLLMDataset
from swift.utils import get_logger

from swiftvln.models.streamvln.arguments import StreamVLNTrainArguments
from swiftvln.models.streamvln.dataset import StreamVLNDataset
from swiftvln.common import VLNMixedTrainingMixin, MixedVLNQADataset

logger = get_logger()


class StreamVLNSft(VLNMixedTrainingMixin, SwiftSft):
    """
    StreamVLN SFT trainer with mixed training support.
    
    Inherits from VLNMixedTrainingMixin to support QA mixed training.
    """
    args_class = StreamVLNTrainArguments
    args: StreamVLNTrainArguments

    def _get_dataset(self):
        """
        Get dataset, creating StreamVLNDataset for VLN data paths.
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
                logger.info(f"[StreamVLN] Detected VLN dataset(s), creating StreamVLNDataset (paths={paths})")
                vln_dataset = StreamVLNDataset(
                    data_path=data_path,
                    num_frames=self.args.num_frames,
                    num_history=self.args.num_history,
                    num_future_steps=self.args.num_future_steps,
                    use_random=self.args.use_random,
                    max_samples=self.args.vln_max_samples,
                    env_type=self.args.vln_env_type,
                )
                logger.info(f"[StreamVLN] VLN Dataset: {len(vln_dataset)} samples")
        
        # Load QA dataset using mixin method
        self._qa_dataset = self._load_qa_dataset()
        
        if vln_dataset is not None:
            if self._qa_dataset is not None:
                logger.info(f"[StreamVLN] Mixed training enabled: VLN + QA")
            return vln_dataset, None
        elif self._qa_dataset is not None:
            # QA only (no VLN data)
            return self._qa_dataset, None
        
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        """Skip HuggingFace preprocessing for StreamVLNDataset."""
        if isinstance(train_dataset, StreamVLNDataset):
            logger.info("[StreamVLN] Skipping HuggingFace preprocessing for StreamVLNDataset")
            return train_dataset, val_dataset
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        """Wrap StreamVLNDataset with LazyLLMDataset and handle mixed training."""
        args = self.args
        template = self.template
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            
            if isinstance(dataset, StreamVLNDataset):
                logger.info("[StreamVLN] Wrapping StreamVLNDataset with LazyLLMDataset")
                vln_lazy = LazyLLMDataset(
                    dataset, 
                    template.encode, 
                    strict=args.strict, 
                    random_state=args.data_seed
                )
                
                # Try to wrap with QA using mixin method
                final_dataset, is_mixed = self._wrap_vln_with_qa(vln_lazy)
                datasets[i] = final_dataset
        
        # Handle non-StreamVLN datasets with parent logic
        has_other = any(
            d is not None and not isinstance(d, (StreamVLNDataset, LazyLLMDataset, MixedVLNQADataset)) 
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
        
        if isinstance(inner_dataset, StreamVLNDataset):
            logger.info(f"[StreamVLN] Dataset: {len(inner_dataset)} samples")
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                logger.info(f"[StreamVLN] Sample keys: {sample.keys()}")
                logger.info(f"[StreamVLN] Messages: {len(sample.get('messages', []))}, Images: {len(sample.get('images', []))}")
            return
        
        super()._show_dataset(train_dataset, val_dataset)


def train_main(args: Optional[Union[List[str], StreamVLNTrainArguments]] = None):
    """Main entry point for StreamVLN training."""
    return StreamVLNSft(args).main()


if __name__ == '__main__':
    train_main()
