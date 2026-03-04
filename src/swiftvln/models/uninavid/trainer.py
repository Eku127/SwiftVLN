# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Training Entry Point

Custom SFT trainer for UniNaVid that:
1. Uses UniNaVidTrainArguments for three-level memory and sampling parameters
2. Creates UniNaVidDataset when detecting VLN data path
3. Sets compression and similarity parameters on the template
4. Integrates with ms-swift's standard training pipeline

Usage:
    python src/swiftvln/models/uninavid/trainer.py --custom_register_path src/swiftvln/models/uninavid ...
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union

from swift.llm.train.sft import SwiftSft
from swift.llm import TrainArguments
from swift.llm.dataset import LazyLLMDataset
from swift.utils import get_logger

from swiftvln.models.uninavid.arguments import UniNaVidArguments
from swiftvln.models.uninavid.dataset import UniNaVidDataset

logger = get_logger()


@dataclass
class UniNaVidTrainArguments(TrainArguments, UniNaVidArguments):
    """
    Training arguments for UniNaVid, combining TrainArguments and UniNaVidArguments.
    
    Adds VLN-specific parameters to the standard training arguments.
    """
    
    # Dataset limiting
    vln_max_samples: int = field(
        default=0,
        metadata={
            "help": "Maximum number of VLN samples to use. 0 means use all."
        }
    )


class UniNaVidSft(SwiftSft):
    """
    UniNaVid SFT trainer with three-level memory compression.
    
    Key features:
    - Uses UniNaVidDataset with three-level frame classification
    - Uses <long_term_image>/<short_term_image>/<current_image> tokens
    - Sets compression and similarity parameters on the template
    - Supports dynamic sequence trimming (N placeholders → M merged tokens)
    """
    args_class = UniNaVidTrainArguments
    args: UniNaVidTrainArguments

    def _prepare_template(self):
        """
        Prepare template and set UniNaVid-specific parameters.
        """
        super()._prepare_template()
        
        # Set compression parameters on the template
        if hasattr(self.template, 'compress_stride'):
            logger.info(f"[UniNaVid] Setting compress_stride={self.args.compress_stride} on template")
            self.template.compress_stride = self.args.compress_stride
        
        if hasattr(self.template, 'short_term_frames'):
            logger.info(f"[UniNaVid] Setting short_term_frames={self.args.short_term_frames} on template")
            self.template.short_term_frames = self.args.short_term_frames
        
        if hasattr(self.template, 'similarity_threshold'):
            logger.info(f"[UniNaVid] Setting similarity_threshold={self.args.similarity_threshold} on template")
            self.template.similarity_threshold = self.args.similarity_threshold
        
        if not hasattr(self.template, 'long_term_image_token_id'):
            logger.warning(f"[UniNaVid] Template {type(self.template).__name__} may not be UniNaVidQwen25VLTemplate")

    def _get_dataset(self):
        """
        Get dataset, creating UniNaVidDataset for VLN data paths.
        
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
                logger.info(f"[UniNaVid] Detected VLN dataset(s), creating UniNaVidDataset (paths={paths})")
                train_dataset = UniNaVidDataset(
                    data_path=data_path,
                    short_term_frames=self.args.short_term_frames,
                    num_future_steps=self.args.num_future_steps,
                    samples_per_episode=self.args.samples_per_episode,
                    drop_frame_prob=self.args.drop_frame_prob,
                    max_samples=self.args.vln_max_samples if self.args.vln_max_samples > 0 else None,
                    image_resize_stride=self.args.image_resize_stride,
                    history_frame_stride=self.args.history_frame_stride,
                    use_precomputed_features=self.args.use_precomputed_features,
                    feature_cache_dir=self.args.feature_cache_dir,
                    feature_cache_size=self.args.feature_cache_size,
                )
                logger.info(f"[UniNaVid] Dataset created: {len(train_dataset)} samples")
                logger.info(f"[UniNaVid] short_term_frames={self.args.short_term_frames}, "
                           f"similarity_threshold={self.args.similarity_threshold}, "
                           f"compress_stride={self.args.compress_stride}")
                logger.info(f"[UniNaVid] samples_per_episode={self.args.samples_per_episode}, "
                           f"drop_frame_prob={self.args.drop_frame_prob}, "
                           f"image_resize_stride={self.args.image_resize_stride}, "
                           f"history_frame_stride={self.args.history_frame_stride}")
                if self.args.use_precomputed_features:
                    logger.info(f"[UniNaVid] Using precomputed features from {self.args.feature_cache_dir or 'auto-detected'}")
                    logger.info(f"[UniNaVid] Feature cache size: {self.args.feature_cache_size} episodes")
                return train_dataset, None
        
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        """
        Skip HuggingFace preprocessing for UniNaVidDataset.
        """
        if isinstance(train_dataset, UniNaVidDataset):
            logger.info("[UniNaVid] Skipping HuggingFace preprocessing for UniNaVidDataset")
            return train_dataset, val_dataset
        
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        """
        Wrap UniNaVidDataset with LazyLLMDataset.
        """
        args = self.args
        template = self.template
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            
            if isinstance(dataset, UniNaVidDataset):
                logger.info("[UniNaVid] Wrapping UniNaVidDataset with LazyLLMDataset")
                datasets[i] = LazyLLMDataset(
                    dataset, 
                    template.encode, 
                    strict=args.strict, 
                    random_state=args.data_seed
                )
        
        # Handle non-UniNaVid datasets with parent logic
        has_other = any(
            d is not None and not isinstance(d, (UniNaVidDataset, LazyLLMDataset)) 
            for d in datasets
        )
        if has_other:
            datasets = super()._post_process_datasets(datasets)
        
        return datasets

    def _show_dataset(self, train_dataset, val_dataset):
        """
        Show dataset info with UniNaVid-specific details.
        """
        inner_dataset = train_dataset
        if isinstance(train_dataset, LazyLLMDataset):
            inner_dataset = train_dataset.dataset
        
        if isinstance(inner_dataset, UniNaVidDataset):
            logger.info(f"[UniNaVid] Dataset: {len(inner_dataset)} samples")
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                logger.info(f"[UniNaVid] Sample keys: {sample.keys()}")
                logger.info(f"[UniNaVid] Number of messages: {len(sample.get('messages', []))}")
                logger.info(f"[UniNaVid] Number of images: {len(sample.get('images', []))}")
                logger.info(f"[UniNaVid] num_long_term_images: {sample.get('num_long_term_images', 0)}")
                logger.info(f"[UniNaVid] num_short_term_images: {sample.get('num_short_term_images', 0)}")
                # Show first message to verify format
                if sample.get('messages'):
                    first_msg = sample['messages'][0]
                    content_preview = first_msg.get('content', '')[:200]
                    logger.info(f"[UniNaVid] User message preview: {content_preview}...")
            return
        
        super()._show_dataset(train_dataset, val_dataset)


def train_main(args: Optional[Union[List[str], UniNaVidTrainArguments]] = None):
    """Main entry point for UniNaVid training."""
    return UniNaVidSft(args).main()


if __name__ == '__main__':
    train_main()
