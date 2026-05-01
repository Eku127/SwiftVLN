# Copyright (c) Alibaba, Inc. and its affiliates.
"""
VLN Trainer Mixin for Mixed Training

Provides mixin class that adds QA mixed training capabilities to VLN trainers.
This can be mixed into any VLN trainer (StreamVLN, CompressVLN, OverlapVLN) to
enable training with both VLN and QA data.
"""

import os
from typing import Optional, Tuple, Any

from swift.dataset import LazyLLMDataset, load_dataset
from swift.utils import get_logger

from .mixed_dataset import MixedVLNQADataset

logger = get_logger()


class VLNMixedTrainingMixin:
    """
    Mixin class that provides QA mixed training capabilities for VLN trainers.
    
    This mixin provides:
    - Loading QA dataset from jsonl file
    - Creating MixedVLNQADataset that interleaves VLN and QA data
    - Proper dataset info display
    
    Usage:
        ```python
        class MyVLNSft(VLNMixedTrainingMixin, SwiftSft):
            # The mixin methods will be used automatically
            pass
        ```
    
    Note: This mixin expects the trainer to have:
    - self.args with qa_dataset, qa_ratio, qa_max_samples attributes
    - self.template for encoding
    """
    
    # Store QA dataset for later processing
    _qa_dataset: Optional[Any] = None
    
    def _load_qa_dataset(self) -> Optional[Any]:
        """
        Load QA dataset if configured.
        
        Returns:
            Loaded QA dataset or None if not configured.
        """
        if not hasattr(self.args, 'qa_dataset') or not self.args.qa_dataset:
            return None
        
        logger.info(f"[VLN] Loading QA dataset from: {self.args.qa_dataset}")
        
        if not os.path.exists(self.args.qa_dataset):
            logger.warning(f"[VLN] QA dataset not found: {self.args.qa_dataset}")
            return None
        
        dataset_kwargs = self.args.get_dataset_kwargs()
        qa_dataset, _ = load_dataset(
            self.args.qa_dataset,
            split_dataset_ratio=0.,  # No validation split for QA
            shuffle=True,
            **dataset_kwargs
        )
        
        # Limit QA samples if specified
        qa_max = getattr(self.args, 'qa_max_samples', None)
        if qa_max and qa_max > 0:
            if len(qa_dataset) > qa_max:
                qa_dataset = qa_dataset.shuffle(seed=42).select(range(qa_max))
                logger.info(f"[VLN] QA Dataset limited to {qa_max} samples")
        
        logger.info(f"[VLN] QA Dataset loaded: {len(qa_dataset)} samples")
        return qa_dataset
    
    def _create_mixed_dataset(
        self, 
        vln_dataset: LazyLLMDataset, 
        qa_dataset: LazyLLMDataset,
    ) -> MixedVLNQADataset:
        """
        Create a mixed dataset from VLN and QA datasets.
        
        Args:
            vln_dataset: Encoded VLN dataset (LazyLLMDataset)
            qa_dataset: Encoded QA dataset (LazyLLMDataset)
            
        Returns:
            MixedVLNQADataset instance
        """
        qa_ratio = getattr(self.args, 'qa_ratio', 0.2)
        
        mixed_dataset = MixedVLNQADataset(
            vln_dataset=vln_dataset,
            qa_dataset=qa_dataset,
            qa_ratio=qa_ratio,
        )
        
        logger.info(f"[VLN] Mixed dataset created: {len(mixed_dataset)} total samples")
        logger.info(f"[VLN]   VLN: {mixed_dataset.vln_len}, QA: {mixed_dataset.qa_len}")
        logger.info(f"[VLN]   QA ratio: {qa_ratio:.1%}")
        
        return mixed_dataset
    
    def _wrap_vln_with_qa(
        self, 
        vln_lazy_dataset: LazyLLMDataset,
    ) -> Tuple[Any, bool]:
        """
        Wrap VLN dataset with QA data if QA dataset is available.
        
        This method checks if a QA dataset was loaded earlier and creates
        a MixedVLNQADataset if so.
        
        Args:
            vln_lazy_dataset: Encoded VLN dataset wrapped in LazyLLMDataset
            
        Returns:
            Tuple of (final_dataset, is_mixed)
            - final_dataset: Either MixedVLNQADataset or original vln_lazy_dataset
            - is_mixed: True if mixed with QA, False otherwise
        """
        # Check if QA dataset was stored earlier
        qa_dataset = getattr(self, '_qa_dataset', None)
        
        if qa_dataset is None:
            return vln_lazy_dataset, False
        
        # Encode QA dataset
        logger.info("[VLN] Encoding QA dataset for mixed training")
        qa_lazy_dataset = LazyLLMDataset(
            qa_dataset,
            self.template.encode,
            strict=self.args.strict,
            random_state=self.args.data_seed
        )
        
        # Create mixed dataset
        mixed_dataset = self._create_mixed_dataset(vln_lazy_dataset, qa_lazy_dataset)
        
        # Clear stored QA dataset
        self._qa_dataset = None
        
        return mixed_dataset, True
    
    def _show_mixed_dataset_info(self, dataset: MixedVLNQADataset):
        """
        Show information about a mixed dataset.
        
        Args:
            dataset: MixedVLNQADataset instance
        """
        logger.info(f"[VLN] Mixed Dataset: {len(dataset)} total samples")
        logger.info(f"[VLN]   VLN: {dataset.vln_len} samples")
        logger.info(f"[VLN]   QA: {dataset.qa_len} samples")
        logger.info(f"[VLN]   QA ratio: {dataset.qa_ratio:.1%}")
        
        # Show a VLN sample info
        vln_inner = dataset.get_inner_vln_dataset()
        if hasattr(vln_inner, '__len__') and len(vln_inner) > 0:
            try:
                sample = vln_inner[0]
                logger.info(f"[VLN] VLN sample keys: {sample.keys()}")
                logger.info(f"[VLN] VLN messages: {len(sample.get('messages', []))}, "
                           f"images: {len(sample.get('images', []))}")
            except Exception as e:
                logger.warning(f"[VLN] Could not display VLN sample: {e}")
