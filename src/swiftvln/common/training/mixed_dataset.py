# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Mixed VLN + QA Dataset

Provides MixedVLNQADataset that combines VLN trajectory data with QA data
for mixed training. This enables training VLN models with auxiliary QA tasks
to improve visual grounding and instruction following.
"""

import random
from typing import Any, Dict, Optional

from swift.utils import get_logger

logger = get_logger()


class MixedVLNQADataset:
    """
    Mixed dataset that combines VLN and QA data with configurable ratio.
    
    This dataset interleaves VLN and QA samples based on the specified ratio.
    For example, with qa_ratio=0.2, approximately 20% of samples will be QA
    and 80% will be VLN.
    
    The interleaving is done by cycling through both datasets and selecting
    based on ratio thresholds, ensuring a balanced mix throughout training.
    
    Usage:
        ```python
        mixed_dataset = MixedVLNQADataset(
            vln_dataset=vln_lazy_dataset,
            qa_dataset=qa_lazy_dataset,
            qa_ratio=0.2,  # 20% QA, 80% VLN
        )
        ```
    """
    
    def __init__(
        self, 
        vln_dataset, 
        qa_dataset, 
        qa_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            vln_dataset: VLN dataset (LazyLLMDataset or similar)
            qa_dataset: QA dataset (LazyLLMDataset or similar)
            qa_ratio: Ratio of QA samples (0.0-1.0), default 0.2 (20% QA)
            seed: Random seed for reproducibility
        """
        self.vln_dataset = vln_dataset
        self.qa_dataset = qa_dataset
        self.qa_ratio = max(0.0, min(1.0, qa_ratio))  # Clamp to [0, 1]
        self.seed = seed
        
        self.vln_len = len(vln_dataset)
        self.qa_len = len(qa_dataset)
        
        # Calculate total length based on ratio
        # If qa_ratio=0.2, we want 20% QA and 80% VLN
        # Total = VLN / (1 - qa_ratio) = VLN / 0.8
        # Or Total = QA / qa_ratio = QA / 0.2
        # Use the smaller to avoid over-sampling
        if self.qa_ratio > 0 and self.qa_ratio < 1:
            total_from_vln = int(self.vln_len / (1 - self.qa_ratio))
            total_from_qa = int(self.qa_len / self.qa_ratio)
            self._total_len = min(total_from_vln, total_from_qa)
        elif self.qa_ratio == 0:
            self._total_len = self.vln_len
        else:  # qa_ratio == 1
            self._total_len = self.qa_len
        
        # Build index mapping: (source_type, idx_in_source)
        self._build_index_mapping()
        
        logger.info(f"[MixedDataset] VLN: {self.vln_len}, QA: {self.qa_len}, "
                   f"Total: {self._total_len}, QA ratio: {self.qa_ratio:.1%}")
    
    def _build_index_mapping(self):
        """Build the index mapping for interleaved access."""
        self._index_map = []
        
        vln_indices = list(range(self.vln_len))
        qa_indices = list(range(self.qa_len))
        
        # Shuffle both with fixed seed for reproducibility
        rng = random.Random(self.seed)
        rng.shuffle(vln_indices)
        rng.shuffle(qa_indices)
        
        vln_ptr = 0
        qa_ptr = 0
        
        # Calculate interval for QA insertion
        # If qa_ratio=0.2, we want to insert QA every 5 samples (1/0.2=5)
        qa_interval = int(1 / self.qa_ratio) if self.qa_ratio > 0 else float('inf')
        
        for i in range(self._total_len):
            # Decide whether this sample should be QA based on ratio
            # Use deterministic interleaving for reproducibility
            is_qa_slot = (self.qa_ratio > 0 and 
                         (i % qa_interval == 0 or self.qa_ratio == 1))
            
            if is_qa_slot and qa_ptr < self.qa_len:
                self._index_map.append(('qa', qa_indices[qa_ptr]))
                qa_ptr += 1
            elif vln_ptr < self.vln_len:
                self._index_map.append(('vln', vln_indices[vln_ptr]))
                vln_ptr += 1
            elif qa_ptr < self.qa_len:
                # VLN exhausted, use remaining QA
                self._index_map.append(('qa', qa_indices[qa_ptr]))
                qa_ptr += 1
            else:
                # Both exhausted, should not happen with correct total_len
                break
        
        # Log actual distribution
        actual_qa_count = sum(1 for t, _ in self._index_map if t == 'qa')
        actual_vln_count = len(self._index_map) - actual_qa_count
        logger.info(f"[MixedDataset] Actual distribution: VLN={actual_vln_count}, QA={actual_qa_count}")
    
    def __len__(self) -> int:
        return self._total_len
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range [0, {self._total_len})")
        
        source, source_idx = self._index_map[idx]
        if source == 'qa':
            return self.qa_dataset[source_idx]
        else:
            return self.vln_dataset[source_idx]
    
    def get_inner_vln_dataset(self):
        """Get the inner VLN dataset (unwrapped)."""
        from swift.dataset import LazyLLMDataset
        if isinstance(self.vln_dataset, LazyLLMDataset):
            return self.vln_dataset.dataset
        return self.vln_dataset
    
    def get_inner_qa_dataset(self):
        """Get the inner QA dataset (unwrapped)."""
        from swift.dataset import LazyLLMDataset
        if isinstance(self.qa_dataset, LazyLLMDataset):
            return self.qa_dataset.dataset
        return self.qa_dataset
