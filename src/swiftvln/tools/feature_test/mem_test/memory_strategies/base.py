# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base Memory Strategy for VLN Evaluation

This module defines the abstract base class for all memory strategies.
Each strategy implements a different way to construct memory tokens
from historical frames for VLN decision making.

Key concepts:
- Memory tokens: Compressed representation of historical observations
- History frames: Raw RGB frames from previous steps
- VIT features: Visual features extracted from frames
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mem_test_dir = os.path.dirname(_current_dir)
_vln_dir = os.path.dirname(os.path.dirname(_mem_test_dir))
if _vln_dir not in sys.path:
    sys.path.insert(0, _vln_dir)


@dataclass
class MemoryOutput:
    """
    Output from a memory strategy.
    
    Attributes:
        memory_tokens: Compressed memory tokens [num_tokens, hidden_size]
        raw_features: Original VIT features before compression [num_frames, tokens_per_frame, hidden_size]
        selected_indices: Indices of selected history frames
        metadata: Additional strategy-specific metadata
    """
    memory_tokens: torch.Tensor
    raw_features: Optional[torch.Tensor] = None
    selected_indices: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseMemoryStrategy(ABC):
    """
    Abstract base class for memory construction strategies.
    
    All memory strategies should inherit from this class and implement
    the required abstract methods.
    
    A memory strategy is responsible for:
    1. Selecting which historical frames to include
    2. Extracting VIT features from selected frames
    3. Compressing features into memory tokens
    
    The output memory tokens are used as input to the VLN model
    for decision making.
    """
    
    def __init__(
        self,
        num_history: int = 8,
        compress_stride: int = 2,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the memory strategy.
        
        Args:
            num_history: Number of history frames to use
            compress_stride: Compression stride for feature pooling
            device: Device to use for computation
            dtype: Data type for tensors
        """
        self.num_history = num_history
        self.compress_stride = compress_stride
        self.device = device
        self.dtype = dtype
        
        # VIT encoder will be set by evaluator
        self._vit_encoder = None
        self._processor = None
    
    def set_encoder(self, model, processor):
        """
        Set the VIT encoder and processor for feature extraction.
        
        Args:
            model: The VLN model (must have visual() method)
            processor: The image processor
        """
        self._vit_encoder = model
        self._processor = processor
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this memory strategy.
        
        This is used for logging and result identification.
        """
        pass
    
    @abstractmethod
    def select_history_frames(
        self,
        total_history: int,
    ) -> List[int]:
        """
        Select which history frames to include in memory.
        
        Args:
            total_history: Total number of available history frames
            
        Returns:
            List of frame indices to include (sorted)
        """
        pass
    
    @abstractmethod
    def compress_features(
        self,
        features: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress VIT features into memory tokens.
        
        Args:
            features: VIT features [num_tokens, hidden_size]
            grid_thw: Grid dimensions [t, h, w] after merge
            
        Returns:
            Compressed features [compressed_tokens, hidden_size]
        """
        pass
    
    def encode_frames(
        self,
        images: List[Image.Image],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode frames through VIT.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tuple of (list of features, list of grid_thw)
        """
        if self._vit_encoder is None or self._processor is None:
            raise RuntimeError("Encoder not set. Call set_encoder() first.")
        
        if not images:
            return [], []
        
        # Process images
        media_inputs = self._processor.image_processor(
            images=images, return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(self.device).type(self.dtype)
        image_grid_thw = media_inputs['image_grid_thw'].to(self.device)
        
        # Extract VIT features
        with torch.no_grad():
            all_vit_features = self._vit_encoder.visual(pixel_values, grid_thw=image_grid_thw)
        
        # Get merge_size from processor
        merge_size = getattr(self._processor.image_processor, 'merge_size', 2)
        merge_length = merge_size ** 2
        
        # Split features by image
        features_list = []
        grid_thw_list = []
        embed_idx = 0
        
        for i in range(len(images)):
            num_tokens = int(image_grid_thw[i].prod() // merge_length)
            img_features = all_vit_features[embed_idx:embed_idx + num_tokens]
            embed_idx += num_tokens
            features_list.append(img_features)
            grid_thw_list.append(image_grid_thw[i])
        
        return features_list, grid_thw_list
    
    def build_memory(
        self,
        history_frames: List[Image.Image],
        current_start_idx: int,
    ) -> MemoryOutput:
        """
        Build memory tokens from history frames.
        
        This is the main entry point for memory construction.
        
        Args:
            history_frames: All available history frames (as PIL Images)
            current_start_idx: The index where current window starts
            
        Returns:
            MemoryOutput containing memory tokens and metadata
        """
        total_history = len(history_frames)
        
        if total_history == 0:
            return MemoryOutput(
                memory_tokens=torch.empty(0, device=self.device, dtype=self.dtype),
                raw_features=None,
                selected_indices=[],
                metadata={'strategy': self.name, 'num_selected': 0}
            )
        
        # Select which frames to use
        selected_indices = self.select_history_frames(total_history)
        
        if not selected_indices:
            return MemoryOutput(
                memory_tokens=torch.empty(0, device=self.device, dtype=self.dtype),
                raw_features=None,
                selected_indices=[],
                metadata={'strategy': self.name, 'num_selected': 0}
            )
        
        # Get selected frames
        selected_frames = [history_frames[i] for i in selected_indices]
        
        # Encode through VIT
        features_list, grid_thw_list = self.encode_frames(selected_frames)
        
        # Store raw features for metrics computation
        raw_features = torch.stack([f for f in features_list]) if features_list else None
        
        # Compress each frame's features
        compressed_list = []
        for features, grid_thw in zip(features_list, grid_thw_list):
            # Adjust grid_thw for after spatial merge
            merge_size = getattr(self._processor.image_processor, 'merge_size', 2)
            grid_after_merge = grid_thw.clone()
            grid_after_merge[1] = grid_after_merge[1] // merge_size
            grid_after_merge[2] = grid_after_merge[2] // merge_size
            
            compressed = self.compress_features(features, grid_after_merge)
            compressed_list.append(compressed)
        
        # Concatenate all compressed features
        memory_tokens = torch.cat(compressed_list, dim=0) if compressed_list else \
            torch.empty(0, device=self.device, dtype=self.dtype)
        
        return MemoryOutput(
            memory_tokens=memory_tokens,
            raw_features=raw_features,
            selected_indices=selected_indices,
            metadata={
                'strategy': self.name,
                'num_selected': len(selected_indices),
                'total_history': total_history,
                'tokens_per_frame': [f.shape[0] for f in features_list] if features_list else [],
                'compressed_tokens_per_frame': [c.shape[0] for c in compressed_list] if compressed_list else [],
            }
        )
    
    def get_memory_features_for_metrics(
        self,
        memory_output: MemoryOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get features for computing evaluation metrics.
        
        Args:
            memory_output: Output from build_memory()
            
        Returns:
            Tuple of (memory_tokens, raw_features) for metrics computation
            - memory_tokens: Used for redundancy calculation
            - raw_features: Used for coverage calculation
        """
        return memory_output.memory_tokens, memory_output.raw_features
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"num_history={self.num_history}, "
                f"compress_stride={self.compress_stride})")
