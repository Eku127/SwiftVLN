# Copyright (c) Alibaba, Inc. and its affiliates.
"""
History Processors Module

A modular framework for different history information processing strategies
in Visual Language Navigation tasks.

## Quick Start

```python
from swiftvln.modeling.history import create_history_processor

# Per-frame compression with uniform sampling (default)
processor = create_history_processor('per_frame', compress_stride=2, num_history=8)

# Per-frame with logarithmic sampling (more recent frames)
processor = create_history_processor('per_frame', num_history=8, log_base=2.0)

# Global Token Clustering
processor = create_history_processor('gtc', output_tokens=512)
```

## Core Interface

All processors implement two methods:
- `get_output_token_count(num_frames, frame_infos)` -> int
- `process(frame_embeds_list, frame_grid_thws)` -> Tensor

## Available Processors

- `PerFrameCompressor`: Per-frame compression with flexible sampling
  - log_base=1.0: Uniform sampling (default)
  - log_base>1.0: Logarithmic sampling (more recent frames)
- `GlobalTokenClustering`: Cross-frame clustering (GTC)
- `SegmentGTC`: Segment-wise GTC (8 segments, preserves coarse temporal order)

## Adding New Processors

1. Create a new file in this directory (e.g., `segment.py`)
2. Inherit from `HistoryProcessor` and implement the two core methods
3. Register in `_PROCESSOR_REGISTRY` below
"""

from typing import Optional

from .base import HistoryProcessor, HistoryProcessorConfig
from .per_frame import PerFrameCompressor
from .gtc import GlobalTokenClustering
from .segment_gtc import SegmentGTC
from .compressor import HistoryTokenCompressor


# =============================================================================
# Processor Registry
# =============================================================================

_PROCESSOR_REGISTRY = {
    'per_frame': PerFrameCompressor,
    'perframe': PerFrameCompressor,
    'frame': PerFrameCompressor,
    'gtc': GlobalTokenClustering,
    'global_token_clustering': GlobalTokenClustering,
    'clustering': GlobalTokenClustering,
    'segment_gtc': SegmentGTC,
    'sgtc': SegmentGTC,
    'segmentgtc': SegmentGTC,
}


def create_history_processor(
    processor_type: str = 'per_frame',
    config: Optional[HistoryProcessorConfig] = None,
    **kwargs
) -> HistoryProcessor:
    """
    Create a history processor instance.
    
    This is the main entry point for creating processors.
    
    Args:
        processor_type: Processor type name
            - 'per_frame': Per-frame compression (default)
            - 'gtc': Global Token Clustering
            - 'segment_gtc': Segment-wise GTC
        config: Optional HistoryProcessorConfig (overrides kwargs)
        **kwargs: Processor-specific arguments
        
    Returns:
        HistoryProcessor instance
        
    Examples:
        >>> # Per-frame with uniform sampling (log_base=1.0)
        >>> processor = create_history_processor('per_frame', num_history=8)
        
        >>> # Per-frame with logarithmic sampling (more recent frames)
        >>> processor = create_history_processor('per_frame', num_history=8, log_base=2.0)
        
        >>> # Per-frame with ToMe compression
        >>> processor = create_history_processor('per_frame', compress_method='tome')
        
        >>> # GTC with 512 output tokens
        >>> processor = create_history_processor('gtc', output_tokens=512)
        
        >>> # Segment GTC (8 segments, 512 total tokens)
        >>> processor = create_history_processor('segment_gtc', output_tokens=512)
        
        >>> # Using config object
        >>> config = HistoryProcessorConfig(processor_type='gtc', output_tokens=512)
        >>> processor = create_history_processor(config=config)
    """
    # Extract parameters from config if provided
    if config is not None:
        processor_type = config.processor_type
        kwargs.update({
            'compress_stride': config.compress_stride,
            'compress_method': config.compress_method,
            'grid_size': config.grid_size,
            'num_history': config.num_history,
            'log_base': config.log_base,
            'output_tokens': config.output_tokens,
            'temperature': config.temperature,
            'num_iterations': config.num_iterations,
            'init_method': config.init_method,
        })
    
    processor_type = processor_type.lower()
    
    if processor_type not in _PROCESSOR_REGISTRY:
        available = list(set(_PROCESSOR_REGISTRY.values()))
        raise ValueError(
            f"Unknown processor_type: '{processor_type}'. "
            f"Available: {[cls.__name__ for cls in available]}"
        )
    
    processor_cls = _PROCESSOR_REGISTRY[processor_type]
    
    # Build kwargs based on processor type
    if processor_cls == PerFrameCompressor:
        return PerFrameCompressor(
            stride=kwargs.get('compress_stride', 2),
            method=kwargs.get('compress_method', 'pooling'),
            num_history=kwargs.get('num_history', 8),
            log_base=kwargs.get('log_base', 1.0),
            grid_size=kwargs.get('grid_size', 2),
        )
    elif processor_cls == GlobalTokenClustering:
        return GlobalTokenClustering(
            output_tokens=kwargs.get('output_tokens', 512),
            temperature=kwargs.get('temperature', 0.1),
            num_iterations=kwargs.get('num_iterations', 1),
            init_method=kwargs.get('init_method', 'uniform'),
        )
    elif processor_cls == SegmentGTC:
        return SegmentGTC(
            output_tokens=kwargs.get('output_tokens', 512),
            temperature=kwargs.get('temperature', 0.1),
            num_iterations=kwargs.get('num_iterations', 1),
        )
    else:
        raise ValueError(f"No factory logic for {processor_cls.__name__}")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core interface
    'HistoryProcessor',
    'HistoryProcessorConfig',
    # Factory
    'create_history_processor',
    # Implementations
    'PerFrameCompressor',
    'GlobalTokenClustering',
    'SegmentGTC',
    'HistoryTokenCompressor',
]
