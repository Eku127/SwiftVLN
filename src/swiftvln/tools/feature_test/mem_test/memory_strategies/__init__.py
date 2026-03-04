# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Memory Strategies for VLN Evaluation

This module provides different memory construction strategies for
evaluating how history information affects VLN decision quality.

Available Strategies:
- UniformMemoryStrategy: Uniform frame sampling + Average Pooling (baseline)
- UniformToMeMemoryStrategy: Uniform frame sampling + ToMe compression

Usage:
    from memory_strategies import get_strategy, STRATEGY_REGISTRY
    
    # Get a specific strategy
    strategy = get_strategy('uniform', num_history=8, compress_stride=2)
    strategy = get_strategy('uniform_tome', num_history=8, compress_stride=2, grid_size=2)
    
    # List available strategies
    print(STRATEGY_REGISTRY.keys())
"""

from .base import BaseMemoryStrategy
from .uniform import UniformMemoryStrategy
from .uniform_tome import UniformToMeMemoryStrategy

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'uniform': UniformMemoryStrategy,
    'uniform_tome': UniformToMeMemoryStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseMemoryStrategy:
    """
    Get a memory strategy by name.
    
    Args:
        name: Strategy name (e.g., 'uniform', 'tome')
        **kwargs: Strategy-specific arguments
        
    Returns:
        Instantiated memory strategy
        
    Raises:
        ValueError: If strategy name is not found
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    
    return STRATEGY_REGISTRY[name](**kwargs)


def register_strategy(name: str):
    """
    Decorator to register a new memory strategy.
    
    Usage:
        @register_strategy('my_strategy')
        class MyStrategy(BaseMemoryStrategy):
            ...
    """
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


__all__ = [
    'BaseMemoryStrategy',
    'UniformMemoryStrategy',
    'UniformToMeMemoryStrategy',
    'get_strategy',
    'register_strategy',
    'STRATEGY_REGISTRY',
]
