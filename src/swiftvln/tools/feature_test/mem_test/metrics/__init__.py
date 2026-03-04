# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Evaluation Metrics for Memory Strategies

This module provides metrics for evaluating memory construction strategies:
- NLL/CE: Negative Log-Likelihood / Cross-Entropy for action prediction
- Redundancy: Memory token redundancy (similarity between tokens)
- Coverage: History coverage (how well memory represents full history)
"""

from .nll import compute_chunk_nll, NLLComputer
from .redundancy import compute_redundancy, RedundancyComputer
from .coverage import compute_coverage, CoverageComputer

__all__ = [
    'compute_chunk_nll',
    'NLLComputer',
    'compute_redundancy', 
    'RedundancyComputer',
    'compute_coverage',
    'CoverageComputer',
]
