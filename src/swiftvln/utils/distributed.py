# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Distributed training/evaluation utilities for VLN.
"""

import os
import datetime
import torch
import torch.distributed as dist
from typing import Tuple


def init_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed environment if available.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if world_size > 1:
            # Set CUDA device BEFORE init_process_group to avoid warnings
            # This ensures the device is known when NCCL initializes
            torch.cuda.set_device(local_rank)
            
            # Set longer NCCL timeout to avoid timeout issues
            timeout = datetime.timedelta(hours=2)
            # Initialize process group - device is already set above
            dist.init_process_group(backend='nccl', timeout=timeout)
        
        return rank, world_size, local_rank
    
    return 0, 1, 0
