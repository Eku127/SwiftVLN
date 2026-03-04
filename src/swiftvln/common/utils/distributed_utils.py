# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Distributed training/evaluation utilities for VLN.
"""

import os
import datetime
import torch
import torch.distributed as dist
from typing import List, Dict, Tuple, Any, Optional


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


def gather_metrics(
    results: List[Dict[str, Any]],
    rank: int,
    world_size: int,
    local_rank: int,
    is_main: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Gather evaluation metrics from all ranks in distributed mode.
    
    Args:
        results: List of result dictionaries from current rank
        rank: Current process rank
        world_size: Total number of processes
        local_rank: Local rank on current node
        is_main: Whether this is the main process
        
    Returns:
        Tuple of (sucs_all, spls_all, oss_all, nes_all, all_results_merged)
        - Tensors contain metrics from all ranks
        - all_results_merged contains all detailed results (only on main rank)
    """
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
        
        # Convert results to tensors for all_gather
        # Extract metrics as tensors
        sucs = torch.tensor([r.get("success", 0.0) for r in results], dtype=torch.float32, device=device)
        spls = torch.tensor([r.get("spl", 0.0) for r in results], dtype=torch.float32, device=device)
        oss = torch.tensor([r.get("oracle_success", 0.0) for r in results], dtype=torch.float32, device=device)
        nes = torch.tensor([r.get("distance_to_goal", 0.0) for r in results], dtype=torch.float32, device=device)
        ep_num = torch.tensor(len(results), dtype=torch.int64, device=device)
        
        # First, gather episode counts from all ranks
        ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
        dist.all_gather(ep_num_all, ep_num)
        
        # Prepare tensors for gathering (with correct sizes)
        sucs_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        spls_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        oss_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        nes_all = [torch.zeros(ep_num_all[i].item(), dtype=torch.float32, device=device) for i in range(world_size)]
        
        # Synchronize before gathering results
        dist.barrier()
        
        # Gather all results
        dist.all_gather(sucs_all, sucs)
        dist.all_gather(spls_all, spls)
        dist.all_gather(oss_all, oss)
        dist.all_gather(nes_all, nes)
        
        # Gather detailed results from all ranks using gather_object
        all_rank_results = [None for _ in range(world_size)]
        dist.gather_object(results, all_rank_results if is_main else None, dst=0)
        
        dist.barrier()
        
        # Concatenate results on main process
        if is_main:
            sucs_all = torch.cat(sucs_all, dim=0)
            spls_all = torch.cat(spls_all, dim=0)
            oss_all = torch.cat(oss_all, dim=0)
            nes_all = torch.cat(nes_all, dim=0)
            
            # Merge all rank results into a single list
            all_results_merged = []
            for rank_results in all_rank_results:
                if rank_results is not None:
                    all_results_merged.extend(rank_results)
        else:
            # Non-main processes return empty merged results
            all_results_merged = []
            
        return sucs_all, spls_all, oss_all, nes_all, all_results_merged
    else:
        # Single process mode
        sucs_all = torch.tensor([r.get("success", 0.0) for r in results])
        spls_all = torch.tensor([r.get("spl", 0.0) for r in results])
        oss_all = torch.tensor([r.get("oracle_success", 0.0) for r in results])
        nes_all = torch.tensor([r.get("distance_to_goal", 0.0) for r in results])
        all_results_merged = results
        
        return sucs_all, spls_all, oss_all, nes_all, all_results_merged
