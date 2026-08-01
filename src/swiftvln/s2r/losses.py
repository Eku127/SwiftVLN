"""Losses and retrieval metrics for Stage-A sim-to-real alignment."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


class _GatherWithGrad(torch.autograd.Function):
    """Gradient-preserving all-gather for contrastive learning."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor):
        if not dist.is_available() or not dist.is_initialized():
            ctx.rank = 0
            return (tensor,)

        ctx.rank = dist.get_rank()
        world_size = dist.get_world_size()
        outputs = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(outputs, tensor)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not dist.is_available() or not dist.is_initialized():
            return grad_outputs[0]

        # Every rank computes a loss for its local queries against all gathered
        # keys.  The gradient for rank r's input therefore contains the
        # contributions from output slot r on *all* ranks.  Selecting only
        # grad_outputs[rank] drops the remote key-side gradients.  Summing the
        # complete slot stack first preserves the true global-batch objective;
        # DDP will subsequently average parameter gradients as usual.
        stacked_grads = torch.stack(
            [gradient.contiguous() for gradient in grad_outputs],
            dim=0,
        )
        dist.all_reduce(stacked_grads, op=dist.ReduceOp.SUM)
        return stacked_grads[ctx.rank]


def gather_features(tensor: torch.Tensor) -> torch.Tensor:
    """Gather features across ranks when distributed training is enabled."""
    outputs = _GatherWithGrad.apply(tensor)
    return torch.cat(outputs, dim=0)


def bidirectional_contrastive_loss(
    uav_features: torch.Tensor,
    sat_features: torch.Tensor,
    *,
    temperature: float = 0.07,
    gather_distributed: bool = True,
) -> torch.Tensor:
    """CLIP-style bidirectional contrastive loss."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if gather_distributed and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        batch_size = uav_features.shape[0]
        gathered_uav = gather_features(uav_features)
        gathered_sat = gather_features(sat_features)
        labels = torch.arange(batch_size, device=uav_features.device) + rank * batch_size
    else:
        gathered_uav = uav_features
        gathered_sat = sat_features
        labels = torch.arange(uav_features.shape[0], device=uav_features.device)

    logits_u2s = uav_features @ gathered_sat.t() / temperature
    logits_s2u = sat_features @ gathered_uav.t() / temperature
    loss_u2s = F.cross_entropy(logits_u2s, labels)
    loss_s2u = F.cross_entropy(logits_s2u, labels)
    return 0.5 * (loss_u2s + loss_s2u)


def global_cosine_loss(uav_pooled: torch.Tensor, sat_pooled: torch.Tensor) -> torch.Tensor:
    """Global cosine distillation loss."""
    u = F.normalize(uav_pooled, dim=-1)
    s = F.normalize(sat_pooled, dim=-1)
    return (1.0 - (u * s).sum(dim=-1)).mean()


def compute_retrieval_metrics(
    uav_features: torch.Tensor,
    sat_features: torch.Tensor,
    *,
    topk: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute one-to-one retrieval metrics given aligned feature rows."""
    if uav_features.shape[0] != sat_features.shape[0]:
        raise ValueError("uav_features and sat_features must have the same length")

    num_samples = uav_features.shape[0]
    if num_samples == 0:
        return {
            "num_samples": 0.0,
            "paired_cosine_mean": 0.0,
        }

    u = F.normalize(uav_features, dim=-1)
    s = F.normalize(sat_features, dim=-1)
    sim = u @ s.t()
    labels = torch.arange(num_samples, device=sim.device)

    max_k = min(max(topk), num_samples)
    u2s_topk = sim.topk(k=max_k, dim=1).indices
    s2u_topk = sim.t().topk(k=max_k, dim=1).indices

    metrics: Dict[str, float] = {"num_samples": float(num_samples)}
    for k in topk:
        actual_k = min(k, num_samples)
        metrics[f"u2s_r@{k}"] = float((u2s_topk[:, :actual_k] == labels.unsqueeze(1)).any(dim=1).float().mean().item())
        metrics[f"s2u_r@{k}"] = float((s2u_topk[:, :actual_k] == labels.unsqueeze(1)).any(dim=1).float().mean().item())

    metrics["paired_cosine_mean"] = float(sim.diag().mean().item())
    return metrics
