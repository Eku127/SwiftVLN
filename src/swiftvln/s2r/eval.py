"""Evaluation for Stage-A sim-to-real alignment."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

if __package__ in {None, ""}:
    from swiftvln.s2r.arguments import EvalConfig, parse_eval_args
    from swiftvln.s2r.dataset import SatDronePairDataset, collate_pair_batch
    from swiftvln.s2r.losses import compute_retrieval_metrics
    from swiftvln.s2r.model import (
        ProjectionHead,
        Sim2RealAdapter,
        TeacherVisionTower,
        masked_mean_pool,
        pad_token_sequences,
        resolve_device,
    )
else:
    from .arguments import EvalConfig, parse_eval_args
    from .dataset import SatDronePairDataset, collate_pair_batch
    from .losses import compute_retrieval_metrics
    from .model import (
        ProjectionHead,
        Sim2RealAdapter,
        TeacherVisionTower,
        masked_mean_pool,
        pad_token_sequences,
        resolve_device,
    )


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def evaluate_model(
    teacher: TeacherVisionTower,
    adapter: Sim2RealAdapter,
    projection_head: ProjectionHead,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, object]:
    """Run retrieval evaluation on one manifest split."""
    adapter.eval()
    projection_head.eval()

    uav_proj_batches: List[torch.Tensor] = []
    sat_proj_batches: List[torch.Tensor] = []
    datasets: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            uav_tokens_list = teacher.encode_images(batch["uav_image"])
            sat_tokens_list = teacher.encode_images(batch["sat_image"])

            uav_tokens, uav_mask = pad_token_sequences(uav_tokens_list, device=device, dtype=torch.float32)
            sat_tokens, sat_mask = pad_token_sequences(sat_tokens_list, device=device, dtype=torch.float32)

            aligned_uav = adapter(uav_tokens, attention_mask=uav_mask)
            uav_pooled = masked_mean_pool(aligned_uav, uav_mask)
            sat_pooled = masked_mean_pool(sat_tokens, sat_mask)

            uav_proj = F.normalize(projection_head(uav_pooled), dim=-1).cpu()
            sat_proj = F.normalize(projection_head(sat_pooled), dim=-1).cpu()
            uav_proj_batches.append(uav_proj)
            sat_proj_batches.append(sat_proj)
            datasets.extend(batch["dataset"])

    uav_proj_all = torch.cat(uav_proj_batches, dim=0) if uav_proj_batches else torch.empty(0, projection_head.net[-1].out_features)
    sat_proj_all = torch.cat(sat_proj_batches, dim=0) if sat_proj_batches else torch.empty(0, projection_head.net[-1].out_features)

    metrics = compute_retrieval_metrics(uav_proj_all, sat_proj_all)
    by_source: Dict[str, Dict[str, float]] = {}
    if datasets:
        dataset_to_indices: Dict[str, List[int]] = defaultdict(list)
        for index, dataset in enumerate(datasets):
            dataset_to_indices[dataset].append(index)
        for dataset, indices in dataset_to_indices.items():
            subset_uav = uav_proj_all[indices]
            subset_sat = sat_proj_all[indices]
            by_source[dataset] = compute_retrieval_metrics(subset_uav, subset_sat)

    metrics["by_source"] = by_source
    return metrics


def load_checkpoint_for_eval(
    checkpoint_path: str,
    *,
    teacher_model_path: Optional[str],
    device: torch.device,
    teacher_dtype: str,
):
    """Load teacher, adapter, and projection head from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter_kwargs = checkpoint["adapter_kwargs"]
    projection_kwargs = checkpoint["projection_kwargs"]
    teacher_path = teacher_model_path or checkpoint["teacher_model_path"]

    teacher = TeacherVisionTower.from_pretrained(
        teacher_path,
        device=device,
        dtype_name=teacher_dtype,
    )
    adapter = Sim2RealAdapter(**adapter_kwargs).to(device=device, dtype=torch.float32)
    projection_head = ProjectionHead(**projection_kwargs).to(device=device, dtype=torch.float32)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
    return checkpoint, teacher, adapter, projection_head


def eval_main(argv=None):
    """CLI entry for Stage-A evaluation."""
    args: EvalConfig = parse_eval_args(argv)
    device = resolve_device(args.device)
    checkpoint, teacher, adapter, projection_head = load_checkpoint_for_eval(
        args.checkpoint_path,
        teacher_model_path=args.teacher_model_path or None,
        device=device,
        teacher_dtype=args.teacher_dtype,
    )
    dataset = SatDronePairDataset(args.manifest_path, split=args.split, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pair_batch,
    )
    metrics = evaluate_model(
        teacher,
        adapter,
        projection_head,
        dataloader,
        device=device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    eval_main()
