"""Stage-A sim-to-real alignment utilities for SwiftVLN."""

from .dataset import PairRecord, SatDronePairDataset, build_manifest_records, load_manifest
from .losses import bidirectional_contrastive_loss, compute_retrieval_metrics, global_cosine_loss
from .model import ProjectionHead, Sim2RealAdapter, TeacherVisionTower, masked_mean_pool

__all__ = [
    "PairRecord",
    "SatDronePairDataset",
    "build_manifest_records",
    "load_manifest",
    "bidirectional_contrastive_loss",
    "compute_retrieval_metrics",
    "global_cosine_loss",
    "ProjectionHead",
    "Sim2RealAdapter",
    "TeacherVisionTower",
    "masked_mean_pool",
]
