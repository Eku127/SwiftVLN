from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from swiftvln.common.embedding_enhancement.pose_utils import (  # noqa: E402
    reconstruct_pose_from_actions,
)
from swiftvln.common.history_processors.compressor import (  # noqa: E402
    HistoryTokenCompressor,
)
from swiftvln.common.history_processors.gtc import (  # noqa: E402
    GlobalTokenClustering,
    soft_kmeans_step,
)
from swiftvln.common.history_processors.per_frame import (  # noqa: E402
    sample_per_frame_history_indices,
)
from swiftvln.common.history_processors.segment_gtc import SegmentGTC  # noqa: E402


class HistorySamplingContractTest(unittest.TestCase):
    def test_uniform_and_logarithmic_indices(self):
        self.assertEqual(sample_per_frame_history_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(
            sample_per_frame_history_indices(10, 4, log_base=2.0),
            [0, 5, 8, 9],
        )
        self.assertEqual(sample_per_frame_history_indices(10, 1), [9])
        self.assertEqual(sample_per_frame_history_indices(3, 8), [0, 1, 2])
        self.assertEqual(sample_per_frame_history_indices(0, 8), [])

    def test_random_indices_are_sorted_unique_and_bounded(self):
        np.random.seed(42)
        indices = sample_per_frame_history_indices(20, 8, use_random=True)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(len(indices), 8)
        self.assertTrue(all(0 <= index < 20 for index in indices))


class CompressionContractTest(unittest.TestCase):
    def test_pooling_token_count_matches_output_shape(self):
        compressor = HistoryTokenCompressor(stride=2, method="pooling")
        embeds = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
        grid = torch.tensor([1, 16, 16], dtype=torch.long)

        compressed, new_grid = compressor.compress(embeds, grid)

        self.assertEqual(compressor.get_compressed_token_count(1, 16, 16), 64)
        self.assertEqual(tuple(compressed.shape), (64, 4))
        self.assertEqual(new_grid.tolist(), [1, 8, 8])

    def test_shared_soft_kmeans_matches_pre_refactor_formula(self):
        torch.manual_seed(7)
        tokens = torch.randn(12, 6)
        centroids = tokens[[0, 3, 6, 9]].clone()
        temperature = 0.2

        tokens_norm = F.normalize(tokens.float(), dim=-1)
        centroids_norm = F.normalize(centroids.float(), dim=-1)
        assignments = F.softmax(
            torch.mm(tokens_norm, centroids_norm.T) / temperature,
            dim=-1,
        )
        expected = torch.mm(assignments.T.to(tokens.dtype), tokens) / (
            assignments.sum(dim=0, keepdim=True)
            .T.to(tokens.dtype)
            .clamp(min=1e-6)
        )

        actual = soft_kmeans_step(tokens, centroids, temperature)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_global_and_segment_gtc_output_shapes(self):
        torch.manual_seed(11)
        frames = [torch.randn(4, 8) for _ in range(9)]
        grids = [torch.tensor([1, 2, 2]) for _ in frames]

        global_output = GlobalTokenClustering(output_tokens=7).process(frames, grids)
        segment_output = SegmentGTC(output_tokens=16).process(frames, grids)

        self.assertEqual(tuple(global_output.shape), (7, 8))
        self.assertEqual(tuple(segment_output.shape), (16, 8))


class PoseContractTest(unittest.TestCase):
    def test_pose_reconstruction_keeps_satnav_action_convention(self):
        poses = reconstruct_pose_from_actions([1, 2, 1, 3])

        self.assertEqual(tuple(poses.shape), (4, 4))
        self.assertTrue(np.allclose(poses[0], [10.0, 0.0, 0.0, 1.0]))
        self.assertTrue(
            np.allclose(
                poses[1],
                [10.0, 0.0, -math.sin(math.radians(15)), math.cos(math.radians(15))],
            )
        )
        self.assertTrue(np.allclose(poses[-1, 2:], [0.0, 1.0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
