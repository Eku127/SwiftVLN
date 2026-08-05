from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from swiftvln.common.embedding_enhancement import create_embedding_pipeline
from swiftvln.common.embedding_enhancement.runtime import (
    configure_embedding_enhancement,
)
from swiftvln.experiment import ExperimentNameError


class EmbeddingContractTest(unittest.TestCase):
    def test_factory_rejects_non_single_mode(self):
        with self.assertRaisesRegex(ExperimentNameError, "exactly one"):
            create_embedding_pipeline(
                embed_dim=16,
                embedding_mode="pose+uav",
            )

    def test_runtime_rejects_non_single_mode_before_model_access(self):
        with self.assertRaisesRegex(ExperimentNameError, "exactly one"):
            configure_embedding_enhancement(
                None,
                embedding_mode="pose+uav",
                uav_adapter_path="",
                uav_adapter_type="transformer_v1",
                uav_adapter_apply_scope="all_images",
                pose_norm_scale=100.0,
            )

    def test_model_loader_rejects_non_single_mode_before_loading_weights(self):
        from swiftvln.model.model import _prepare_loader_kwargs

        with self.assertRaisesRegex(ExperimentNameError, "exactly one"):
            _prepare_loader_kwargs(
                {
                    "embedding_mode": "pose+uav",
                }
            )

    def test_model_loader_rejects_legacy_boolean_surface(self):
        from swiftvln.model.model import _prepare_loader_kwargs

        with self.assertRaisesRegex(TypeError, "Use embedding_mode"):
            _prepare_loader_kwargs({"use_pose_embed": True})

    def test_each_public_embedding_mode_builds_exactly_one_module(self):
        none_pipeline = create_embedding_pipeline(
            embed_dim=16,
            embedding_mode="none",
        )
        pose_pipeline = create_embedding_pipeline(
            embed_dim=16,
            embedding_mode="pose",
        )
        posefilm_pipeline = create_embedding_pipeline(
            embed_dim=16,
            embedding_mode="posefilm",
        )
        uav_pipeline = create_embedding_pipeline(
            embed_dim=16,
            embedding_mode="uav",
        )

        self.assertTrue(none_pipeline.is_empty)
        self.assertEqual(pose_pipeline.enhancement_names, ["pose"])
        self.assertEqual(pose_pipeline.enhancements["pose"].fusion, "additive")
        self.assertEqual(posefilm_pipeline.enhancement_names, ["pose"])
        self.assertEqual(posefilm_pipeline.enhancements["pose"].fusion, "film")
        self.assertEqual(uav_pipeline.enhancement_names, ["uav"])

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            pose_pipeline.add("second", posefilm_pipeline.enhancements["pose"])

    def test_runtime_rebuilds_when_pose_fusion_mode_changes(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(hidden_size=16)
                self.visual = SimpleNamespace(dtype=torch.float32)

        model = DummyModel()
        common = {
            "uav_adapter_path": "",
            "uav_adapter_type": "transformer_v1",
            "uav_adapter_apply_scope": "all_images",
            "pose_norm_scale": 100.0,
            "logger": lambda _message: None,
        }
        configure_embedding_enhancement(model, embedding_mode="pose", **common)
        self.assertEqual(model.pose_embed.fusion, "additive")

        configure_embedding_enhancement(model, embedding_mode="posefilm", **common)
        self.assertEqual(model.pose_embed.fusion, "film")
        self.assertEqual(model.config.embedding_mode, "posefilm")

        configure_embedding_enhancement(model, embedding_mode="none", **common)
        self.assertTrue(model.embed_enhance.is_empty)
        self.assertIsNone(model.pose_embed)
        self.assertIsNone(model.uav_adapter)
        self.assertEqual(model.config.embedding_mode, "none")


if __name__ == "__main__":
    unittest.main()
