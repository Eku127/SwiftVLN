from __future__ import annotations

import unittest

from swiftvln.common.embedding_enhancement import create_embedding_pipeline
from swiftvln.common.embedding_enhancement.runtime import (
    configure_embedding_enhancement,
)
from swiftvln.experiment import ExperimentNameError


class EmbeddingContractTest(unittest.TestCase):
    def test_factory_rejects_combined_pose_and_uav(self):
        with self.assertRaisesRegex(ExperimentNameError, "mutually exclusive"):
            create_embedding_pipeline(
                embed_dim=16,
                use_pose_embed=True,
                use_uav_adapter=True,
            )

    def test_runtime_rejects_combination_before_model_access(self):
        with self.assertRaisesRegex(ExperimentNameError, "mutually exclusive"):
            configure_embedding_enhancement(
                None,
                use_pose_embed=True,
                use_uav_adapter=True,
                uav_adapter_path="",
                uav_adapter_type="transformer_v1",
                uav_adapter_apply_scope="all_images",
                pose_fusion_method="film",
                pose_norm_scale=100.0,
            )

    def test_model_loader_rejects_combination_before_loading_weights(self):
        from swiftvln.model.model import _prepare_loader_kwargs

        with self.assertRaisesRegex(ExperimentNameError, "mutually exclusive"):
            _prepare_loader_kwargs(
                {
                    "use_pose_embed": True,
                    "use_uav_adapter": True,
                }
            )

    def test_each_public_embedding_mode_still_builds_independently(self):
        none_pipeline = create_embedding_pipeline(embed_dim=16)
        pose_pipeline = create_embedding_pipeline(
            embed_dim=16,
            use_pose_embed=True,
        )
        uav_pipeline = create_embedding_pipeline(
            embed_dim=16,
            use_uav_adapter=True,
        )

        self.assertTrue(none_pipeline.is_empty)
        self.assertEqual(pose_pipeline.enhancement_names, ["pose"])
        self.assertEqual(uav_pipeline.enhancement_names, ["uav"])


if __name__ == "__main__":
    unittest.main()
