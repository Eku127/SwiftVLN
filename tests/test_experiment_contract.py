from __future__ import annotations

import unittest
from dataclasses import replace

from swiftvln.experiment import (
    ExperimentNameError,
    SwiftVLNExperimentSpec,
    embedding_from_flags,
    parse_model_name,
)
from tests.test_model_name_contract import MODEL_NAME_CASES


class ExperimentNameContractTest(unittest.TestCase):
    def _configuration_without_run_metadata(self, spec):
        values = spec.to_dict()
        for key in ("effective_batch_size", "learning_rate", "timestamp"):
            values.pop(key)
        return values

    def test_all_golden_names_parse_and_preserve_source(self):
        for model_name in MODEL_NAME_CASES:
            with self.subTest(model_name=model_name):
                spec = parse_model_name(model_name)
                self.assertEqual(spec.to_model_name("source"), model_name)
                reparsed = parse_model_name(spec.to_model_name("short"))
                self.assertEqual(
                    self._configuration_without_run_metadata(reparsed),
                    self._configuration_without_run_metadata(spec),
                )

    def test_backbone_family_and_size_are_explicit(self):
        qwen25 = parse_model_name(
            "swiftvln-satnav-7b-1ep-f32s4-overlap0-"
            "pf-h8-pool-s2-noembed"
        )
        qwen3 = parse_model_name(
            "swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-"
            "pf-h8-pool-s2-noembed"
        )
        self.assertEqual((qwen25.model_family, qwen25.model_size), ("qwen2_5_vl", "7b"))
        self.assertEqual((qwen3.model_family, qwen3.model_size), ("qwen3_vl", "8b"))

    def test_current_train_name_is_stable(self):
        spec = SwiftVLNExperimentSpec(
            env_type="satnav",
            effective_batch_size=8,
            learning_rate="2e-5",
            timestamp="151815",
        )
        self.assertEqual(
            spec.to_model_name("run"),
            "swiftvln-satnav-3b-1ep-f32s4-overlap0-"
            "pf-h8-b1.0-pool-s2-noembed-bs8-lr2e-5-151815",
        )

    def test_public_embedding_mode_is_mutually_exclusive(self):
        self.assertEqual(embedding_from_flags(False, False), "none")
        self.assertEqual(embedding_from_flags(True, False), "pose")
        self.assertEqual(embedding_from_flags(True, False, "film"), "posefilm")
        self.assertEqual(embedding_from_flags(False, True), "uav")
        with self.assertRaisesRegex(ExperimentNameError, "mutually exclusive"):
            embedding_from_flags(True, True)
        with self.assertRaisesRegex(ExperimentNameError, "no longer supported"):
            parse_model_name(
                "swiftvln-satnav-3b-1ep-f32s4-overlap0-"
                "pf-h8-pool-s2-pose+uav"
            )

    def test_map_constraints_are_centralized(self):
        map_spec = SwiftVLNExperimentSpec(
            env_type="satnav",
            memory_method="map",
        )
        self.assertIn("map-g1000-l400-r448-d20-s2", map_spec.to_model_name())
        with self.assertRaisesRegex(ExperimentNameError, "only for SatNav"):
            replace(map_spec, env_type="habitat")
        with self.assertRaisesRegex(ExperimentNameError, "cannot use"):
            replace(map_spec, embedding="pose")

    def test_non_default_gtc_settings_are_not_lost_in_the_name(self):
        spec = SwiftVLNExperimentSpec(
            env_type="satnav",
            history_processor_type="gtc",
            gtc_output_tokens=256,
            gtc_temperature=0.2,
            gtc_num_iterations=2,
        )
        model_name = spec.to_model_name()
        self.assertIn("gtc-k256-t0.2-i2", model_name)
        self.assertEqual(parse_model_name(model_name).to_dict(), spec.to_dict())

    def test_overlap_and_run_metadata_are_validated(self):
        with self.assertRaisesRegex(ExperimentNameError, "num_overlap"):
            SwiftVLNExperimentSpec(env_type="satnav", num_overlap=32)
        with self.assertRaisesRegex(ExperimentNameError, "together"):
            SwiftVLNExperimentSpec(
                env_type="satnav",
                effective_batch_size=8,
            )


if __name__ == "__main__":
    unittest.main()
