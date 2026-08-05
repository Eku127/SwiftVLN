from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from swiftvln.common.eval.environment import EvaluationEnvironment
from swiftvln.model.evaluator import SwiftVLNEvaluator


class EvaluationStructureContractTest(unittest.TestCase):
    def test_swiftvln_evaluator_uses_environment_composition(self):
        self.assertEqual(SwiftVLNEvaluator.__bases__, (object,))
        self.assertFalse(
            (
                Path(__file__).parents[1] / "src/swiftvln/common/eval/evaluator.py"
            ).exists()
        )

    def test_satnav_config_and_action_parsing_do_not_import_habitat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "satnav.yaml"
            config_path.write_text(
                "DATASET:\n  SPLIT: placeholder\n  DATA_PATH: /static/episodes.json\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                eval_split="val_unseen",
                save_video=False,
                output_dir=tmpdir,
            )

            environment = EvaluationEnvironment(str(config_path), args, "satnav")

        self.assertEqual(environment.config.DATASET.SPLIT, "val_unseen")
        self.assertEqual(
            environment.parse_actions("noise ↑ then ← → and STOP"),
            [1, 2, 3, 0],
        )

    def test_unknown_environment_is_rejected_at_the_boundary(self):
        args = SimpleNamespace(eval_split="test")
        with self.assertRaisesRegex(ValueError, "Unknown env_type"):
            EvaluationEnvironment("unused.yaml", args, "unknown")


if __name__ == "__main__":
    unittest.main()
