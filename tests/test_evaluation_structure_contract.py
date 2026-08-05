from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from swiftvln.common.eval.environment import EvaluationEnvironment
from swiftvln.model.eval import build_summary_extras, parse_eval_args
from swiftvln.model.eval_runner import SwiftVLNEvaluationRunner, distribute_episodes
from swiftvln.model.evaluator import SwiftVLNEvaluator


class EvaluationStructureContractTest(unittest.TestCase):
    def test_swiftvln_evaluator_uses_environment_composition(self):
        self.assertEqual(SwiftVLNEvaluator.__bases__, (object,))
        self.assertFalse(
            (
                Path(__file__).parents[1] / "src/swiftvln/common/eval/evaluator.py"
            ).exists()
        )
        self.assertFalse(
            (Path(__file__).parents[1] / "src/swiftvln/common/eval/runner.py").exists()
        )

    def test_failure_classification_is_removed(self):
        source_root = Path(__file__).parents[1] / "src/swiftvln"
        self.assertFalse((source_root / "common/utils/error_analyzer.py").exists())
        evaluation_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                source_root / "common/eval/environment.py",
                source_root / "model/evaluator.py",
                source_root / "model/eval_runner.py",
            )
        )
        for removed_symbol in (
            "ErrorAnalyzer",
            "TrajectoryRecorder",
            "error_tags",
            "had_deviation",
            "deviation_recovered",
        ):
            self.assertNotIn(removed_symbol, evaluation_sources)

    def test_habitat_and_video_capabilities_are_retained(self):
        source_root = Path(__file__).parents[1] / "src/swiftvln"
        self.assertTrue((source_root / "common/env/habitat.py").exists())
        self.assertTrue((source_root / "habitat_extensions/measures.py").exists())
        self.assertTrue((source_root / "configs/vln_r2r.yaml").exists())
        self.assertTrue((source_root / "configs/vln_r2r_smoke.yaml").exists())

        for method in (
            "collect_habitat_frame",
            "collect_satnav_topdown",
            "save_habitat_video",
            "save_satnav_video",
        ):
            self.assertTrue(hasattr(EvaluationEnvironment, method))

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

    def test_eval_cli_keeps_model_and_summary_defaults(self):
        args = parse_eval_args(["--model_path", "/model", "--env-type", "satnav"])
        extras = build_summary_extras(args)

        self.assertEqual(args.model_type, "swiftvln_qwen2_5_vl")
        self.assertFalse(hasattr(args, "template_type"))
        self.assertNotIn("template_type", extras)
        self.assertEqual(args.num_frames, 32)
        self.assertEqual(args.num_history, 8)
        self.assertEqual(args.compress_stride, 2)
        self.assertEqual(args.embedding_mode, "none")
        self.assertEqual(extras["embedding_mode"], "none")
        self.assertEqual(extras["history_processor_type"], "per_frame")
        self.assertEqual(extras["log_base"], 1.0)

    def test_eval_runtime_has_no_unused_template_chain(self):
        self.assertFalse(hasattr(SwiftVLNEvaluationRunner, "load_template"))
        self.assertEqual(
            tuple(inspect.signature(SwiftVLNEvaluationRunner.create_evaluator).parameters),
            ("self", "model", "processor"),
        )
        with self.assertRaises(SystemExit):
            parse_eval_args(
                [
                    "--model_path",
                    "/model",
                    "--template_type",
                    "swiftvln_qwen2_5_vl",
                ]
            )

    def test_eval_cli_accepts_exactly_one_embedding_mode(self):
        for mode in ("none", "pose", "posefilm", "uav"):
            with self.subTest(mode=mode):
                args = parse_eval_args(
                    [
                        "--model_path",
                        "/model",
                        "--env-type",
                        "satnav",
                        "--embedding_mode",
                        mode,
                    ]
                )
                self.assertEqual(args.embedding_mode, mode)

        with self.assertRaises(SystemExit):
            parse_eval_args(
                [
                    "--model_path",
                    "/model",
                    "--env-type",
                    "satnav",
                    "--embedding_mode",
                    "pose+uav",
                ]
            )

    def test_episode_distribution_is_global_and_balanced(self):
        episodes = [
            SimpleNamespace(episode_id=index, scene_id=f"scene-{index // 3}")
            for index in range(8)
        ]
        shards = [
            distribute_episodes(episodes, rank=rank, world_size=8)[0]
            for rank in range(8)
        ]

        self.assertEqual([len(shard) for shard in shards], [1] * 8)
        self.assertEqual(
            {episode.episode_id for shard in shards for episode in shard},
            set(range(8)),
        )


if __name__ == "__main__":
    unittest.main()
