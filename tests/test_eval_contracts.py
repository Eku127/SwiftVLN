from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from swiftvln.common.eval.results import ResultRecorder  # noqa: E402
from swiftvln.model.eval_runner import SwiftVLNEvaluationRunner  # noqa: E402
from swiftvln.model.inference import (  # noqa: E402
    SwiftVLNInferenceSession,
    TurnContext,
)


def make_turn(index: int) -> TurnContext:
    return TurnContext(
        user_input_ids=torch.tensor([[index]], dtype=torch.long),
        assistant_response=f"action-{index}",
        image_embed=torch.full((1, 2), float(index), dtype=torch.float32),
    )


class EvalJsonlContractTest(unittest.TestCase):
    def test_resume_deduplicates_by_scene_and_episode_and_keeps_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ResultRecorder(tmpdir)
            rows = [
                {"scene_id": "a", "episode_id": 1, "steps": 3},
                {"scene_id": "b", "episode_id": 1, "steps": 4},
                {"scene_id": "a", "episode_id": 1, "steps": 5},
            ]
            for row in rows:
                recorder.append(row)
            with Path(recorder.result_file).open("a", encoding="utf-8") as handle:
                handle.write("{partial-json\n")

            results = recorder.load_results()
            indexed = {
                recorder.episode_key(
                    result["episode_id"],
                    result["scene_id"],
                ): result
                for result in results
            }

            self.assertEqual(set(indexed), {"a::1", "b::1"})
            self.assertEqual(indexed["a::1"]["steps"], 5)
            self.assertEqual(indexed["b::1"]["steps"], 4)

    def test_rank_completion_marker_is_atomic_and_clearable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ResultRecorder(tmpdir, rank=3, world_size=4)
            recorder.mark_rank_complete(
                processed_count=2,
                resumed_count=1,
                local_total=3,
            )
            recorder.wait_for_ranks([3], timeout_seconds=0)

            marker_path = Path(recorder.rank_sync_dir) / "rank_3.done.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["rank"], 3)
            self.assertEqual(marker["processed_count"], 2)
            self.assertFalse(Path(f"{marker_path}.tmp").exists())

            recorder.clear_rank_markers()
            self.assertFalse(marker_path.exists())

    @patch("swiftvln.common.eval.results.get_swanlab_url_from_train_metadata")
    @patch("swiftvln.common.eval.results.get_swanlab_url")
    def test_summary_metrics_and_public_jsonl_schema(
        self,
        get_swanlab_url,
        get_swanlab_url_from_train_metadata,
    ):
        get_swanlab_url.return_value = None
        get_swanlab_url_from_train_metadata.return_value = None
        args = SimpleNamespace(
            eval_split="val_unseen",
            model_path="/model/checkpoint",
            num_history=8,
            env_type="habitat",
            video_compression=False,
            save_video=False,
        )
        results = [
            {
                "scene_id": "scene",
                "episode_id": 1,
                "success": 1.0,
                "spl": 0.5,
                "oracle_success": 1.0,
                "distance_to_goal": 2.0,
                "steps": 3,
                "_timing_stats": {"model": 1.0},
                "_total_time": 1.0,
                "_step_count": 3,
            },
            {
                "scene_id": "scene",
                "episode_id": 2,
                "success": 0.0,
                "spl": 0.0,
                "oracle_success": 0.0,
                "distance_to_goal": float("inf"),
                "steps": 5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir, redirect_stdout(StringIO()):
            recorder = ResultRecorder(tmpdir, world_size=2)
            recorder.save_summary(
                results,
                [],
                args=args,
                model_description="SwiftVLN",
                summary_extras={"history_processor_type": "per_frame"},
                uses_compression=True,
            )

            summary = json.loads(
                (Path(tmpdir) / "evaluation_summary.json").read_text(encoding="utf-8")
            )
            public_results = [
                json.loads(line)
                for line in (Path(tmpdir) / "all_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["mean_spl"], 0.25)
        self.assertEqual(summary["navigation_error"], 2.0)
        self.assertEqual(summary["avg_steps"], 4.0)
        self.assertEqual(summary["world_size"], 2)
        self.assertEqual(summary["history_processor_type"], "per_frame")
        self.assertEqual([row["episode_id"] for row in public_results], [2, 1])
        self.assertNotIn("_timing_stats", public_results[1])

    def test_episode_exception_becomes_durable_error_result(self):
        runner = SwiftVLNEvaluationRunner(
            SimpleNamespace(env_type="habitat"),
            summary_extras={},
        )
        episode = SimpleNamespace(
            episode_id="ep-7",
            scene_id="scene.glb",
            trajectory_type=None,
        )
        env_wrapper = SimpleNamespace(
            get_instruction=lambda current_episode: "go forward"
        )

        class BrokenEvaluator:
            def eval_episode(self, *args, **kwargs):
                raise RuntimeError("expected failure")

        result = runner.evaluate_episode(BrokenEvaluator(), env_wrapper, episode)

        self.assertEqual(result["episode_id"], "ep-7")
        self.assertEqual(result["scene_id"], "scene")
        self.assertEqual(result["success"], 0.0)
        self.assertEqual(result["error"], "expected failure")
        self.assertEqual(result["error_tags"], [])


class EvalWindowContractTest(unittest.TestCase):
    def _make_evaluator(self, overlap_turns: int, num_turns: int = 3):
        session = object.__new__(SwiftVLNInferenceSession)
        session.overlap_turns = overlap_turns
        session.window_turns = [make_turn(index) for index in range(num_turns)]
        session.overlap_context = object()
        session._build_assistant_turn_ids = lambda response: torch.tensor(
            [[99]], dtype=torch.long
        )
        return session

    def test_zero_overlap_does_not_reuse_previous_window_context(self):
        evaluator = self._make_evaluator(overlap_turns=0)
        evaluator._prepare_overlap_context()
        self.assertIsNone(evaluator.overlap_context)

    def test_positive_overlap_reuses_only_last_overlap_turns(self):
        evaluator = self._make_evaluator(overlap_turns=2)
        evaluator._prepare_overlap_context()

        self.assertEqual(
            evaluator.overlap_context.input_ids.tolist(),
            [[1, 99, 2, 99]],
        )
        self.assertEqual(len(evaluator.overlap_context.image_embeds), 2)
        self.assertTrue(
            torch.equal(
                evaluator.overlap_context.image_embeds[0],
                torch.full((1, 2), 1.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
