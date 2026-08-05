from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from swiftvln.common.eval.runner import BaseVLNEval  # noqa: E402
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
            path = Path(tmpdir) / "result.jsonl"
            rows = [
                {"scene_id": "a", "episode_id": 1, "steps": 3},
                {"scene_id": "b", "episode_id": 1, "steps": 4},
                {"scene_id": "a", "episode_id": 1, "steps": 5},
            ]
            for row in rows:
                BaseVLNEval.append_result_jsonl(str(path), row)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("{partial-json\n")

            results = BaseVLNEval.load_dedup_results(str(path))
            indexed = {
                BaseVLNEval.build_episode_key(
                    result["episode_id"],
                    result["scene_id"],
                ): result
                for result in results
            }

            self.assertEqual(set(indexed), {"a::1", "b::1"})
            self.assertEqual(indexed["a::1"]["steps"], 5)
            self.assertEqual(indexed["b::1"]["steps"], 4)

    def test_episode_exception_becomes_durable_error_result(self):
        runner = BaseVLNEval()
        runner.args = SimpleNamespace(env_type="habitat")
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
