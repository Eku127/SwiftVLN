import sys
import unittest
from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.model.evaluator import OverlapVLNEvaluator, TurnContext


def make_turn(index: int) -> TurnContext:
    return TurnContext(
        user_input_ids=torch.tensor([[index]], dtype=torch.long),
        assistant_response=f"action-{index}",
        image_embed=torch.full((1, 2), float(index), dtype=torch.float32),
    )


class OverlapVLNEvalWindowingTest(unittest.TestCase):
    def make_evaluator(self, overlap_turns: int, num_turns: int = 3):
        evaluator = object.__new__(OverlapVLNEvaluator)
        evaluator.overlap_turns = overlap_turns
        evaluator.window_turns = [make_turn(i) for i in range(num_turns)]
        evaluator.overlap_context = object()
        evaluator._build_assistant_turn_ids = (
            lambda response: torch.tensor([[99]], dtype=torch.long)
        )
        return evaluator

    def test_zero_overlap_does_not_reuse_previous_window_context(self):
        evaluator = self.make_evaluator(overlap_turns=0)

        evaluator._prepare_overlap_context()

        self.assertIsNone(evaluator.overlap_context)

    def test_positive_overlap_reuses_only_last_overlap_turns(self):
        evaluator = self.make_evaluator(overlap_turns=2, num_turns=3)

        evaluator._prepare_overlap_context()

        self.assertIsNotNone(evaluator.overlap_context)
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
        self.assertTrue(
            torch.equal(
                evaluator.overlap_context.image_embeds[1],
                torch.full((1, 2), 2.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
