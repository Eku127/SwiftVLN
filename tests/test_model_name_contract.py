from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_BY_NAME = REPO_ROOT / "src/swiftvln/scripts/eval/eval_by_name.sh"


MODEL_NAME_CASES = {
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed": (
        "模型族:         qwen2_5_vl",
        "MEMORY_METHOD:  history",
        "HISTORY_PROCESSOR_TYPE: per_frame",
        "LOG_BASE:       1.0",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed": (
        "LOG_BASE:       2.0",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed": (
        "USE_RANDOM:     true",
        "SAMPLING_MODE:  random",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed": (
        "NUM_HISTORY:    0",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed": (
        "HISTORY_PROCESSOR_TYPE: gtc",
        "GTC_OUTPUT_TOKENS: 512",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed": (
        "HISTORY_PROCESSOR_TYPE: segment_gtc",
        "SGTC_OUTPUT_TOKENS: 512",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed": (
        "MEMORY_METHOD:  map",
        "MAP_GLOBAL_SIDE_M: 1000",
        "MAP_LOCAL_SIDE_M: 400",
        "MAP_RENDER_PX: 448",
        "MAP_MASK_METHOD: dilate20",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed": (
        "SYSTEM_PROMPT:  initial",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm": (
        "USE_POSE_EMBED: true",
        "POSE_FUSION_METHOD: film",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed": (
        "NUM_OVERLAP:    4",
    ),
    "swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed": (
        "NUM_OVERLAP:    16",
    ),
    "swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed": (
        "模型大小:       7b",
        "模型族:         qwen2_5_vl",
    ),
    "swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed": (
        "模型大小:       2b",
        "模型族:         qwen3_vl",
    ),
    "swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed": (
        "模型大小:       8b",
        "模型族:         qwen3_vl",
    ),
    "swiftvln-habitat-3b-1ep-f32s4-overlap16-pf-h8-b1.0-pool-s2-noembed-bs64-lr2e-5-123456": (
        "环境类型:       habitat",
        "BATCH_SIZE:     64",
        "LEARNING_RATE:  2e-5",
    ),
}


class ModelNameShellContractTest(unittest.TestCase):
    def test_model_zoo_and_historical_names_keep_their_meaning(self):
        env = os.environ.copy()
        env.update({"DRY_RUN": "true", "ENV_TYPE": ""})

        for model_name, expected_fragments in MODEL_NAME_CASES.items():
            with self.subTest(model_name=model_name):
                proc = subprocess.run(
                    ["bash", str(EVAL_BY_NAME), model_name],
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                self.assertEqual(proc.returncode, 0, output)
                self.assertIn("模型架构:       swiftvln", output)
                self.assertIn("NUM_FRAMES:     32", output)
                self.assertIn("NUM_FUTURE_STEPS: 4", output)
                for fragment in expected_fragments:
                    self.assertIn(fragment, output)

    def test_unknown_model_prefix_is_rejected(self):
        env = os.environ.copy()
        env["DRY_RUN"] = "true"
        proc = subprocess.run(
            ["bash", str(EVAL_BY_NAME), "streamvln-satnav-3b"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("swiftvln-", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
