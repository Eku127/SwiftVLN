from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_LIB = REPO_ROOT / "src/swiftvln/scripts/eval/eval_lib.sh"
ENQUEUE = REPO_ROOT / "src/swiftvln/scripts/eval/enqueue_eval.sh"
MODEL_NAME = (
    "swiftvln-satnav-3b-1ep-f32s4-overlap0-"
    "pf-h8-pool-s2-uav"
)


class EvalQueueNameContractTest(unittest.TestCase):
    def _env(self):
        env = os.environ.copy()
        src_root = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (src_root, env.get("PYTHONPATH", "")) if part
        )
        return env

    def test_eval_lib_reads_environment_and_embedding_from_spec(self):
        proc = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; parse_env_type_from_model "$2"; '
                'parse_embed_slot_from_model "$2"',
                "bash",
                str(EVAL_LIB),
                MODEL_NAME,
            ],
            cwd=REPO_ROOT,
            env=self._env(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.splitlines(), ["satnav", "uav"])

    def test_enqueue_validates_name_without_touching_real_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._env()
            env["EVAL_QUEUE_DIR"] = temp_dir
            proc = subprocess.run(
                ["bash", str(ENQUEUE), MODEL_NAME, "--skip-checkpoint"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            todo = Path(temp_dir, "eval_todo.txt").read_text(encoding="utf-8")
            self.assertEqual(todo, MODEL_NAME + "\n")

            invalid = subprocess.run(
                ["bash", str(ENQUEUE), "streamvln-satnav", "--skip-checkpoint"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("Invalid SwiftVLN model name", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
