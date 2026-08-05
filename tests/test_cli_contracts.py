from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliContractTest(unittest.TestCase):
    def _run_help(self, *args: str) -> str:
        env = os.environ.copy()
        src_root = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (src_root, env.get("PYTHONPATH", "")) if part
        )
        proc = subprocess.run(
            [sys.executable, "-m", "swiftvln", *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        return output

    def test_top_level_help_keeps_supported_workflows(self):
        output = self._run_help("--help")
        for command in ("train", "eval", "queue", "s2r-data"):
            self.assertIn(command, output)

    def test_train_eval_and_queue_help_are_available(self):
        self.assertIn("usage:", self._run_help("train", "--help").lower())
        self.assertIn("usage:", self._run_help("eval", "--help").lower())
        self.assertIn("train", self._run_help("queue", "--help"))
        self.assertIn("eval", self._run_help("queue", "--help"))


if __name__ == "__main__":
    unittest.main()
