from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from swiftvln import cli


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
        self.assertNotIn("deploy", output)

    def test_train_eval_and_queue_help_are_available(self):
        train_help = self._run_help("train", "--help")
        eval_help = self._run_help("eval", "--help")
        self.assertIn("usage:", train_help.lower())
        self.assertIn("usage:", eval_help.lower())
        self.assertNotIn("--model {swiftvln}", train_help)
        self.assertNotIn("--model {swiftvln}", eval_help)
        self.assertIn("train", self._run_help("queue", "--help"))
        self.assertIn("eval", self._run_help("queue", "--help"))

    def test_train_and_eval_forward_model_runtime_arguments(self):
        with patch.object(cli, "_run_train", return_value=0) as run_train:
            self.assertEqual(
                cli.main(["train", "--model", "Qwen/Qwen3-VL-2B-Instruct"]),
                0,
            )
            run_train.assert_called_once_with(
                ["--model", "Qwen/Qwen3-VL-2B-Instruct"]
            )

        with patch.object(cli, "_run_eval", return_value=0) as run_eval:
            self.assertEqual(
                cli.main(["eval", "--model_path", "/tmp/checkpoint"]),
                0,
            )
            run_eval.assert_called_once_with(
                ["--model_path", "/tmp/checkpoint"]
            )

    def test_queue_target_is_forwarded_without_runner_registry(self):
        with patch.object(cli, "_run_queue", return_value=0) as run_queue:
            self.assertEqual(cli.main(["queue", "eval", "--check-only"]), 0)
            run_queue.assert_called_once_with("eval", ["--check-only"])


if __name__ == "__main__":
    unittest.main()
