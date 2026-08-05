from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_QUEUE = REPO_ROOT / "src/swiftvln/scripts/eval/eval_queue.sh"
MODEL_NAME = "swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-uav"


class EvalQueueProtocolTest(unittest.TestCase):
    def _env(self, queue_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        src_root = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (src_root, env.get("PYTHONPATH", "")) if part
        )
        env["PATH"] = os.pathsep.join(
            (str(Path(sys.executable).parent), env.get("PATH", ""))
        )
        env["EVAL_QUEUE_DIR"] = str(queue_dir)
        env.pop("AUTO_TODO", None)
        env.pop("DYNAMIC_TODO", None)
        env.pop("WAIT_FOR_NEW_TASKS", None)
        return env

    def _run(
        self,
        queue_dir: Path,
        *args: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self._env(queue_dir)
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", str(EVAL_QUEUE), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_valid_todo_is_checked_without_launching_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir)
            todo = queue_dir / "eval_todo.txt"
            todo.write_text(f"# queued model\n{MODEL_NAME}\n", encoding="utf-8")

            proc = self._run(queue_dir, "--check-queue")
            output = proc.stdout + proc.stderr

            self.assertEqual(proc.returncode, 0, output)
            self.assertIn("Validated 1 queued models", output)
            self.assertIn("no evaluation was launched", output)
            self.assertEqual(
                todo.read_text(encoding="utf-8"), f"# queued model\n{MODEL_NAME}\n"
            )
            self.assertEqual((queue_dir / "eval_done.txt").read_text(), "")
            self.assertEqual((queue_dir / "eval_failed_todo.txt").read_text(), "")
            self.assertEqual(list(queue_dir.glob("eval_queue_last_run_*.json")), [])

    def test_invalid_model_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir)
            (queue_dir / "eval_todo.txt").write_text(
                "streamvln-satnav\n", encoding="utf-8"
            )
            proc = self._run(queue_dir, "--check-queue")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(
                "Invalid SwiftVLN model name",
                proc.stdout + proc.stderr,
            )

    def test_auto_todo_keeps_dynamic_worker_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proc = self._run(
                Path(temp_dir),
                "--check-queue",
                extra_env={
                    "AUTO_TODO": "true",
                    "WAIT_FOR_NEW_TASKS": "true",
                },
            )
            output = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, 0, output)
            self.assertIn("DYNAMIC_TODO", output)
            self.assertIn("no evaluation was launched", output)

    def test_waiting_requires_dynamic_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proc = self._run(
                Path(temp_dir),
                "--check-queue",
                extra_env={"WAIT_FOR_NEW_TASKS": "true"},
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(
                "WAIT_FOR_NEW_TASKS=true requires DYNAMIC_TODO=true",
                proc.stdout + proc.stderr,
            )

    def test_positional_model_list_is_rejected_without_queue_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir)
            proc = self._run(queue_dir, MODEL_NAME)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn("Unknown argument", proc.stdout + proc.stderr)
            self.assertEqual(list(queue_dir.iterdir()), [])

    def test_interactive_wizard_code_is_absent(self):
        source = EVAL_QUEUE.read_text(encoding="utf-8")
        self.assertNotIn("read -p", source)
        self.assertNotIn("interactive_setup", source)
        self.assertNotIn("IFS=';' read -ra MODELS", source)


if __name__ == "__main__":
    unittest.main()
