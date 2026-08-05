from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_QUEUE = REPO_ROOT / "src/swiftvln/scripts/train/train_queue.sh"


class TrainQueueProtocolTest(unittest.TestCase):
    def _check_config(self, config_text: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir, "experiments.sh")
            config_path.write_text(config_text, encoding="utf-8")
            env = os.environ.copy()
            env["TRAIN_EXPERIMENTS_FILE"] = str(config_path)
            return subprocess.run(
                ["bash", str(TRAIN_QUEUE), "--check-config"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_valid_file_backed_queue_config_does_not_launch_training(self):
        proc = self._check_config(
            """
ENV_TYPE="satnav"
USE_SWANLAB="false"
EXPERIMENTS=(
  "swiftvln|NUM_HISTORY=8,COMPRESS_STRIDE=2|history pool|SatNav|/tmp/data"
)
"""
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("Validated 1 experiments", output)
        self.assertIn("no training was launched", output)

    def test_removed_shortcode_is_rejected(self):
        proc = self._check_config(
            """
ENV_TYPE="satnav"
EXPERIMENTS=(
  "swiftvln|a=32|legacy shortcode|SatNav|/tmp/data"
)
"""
        )
        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertIn("Invalid KEY=VALUE override: a=32", output)

    def test_interactive_wizard_code_is_absent(self):
        source = TRAIN_QUEUE.read_text(encoding="utf-8")
        self.assertNotIn("read -p", source)
        self.assertNotIn("interactive_setup", source)
        self.assertNotIn("expand_shortcodes", source)


if __name__ == "__main__":
    unittest.main()
