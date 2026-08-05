from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


REPO_ROOT = Path(__file__).parents[1]
BUILD_SCRIPT = REPO_ROOT / "packaging/build_inference_wheel.py"


class InferencePackageContractTest(unittest.TestCase):
    def test_wheel_excludes_data_production_and_keeps_runtime_capabilities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "dist"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BUILD_SCRIPT),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            wheels = list(output_dir.glob("swiftvln_inference-*.whl"))
            self.assertEqual(len(wheels), 1)

            with zipfile.ZipFile(wheels[0]) as archive:
                members = set(archive.namelist())
                self.assertFalse(
                    any(
                        member.startswith("swiftvln/s2r/data_generation/")
                        for member in members
                    )
                )
                for required in (
                    "swiftvln/common/embedding_enhancement/uav_adapter.py",
                    "swiftvln/common/env/habitat.py",
                    "swiftvln/common/utils/video_utils.py",
                    "swiftvln/configs/satnav_task.yaml",
                    "swiftvln/configs/vln_r2r.yaml",
                    "swiftvln/habitat_extensions/measures.py",
                    "swiftvln/s2r/model.py",
                ):
                    self.assertIn(required, members)

                entry_points_path = next(
                    member
                    for member in members
                    if member.endswith(".dist-info/entry_points.txt")
                )
                entry_points = archive.read(entry_points_path).decode("utf-8")
                self.assertIn("swiftvln-eval = swiftvln.model.eval:main", entry_points)

                extract_dir = Path(tmpdir) / "unpacked"
                archive.extractall(extract_dir)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(extract_dir)
            help_result = subprocess.run(
                [sys.executable, "-m", "swiftvln", "--help"],
                cwd=tmpdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
            self.assertIn("eval", help_result.stdout)
            self.assertNotIn("s2r-data", help_result.stdout)

    def test_repository_keeps_reproducibility_tools(self):
        self.assertTrue(
            (REPO_ROOT / "src/swiftvln/s2r/data_generation/main.py").is_file()
        )


if __name__ == "__main__":
    unittest.main()
