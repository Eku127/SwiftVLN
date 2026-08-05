from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import numpy as np

from swiftvln.common.utils.image_utils import append_text_to_image
from swiftvln.common.utils.video_utils import compress_videos


class VideoContractTest(unittest.TestCase):
    def test_instruction_text_is_rendered_above_or_below_rgb(self):
        rgb = np.zeros((12, 32, 3), dtype=np.uint8)

        top = append_text_to_image(rgb, "go forward", position="top")
        bottom = append_text_to_image(rgb, "go forward", position="bottom")

        self.assertEqual(top.shape, bottom.shape)
        self.assertGreater(top.shape[0], rgb.shape[0])
        self.assertEqual(top.shape[1:], rgb.shape[1:])
        self.assertTrue(np.array_equal(top[-rgb.shape[0] :], rgb))
        self.assertTrue(np.array_equal(bottom[: rgb.shape[0]], rgb))

    def test_video_compression_preserves_files_in_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_dir = Path(tmpdir) / "videos"
            video_dir.mkdir()
            first_video = video_dir / "ep-1_100.mp4"
            second_video = video_dir / "ep-2_0.mp4"
            first_video.write_bytes(b"first-video")
            second_video.write_bytes(b"second-video")

            compress_videos(
                tmpdir,
                [
                    SimpleNamespace(episode_id="ep-1"),
                    {"episode_id": "ep-2"},
                ],
                chunk_size=2,
            )

            archive_path = video_dir / "0-2.zip"
            self.assertTrue(archive_path.is_file())
            self.assertFalse(first_video.exists())
            self.assertFalse(second_video.exists())
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"ep-1_100.mp4", "ep-2_0.mp4"},
                )
                self.assertEqual(archive.read("ep-1_100.mp4"), b"first-video")
                self.assertEqual(archive.read("ep-2_0.mp4"), b"second-video")


if __name__ == "__main__":
    unittest.main()
