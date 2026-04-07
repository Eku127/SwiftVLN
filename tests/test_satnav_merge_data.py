import sys
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.scripts.data_process.merge_satnav_data import (
    build_video_rel,
    merge_episode_lists,
    parse_scene_and_episode_from_video,
    transform_annotations,
)


class SatNavMergeDataTest(unittest.TestCase):
    def test_merge_episode_lists_remaps_conflicting_episode_ids(self):
        primary = [
            {
                "episode_id": 0,
                "scene_id": "Berlin-1",
                "instruction": {"instruction_text": "primary"},
            }
        ]
        secondary = [
            {
                "episode_id": "0",
                "scene_id": "Berlin-1",
                "instruction": {"instruction_text": "secondary-conflict"},
            },
            {
                "episode_id": "1",
                "scene_id": "Berlin-1",
                "instruction": {"instruction_text": "secondary-unique"},
            },
        ]

        merged, remap, stats = merge_episode_lists(primary, secondary, "Berlin-1", "ver_260403")

        self.assertEqual(len(merged), 3)
        self.assertEqual(stats["conflicting_overlap"], 1)
        self.assertEqual(stats["unique_secondary_additions"], 1)
        self.assertEqual(stats["remapped_secondary_additions"], 1)
        self.assertEqual(remap[("Berlin-1", "0")], "2")
        self.assertEqual({str(item["episode_id"]) for item in merged}, {"0", "1", "2"})

    def test_merge_episode_lists_dedups_identical_payload(self):
        primary = [
            {
                "episode_id": 0,
                "scene_id": "Rome-1",
                "instruction": {"instruction_text": "same"},
            }
        ]
        secondary = [
            {
                "episode_id": "0",
                "scene_id": "Rome-1",
                "instruction": {"instruction_text": "same"},
            }
        ]

        merged, remap, stats = merge_episode_lists(primary, secondary, "Rome-1", "ver_260403")

        self.assertEqual(len(merged), 1)
        self.assertEqual(remap, {})
        self.assertEqual(stats["identical_overlap"], 1)

    def test_transform_annotations_rewrites_video_path(self):
        annotations = [{"id": 0, "video": "images/Berlin-1_satnav_000000", "steps": 1}]
        transformed = transform_annotations(
            annotations,
            {"images/Berlin-1_satnav_000000": "images/Berlin-1_satnav_001234"},
        )

        self.assertEqual(transformed[0]["video"], "images/Berlin-1_satnav_001234")
        self.assertEqual(annotations[0]["video"], "images/Berlin-1_satnav_000000")

    def test_video_path_helpers_round_trip(self):
        video = build_video_rel("Auckland-1", "123")
        scene_name, episode_id = parse_scene_and_episode_from_video(video)
        self.assertEqual(video, "images/Auckland-1_satnav_000123")
        self.assertEqual(scene_name, "Auckland-1")
        self.assertEqual(episode_id, "000123")


if __name__ == "__main__":
    unittest.main()
