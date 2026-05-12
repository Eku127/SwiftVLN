import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.deployment.gpu import GPUInfo, mark_busy_gpus, select_idle_h100
from swiftvln.deployment.model_resolver import (
    DEFAULT_SWIFTVLN_DEPLOY_MODEL_NAME,
    DeploymentModelSpecError,
    SwiftVLNDeploySpec,
    resolve_swiftvln_deploy_spec,
)
from swiftvln.deployment.policy import SwiftVLNBaselinePolicy
from swiftvln.deployment.server import JsonlDeploymentServer
from swiftvln.deployment.session import DeploymentSessionError, SwiftVLNDeploySession


ACTION_TEXT = {0: "STOP", 1: "↑", 2: "←", 3: "→"}


def make_spec(root: Path) -> SwiftVLNDeploySpec:
    return SwiftVLNDeploySpec(
        model_name="swiftvln-satnav-stage1-3b-1ep-f32s4-overlap8-pf-h8-random-b1.0-pool-s2-noembed-bs64-lr2e-5-20260421-123456",
        model_dir=str(root / "model"),
        checkpoint_path=str(root / "checkpoint-100"),
        env_type="satnav",
        stage="stage1",
        model_size="3b",
        num_epochs=1,
        num_frames=32,
        num_future_steps=4,
        num_overlap=8,
        num_history=8,
        log_base=1.0,
        use_random=True,
        compress_stride=2,
        history_method="pool",
        embed_slot="noembed",
    )


class FakePolicy:
    def __init__(self, action_batches):
        self.action_batches = [list(batch) for batch in action_batches]
        self.model = object()
        self.closed = False
        self.reset_calls = 0
        self.observe_calls = 0
        self.completed_actions = []

    def reset(self):
        self.reset_calls += 1
        self.completed_actions = []
        self.observe_calls = 0

    def record_completed_action(self, action):
        self.completed_actions.append(int(action))

    def observe_image(self, image):
        self.observe_calls += 1
        return [0.0, 0.0, 0.0, 1.0]

    def infer_actions(self, instruction, current_image):
        del instruction, current_image
        actions = list(self.action_batches.pop(0)) if self.action_batches else [0]
        raw_text = "".join(ACTION_TEXT[action] for action in actions)
        return SimpleNamespace(raw_action_text=raw_text, actions=actions)

    def snapshot(self):
        return {
            "executed_actions": list(self.completed_actions),
            "observe_calls": self.observe_calls,
            "window_turns": 0,
            "history_cache_size": 0,
        }

    def close(self):
        self.closed = True
        self.model = None


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<history_memory>": 101, "<current_image>": 102}.get(token, 1)

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict):
        del messages, add_generation_prompt, return_tensors, return_dict
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def decode(self, generated_ids, skip_special_tokens=True):
        del generated_ids, skip_special_tokens
        return "not an action"


class FakeImageProcessor:
    merge_size = 2


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()


class FakeEmbedTokens:
    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        return torch.zeros((batch, seq_len, 4), dtype=torch.float32)


class FakeModelForPolicy:
    dtype = torch.float32

    def __init__(self):
        self.model = SimpleNamespace(embed_tokens=FakeEmbedTokens())
        self.embed_enhance = SimpleNamespace(is_empty=True)

    def reset(self, env_num=1):
        self._env_num = env_num

    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[0, 0, 0, 0, 0, 7, 8]], dtype=torch.long)


class SwiftVLNDeploymentResolverTest(unittest.TestCase):
    def test_default_deploy_model_name_is_baseline(self):
        self.assertEqual(
            DEFAULT_SWIFTVLN_DEPLOY_MODEL_NAME,
            "swiftvln-satnav-stage1-3b-1ep-f32s4-overlap0-"
            "pf-h8-b1.0-pool-s2-noembed-data260418-bs64-lr2e-5-20260419-113050",
        )

    def test_resolve_supported_baseline_name(self):
        model_name = (
            "swiftvln-satnav-stage1-3b-1ep-f32s4-overlap8-"
            "pf-h8-random-b1.0-pool-s2-noembed-bs64-lr2e-5-20260421-123456"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "output" / "swiftvln" / model_name
            checkpoint_dir = model_dir / "v2" / "checkpoint-70"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")

            older_checkpoint = model_dir / "v1" / "checkpoint-10"
            older_checkpoint.mkdir(parents=True)
            (older_checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (older_checkpoint / "model.safetensors").write_text("weights\n", encoding="utf-8")

            spec = resolve_swiftvln_deploy_spec(root, model_name, output_root=root / "output")

        self.assertEqual(spec.env_type, "satnav")
        self.assertTrue(spec.use_random)
        self.assertEqual(spec.num_overlap, 8)
        self.assertEqual(spec.num_history, 8)
        self.assertTrue(spec.checkpoint_path.endswith("v2/checkpoint-70"))

    def test_reject_unsupported_variants(self):
        cases = [
            "swiftvln-satnav-stage1-3b-1ep-f32s4-overlap8-map-g1000-l400-r448-d20-s2-noembed-bs64-lr2e-5-20260421-123456",
            "swiftvln-satnav-stage1-3b-1ep-f32s4-overlap8-gtc-k512-noembed-bs64-lr2e-5-20260421-123456",
            "swiftvln-satnav-stage1-3b-1ep-f32s4-overlap8-pf-h8-b1.0-tome-s2-noembed-bs64-lr2e-5-20260421-123456",
            "swiftvln-satnav-stage1-3b-1ep-f32s4-overlap8-pf-h8-b1.0-pool-s2-pixel-bs64-lr2e-5-20260421-123456",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for model_name in cases:
                with self.subTest(model_name=model_name):
                    with self.assertRaises(DeploymentModelSpecError):
                        resolve_swiftvln_deploy_spec(root, model_name, output_root=root / "output")


class SwiftVLNDeploymentGPUTest(unittest.TestCase):
    def test_select_idle_h100_prefers_lowest_memory_then_index(self):
        gpus = [
            GPUInfo(index=2, uuid="GPU-2", name="NVIDIA H100 80GB HBM3", memory_used_mib=512),
            GPUInfo(index=0, uuid="GPU-0", name="NVIDIA H100 80GB HBM3", memory_used_mib=256),
            GPUInfo(index=1, uuid="GPU-1", name="NVIDIA A100 80GB PCIe", memory_used_mib=0),
        ]
        marked = mark_busy_gpus(gpus, {"GPU-2"})
        selected = select_idle_h100(marked)
        self.assertEqual(selected.index, 0)


class SwiftVLNDeploymentSessionTest(unittest.TestCase):
    def _create_image(self, path: Path, color: tuple[int, int, int]):
        Image.new("RGB", (8, 8), color).save(path)

    def test_session_state_machine_and_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = make_spec(root)
            policy = FakePolicy([[1, 2], [3]])
            session = SwiftVLNDeploySession(spec, policy, root / "sessions")

            img1 = root / "img1.jpg"
            img2 = root / "img2.jpg"
            img3 = root / "img3.jpg"
            self._create_image(img1, (255, 0, 0))
            self._create_image(img2, (0, 255, 0))
            self._create_image(img3, (0, 0, 255))

            start = session.start("Drive to the bank.", session_id="demo")
            self.assertEqual(start["state"], "waiting_image")

            first = session.handle_image(str(img1))
            self.assertTrue(first["performed_inference"])
            self.assertEqual(first["actions"], [1, 2])
            self.assertEqual(first["next_action"], 1)
            self.assertEqual(first["remaining_actions"], [1, 2])
            self.assertTrue(Path(first["stored_image_path"]).is_file())

            second = session.handle_image(str(img2))
            self.assertFalse(second["performed_inference"])
            self.assertEqual(second["completed_action"], 1)
            self.assertEqual(second["remaining_actions"], [2])

            third = session.handle_image(str(img3))
            self.assertTrue(third["performed_inference"])
            self.assertEqual(third["completed_action"], 2)
            self.assertEqual(third["actions"], [3])
            self.assertEqual(third["remaining_actions"], [3])
            self.assertTrue(Path(third["stored_image_path"]).name.endswith("feedback_infer_input.jpg"))

            summary = session.end("done")
            self.assertEqual(summary["state"], "closed")
            self.assertTrue(policy.closed)

            session_dir = Path(start["session_dir"])
            self.assertTrue((session_dir / "session_meta.json").is_file())
            self.assertTrue((session_dir / "events.jsonl").is_file())
            self.assertTrue((session_dir / "session_summary.json").is_file())
            self.assertEqual(len(list((session_dir / "images").iterdir())), 3)

            with self.assertRaises(DeploymentSessionError):
                session.handle_image(str(img1))


class SwiftVLNDeploymentPolicyTest(unittest.TestCase):
    def test_empty_parse_falls_back_to_stop(self):
        spec = make_spec(Path("/tmp"))
        policy = SwiftVLNBaselinePolicy(FakeModelForPolicy(), FakeProcessor(), spec)
        policy._build_complete_prompt_embeds = lambda **kwargs: (
            torch.zeros((1, 5, 4), dtype=torch.float32),
            5,
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([1, 2, 2], dtype=torch.long),
        )
        policy._save_turn_to_window = lambda **kwargs: None

        image = Image.new("RGB", (8, 8), (255, 255, 255))
        policy.observe_image(image)
        result = policy.infer_actions("Drive forward.", image)
        self.assertEqual(result.actions, [0])


class SwiftVLNDeploymentServerTest(unittest.TestCase):
    def _create_image(self, path: Path):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(path)

    def test_server_handles_invalid_json_and_invalid_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = make_spec(root)
            policy = FakePolicy([[1]])
            session = SwiftVLNDeploySession(spec, policy, root / "sessions")
            server = JsonlDeploymentServer(session=session, spec=spec)
            try:
                invalid_json = server._handle_line("not-json")
                self.assertFalse(invalid_json["ok"])
                self.assertEqual(invalid_json["type"], "error")

                image_before_start = server._handle_line(
                    json.dumps({"type": "image", "image_path": "/tmp/missing.jpg"})
                )
                self.assertFalse(image_before_start["ok"])
                self.assertEqual(image_before_start["state"], "waiting_start")

                start = server._handle_line(json.dumps({"type": "start", "instruction": "Drive."}))
                self.assertTrue(start["ok"])

                missing = server._handle_line(
                    json.dumps({"type": "image", "image_path": "/tmp/missing.jpg"})
                )
                self.assertFalse(missing["ok"])
                self.assertEqual(missing["type"], "error")
            finally:
                session.close_if_needed()

    def test_server_run_emits_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = make_spec(root)
            policy = FakePolicy([[0]])
            session = SwiftVLNDeploySession(spec, policy, root / "sessions")
            server = JsonlDeploymentServer(session=session, spec=spec)

            image_path = root / "frame.jpg"
            self._create_image(image_path)

            input_stream = io.StringIO(
                json.dumps({"type": "start", "instruction": "Drive."})
                + "\n"
                + json.dumps({"type": "image", "image_path": str(image_path)})
                + "\n"
                + json.dumps({"type": "end", "reason": "done"})
                + "\n"
            )
            output_stream = io.StringIO()
            rc = server.run(input_stream=input_stream, output_stream=output_stream)

            lines = [json.loads(line) for line in output_stream.getvalue().splitlines() if line.strip()]
            self.assertEqual(rc, 0)
            self.assertEqual(lines[0]["type"], "ready")
            self.assertEqual(lines[-1]["state"], "closed")


if __name__ == "__main__":
    unittest.main()
