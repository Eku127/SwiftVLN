import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.s2r.losses import bidirectional_contrastive_loss, compute_retrieval_metrics, global_cosine_loss
from swiftvln.s2r.model import (
    ProjectionHead,
    Sim2RealAdapter,
    TeacherVisionTower,
    masked_mean_pool,
    pad_token_sequences,
    split_visual_embeddings,
)


def _distributed_contrastive_gradient_worker(rank, world_size, init_file, output_dir):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(123)
        all_uav = torch.randn(world_size * 2, 4)
        all_sat = torch.randn(world_size * 2, 4)
        start = rank * 2
        local_uav = all_uav[start:start + 2].clone().requires_grad_(True)
        local_sat = all_sat[start:start + 2].clone().requires_grad_(True)
        loss = bidirectional_contrastive_loss(
            local_uav,
            local_sat,
            temperature=0.2,
            gather_distributed=True,
        )
        loss.backward()
        torch.save(
            {
                "uav_grad": local_uav.grad,
                "sat_grad": local_sat.grad,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


class S2RModelLossTest(unittest.TestCase):
    def test_split_visual_embeddings_and_padding(self):
        all_embeddings = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long)
        outputs = split_visual_embeddings(all_embeddings, grid, merge_size=2)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), (4, 3))
        self.assertEqual(tuple(outputs[1].shape), (2, 3))

        padded, mask = pad_token_sequences(outputs, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(tuple(padded.shape), (2, 4, 3))
        self.assertEqual(mask.tolist(), [[True, True, True, True], [True, True, False, False]])

    def test_adapter_projection_and_losses_are_finite(self):
        torch.manual_seed(0)
        tokens = torch.randn(2, 5, 8)
        mask = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])

        adapter = Sim2RealAdapter(dim=8, num_layers=2, num_heads=2, mlp_ratio=2.0, dropout=0.0)
        projection = ProjectionHead(input_dim=8, hidden_dim=8, output_dim=4)

        aligned = adapter(tokens, attention_mask=mask)
        pooled = masked_mean_pool(aligned, mask)
        sat = torch.randn_like(pooled)

        uav_proj = torch.nn.functional.normalize(projection(pooled), dim=-1)
        sat_proj = torch.nn.functional.normalize(projection(sat), dim=-1)

        contrast = bidirectional_contrastive_loss(uav_proj, sat_proj, temperature=0.07, gather_distributed=False)
        cosine = global_cosine_loss(pooled, sat)

        self.assertEqual(tuple(aligned.shape), tuple(tokens.shape))
        self.assertEqual(tuple(pooled.shape), (2, 8))
        self.assertTrue(torch.isfinite(contrast))
        self.assertTrue(torch.isfinite(cosine))

    def test_retrieval_metrics_one_to_one(self):
        uav = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        sat = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        metrics = compute_retrieval_metrics(uav, sat, topk=(1, 2))
        self.assertEqual(metrics["u2s_r@1"], 1.0)
        self.assertEqual(metrics["s2u_r@1"], 1.0)
        self.assertEqual(metrics["u2s_r@2"], 1.0)
        self.assertGreater(metrics["paired_cosine_mean"], 0.99)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "gloo unavailable")
    def test_distributed_gather_preserves_remote_key_gradients(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(Path(tmp) / "dist_init")
            mp.spawn(
                _distributed_contrastive_gradient_worker,
                args=(world_size, init_file, tmp),
                nprocs=world_size,
                join=True,
            )

            actual = [
                torch.load(Path(tmp) / f"rank_{rank}.pt", weights_only=True)
                for rank in range(world_size)
            ]

        torch.manual_seed(123)
        all_uav = torch.randn(world_size * 2, 4, requires_grad=True)
        all_sat = torch.randn(world_size * 2, 4, requires_grad=True)
        reference_loss = 0.0
        for rank in range(world_size):
            start = rank * 2
            local_uav = all_uav[start:start + 2]
            local_sat = all_sat[start:start + 2]
            labels = torch.arange(2) + start
            logits_u2s = local_uav @ all_sat.t() / 0.2
            logits_s2u = local_sat @ all_uav.t() / 0.2
            reference_loss = reference_loss + 0.5 * (
                torch.nn.functional.cross_entropy(logits_u2s, labels)
                + torch.nn.functional.cross_entropy(logits_s2u, labels)
            )
        reference_loss.backward()

        for rank in range(world_size):
            start = rank * 2
            torch.testing.assert_close(
                actual[rank]["uav_grad"],
                all_uav.grad[start:start + 2],
            )
            torch.testing.assert_close(
                actual[rank]["sat_grad"],
                all_sat.grad[start:start + 2],
            )

    def test_teacher_vision_tower_uses_qwen25_loader(self):
        class DummyVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        class DummyModel(torch.nn.Module):
            last_call = None

            def __init__(self):
                super().__init__()
                self.visual = DummyVisual()
                self.config = types.SimpleNamespace(hidden_size=32)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.last_call = (args, kwargs)
                return cls()

        class DummyProcessor:
            last_call = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.last_call = (args, kwargs)
                return types.SimpleNamespace(
                    image_processor=types.SimpleNamespace(merge_size=2),
                )

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=DummyProcessor,
            Qwen2_5_VLForConditionalGeneration=DummyModel,
        )

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            tower = TeacherVisionTower.from_pretrained(
                "dummy-model",
                device=torch.device("cpu"),
                dtype_name="bf16",
            )

        self.assertEqual(tower.hidden_size, 32)
        self.assertEqual(tower.merge_size, 2)
        self.assertEqual(tower.dtype, torch.float32)
        self.assertFalse(any(parameter.requires_grad for parameter in tower.visual.parameters()))
        self.assertEqual(DummyProcessor.last_call[0], ("dummy-model",))
        self.assertTrue(DummyProcessor.last_call[1]["trust_remote_code"])
        self.assertEqual(DummyModel.last_call[0], ("dummy-model",))
        self.assertTrue(DummyModel.last_call[1]["trust_remote_code"])
        self.assertTrue(DummyModel.last_call[1]["low_cpu_mem_usage"])
        self.assertEqual(DummyModel.last_call[1]["torch_dtype"], torch.float32)


if __name__ == "__main__":
    unittest.main()
