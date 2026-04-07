import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

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
