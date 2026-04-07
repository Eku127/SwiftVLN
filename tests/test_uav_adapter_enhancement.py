import sys
import tempfile
import unittest
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.common.embedding_enhancement.uav_adapter import (
    UAVAdapterEnhancement,
    resolve_stagea_checkpoint_path,
)
from swiftvln.s2r.model import Sim2RealAdapter


def _write_stagea_checkpoint(path: Path, *, dim: int = 16, num_layers: int = 2, num_heads: int = 4):
    torch.manual_seed(0)
    adapter = Sim2RealAdapter(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    checkpoint = {
        "step": 1,
        "adapter_kwargs": {
            "dim": dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
        },
        "adapter_state_dict": adapter.state_dict(),
    }
    torch.save(checkpoint, path)
    return checkpoint


class UAVAdapterEnhancementTest(unittest.TestCase):
    def test_resolve_stagea_checkpoint_from_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stagea_checkpoint(root / "best.pt")
            _write_stagea_checkpoint(root / "latest.pt")
            resolved = resolve_stagea_checkpoint_path(str(root))
            self.assertEqual(resolved, str(root / "best.pt"))

    def test_load_checkpoint_and_run_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            _write_stagea_checkpoint(checkpoint_path, dim=16)

            module = UAVAdapterEnhancement(embed_dim=16, checkpoint_path=str(checkpoint_path))
            inputs = torch.randn(6, 16)
            outputs = module(inputs, H=2, W=3)

            self.assertEqual(tuple(outputs.shape), (6, 16))
            self.assertEqual(Path(module.loaded_checkpoint_path), checkpoint_path)

    def test_dim_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            _write_stagea_checkpoint(checkpoint_path, dim=32)

            with self.assertRaises(ValueError):
                UAVAdapterEnhancement(embed_dim=16, checkpoint_path=str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
