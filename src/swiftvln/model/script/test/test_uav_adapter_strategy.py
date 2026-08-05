#!/usr/bin/env python3
"""Smoke test for SwiftVLN UAV adapter propagation and external loading."""

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.common.embedding_enhancement.uav_adapter import load_stagea_adapter_checkpoint  # noqa: E402
from swiftvln.s2r.model import Sim2RealAdapter  # noqa: E402


def _load_overlap_model_module(repo_root: str):
    module_path = os.path.join(repo_root, 'src', 'swiftvln', 'model', 'model.py')
    spec = importlib.util.spec_from_file_location('swiftvln_model_for_test', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load module spec from: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_temp_stagea_checkpoint(path: Path, *, dim: int = 64):
    torch.manual_seed(0)
    adapter = Sim2RealAdapter(dim=dim, num_layers=2, num_heads=4, mlp_ratio=2.0, dropout=0.0)
    checkpoint = {
        "step": 1,
        "adapter_kwargs": {
            "dim": dim,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
        },
        "adapter_state_dict": adapter.state_dict(),
    }
    torch.save(checkpoint, path)
    return str(path)


class DummyModel:
    def __init__(self, hidden_size: int):
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=100)
        self.visual = SimpleNamespace(dtype=torch.float32)
        self.pose_embed = None
        self.uav_adapter = None

    def resize_token_embeddings(self, new_vocab_size):
        self.config.vocab_size = new_vocab_size


def _run_case(module, checkpoint_path: str):
    adapter_kwargs, _, resolved_path = load_stagea_adapter_checkpoint(checkpoint_path)
    model = DummyModel(hidden_size=int(adapter_kwargs["dim"]))
    module._attach_embedding_enhancement(
        model,
        model_dir='dummy',
        use_pose_embed=False,
        use_uav_adapter=True,
        uav_adapter_path=resolved_path,
        uav_adapter_type='transformer_v1',
        uav_adapter_apply_scope='all_images',
        pose_fusion_method='additive',
        pose_norm_scale=100.0,
    )
    return model, resolved_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', help='Stage-A checkpoint (.pt or s2r output dir)')
    args = parser.parse_args()

    module = _load_overlap_model_module(str(REPO_ROOT))

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = _write_temp_stagea_checkpoint(Path(tmp) / 'best.pt')
            loaded_model, resolved_path = _run_case(module, checkpoint_path)
            _assert_loaded_model(loaded_model, resolved_path)
            print('PASS: use_uav_adapter/uav_adapter_path are propagated to SwiftVLN model loader.')
            return

    loaded_model, resolved_path = _run_case(module, checkpoint_path)
    _assert_loaded_model(loaded_model, resolved_path)
    print('PASS: use_uav_adapter/uav_adapter_path are propagated to SwiftVLN model loader.')


def _assert_loaded_model(loaded_model, resolved_path: str):
    assert loaded_model.uav_adapter is not None, 'uav_adapter should be attached when use_uav_adapter=True'
    assert hasattr(loaded_model, 'embed_enhance') and 'uav' in loaded_model.embed_enhance.enhancements
    assert loaded_model.uav_adapter.loaded_checkpoint_path == resolved_path

    dim = loaded_model.config.hidden_size
    tokens = torch.randn(16, dim)
    outputs = loaded_model.uav_adapter(tokens, H=4, W=4)
    assert tuple(outputs.shape) == (16, dim), 'uav_adapter forward shape mismatch'


if __name__ == '__main__':
    main()
