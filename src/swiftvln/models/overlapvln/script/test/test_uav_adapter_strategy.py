#!/usr/bin/env python3
"""Smoke test for OverlapVLN UAV adapter propagation and external loading."""

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[5]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swiftvln.common.embedding_enhancement.uav_adapter import load_stagea_adapter_checkpoint
from swiftvln.s2r.model import Sim2RealAdapter


def _load_overlap_model_module(repo_root: str):
    module_path = os.path.join(repo_root, 'src', 'swiftvln', 'models', 'overlapvln', 'model.py')
    spec = importlib.util.spec_from_file_location('overlapvln_model_for_test', module_path)
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


class DummyTokenizer:
    def __init__(self):
        self.unk_token_id = 0
        self._next_id = 100
        self._token_to_id = {}

    def add_special_tokens(self, special_tokens_dict):
        tokens = special_tokens_dict.get('additional_special_tokens', [])
        added = 0
        for token in tokens:
            if token not in self._token_to_id:
                self._token_to_id[token] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token, self.unk_token_id)

    def __len__(self):
        return self._next_id


class DummyModel:
    def __init__(self, hidden_size: int):
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=100)
        self.visual = SimpleNamespace(dtype=torch.float32)
        self.pixel_embed = None
        self.pose_embed = None
        self.uav_adapter = None

    def resize_token_embeddings(self, new_vocab_size):
        self.config.vocab_size = new_vocab_size


def _run_case(module, checkpoint_path: str):
    adapter_kwargs, _, resolved_path = load_stagea_adapter_checkpoint(checkpoint_path)
    model = DummyModel(hidden_size=int(adapter_kwargs["dim"]))
    processor = SimpleNamespace(tokenizer=DummyTokenizer())

    with patch('swift.llm.model.model.qwen.get_model_tokenizer_qwen2_5_vl', return_value=(model, processor)):
        loaded_model, _ = module.get_model_tokenizer_overlapvln_qwen2_5_vl(
            model_dir='dummy',
            model_info=SimpleNamespace(),
            model_kwargs={},
            load_model=True,
            use_uav_adapter=True,
            uav_adapter_path=resolved_path,
        )
    return loaded_model, resolved_path


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
            print('PASS: use_uav_adapter/uav_adapter_path are propagated to OverlapVLN model loader.')
            return

    loaded_model, resolved_path = _run_case(module, checkpoint_path)
    _assert_loaded_model(loaded_model, resolved_path)
    print('PASS: use_uav_adapter/uav_adapter_path are propagated to OverlapVLN model loader.')


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
