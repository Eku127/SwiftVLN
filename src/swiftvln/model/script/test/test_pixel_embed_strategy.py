#!/usr/bin/env python3
"""Smoke test for SwiftVLN pixel embedding propagation."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_overlap_model_module(repo_root: str):
    module_path = os.path.join(repo_root, 'src', 'swiftvln', 'model', 'model.py')
    spec = importlib.util.spec_from_file_location('swiftvln_model_for_test', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load module spec from: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=64, vocab_size=100)
        self.visual = SimpleNamespace(dtype=torch.float32)
        self.pixel_embed = None

    def resize_token_embeddings(self, new_vocab_size):
        self.config.vocab_size = new_vocab_size


def _run_case(module, use_pixel_embed: bool, use_pose_embed: bool):
    model = DummyModel()
    module._attach_embedding_enhancement(
        model,
        model_dir='dummy',
        use_pixel_embed=use_pixel_embed,
        use_pose_embed=use_pose_embed,
        use_uav_adapter=False,
        uav_adapter_path='',
        uav_adapter_type='transformer_v1',
        uav_adapter_apply_scope='all_images',
        pose_fusion_method='additive',
        pose_norm_scale=100.0,
    )
    return model


def main():
    module = _load_overlap_model_module(str(REPO_ROOT))

    model_with_pixel = _run_case(module, use_pixel_embed=True, use_pose_embed=False)
    assert model_with_pixel.pixel_embed is not None, 'pixel_embed should be attached when use_pixel_embed=True'
    assert model_with_pixel.pixel_embed.embed_dim == model_with_pixel.config.hidden_size

    model_with_pose = _run_case(module, use_pixel_embed=False, use_pose_embed=True)
    assert hasattr(model_with_pose, 'pose_embed') and model_with_pose.pose_embed is not None, \
        'pose_embed should be attached when use_pose_embed=True'
    assert model_with_pose.pose_embed.embed_dim == model_with_pose.config.hidden_size

    model_without_pixel = _run_case(module, use_pixel_embed=False, use_pose_embed=False)
    assert model_without_pixel.pixel_embed is None, 'pixel_embed should be None when use_pixel_embed=False'
    assert getattr(model_without_pixel, 'pose_embed', None) is None, \
        'pose_embed should be None when use_pose_embed=False'

    print('PASS: use_pixel_embed/use_pose_embed flags are propagated to SwiftVLN embedding attachment.')


if __name__ == '__main__':
    main()
