#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Saliency Evaluation with Preprocessing Methods

This script evaluates different feature extraction strategies for their 
ability to distinguish between frames (feature saliency/discrimination).

Preprocessing Methods:
1. Direct: No token compression, directly apply pooling
2. ToMe: One-step soft K-means clustering to compress tokens before pooling

Pooling Strategies:
1. Mean Pooling (baseline)
2. Max Pooling
3. Hybrid Pooling (Mean + Max concatenation)
4. GeM Pooling (Generalized Mean)
5. Patch-wise Similarity (no pooling)
6. Centered features (subtract mean)

Evaluation Metrics:
- Surprise Range: max(surprise) - min(surprise)
- Surprise Std: standard deviation of surprise scores
- Turn vs Forward Ratio: average surprise of turn frames / forward frames

Usage:
    python test_feature_saliency.py --model_type dinov2
    python test_feature_saliency.py --model_type dinov2 --target_tokens 64
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# Token Compression Methods (Preprocessing)
# ============================================================

def tome_compress(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """
    One-step Soft K-Means compression (ToMe style).
    
    Uses spatial average pooling result as initial centroids, 
    then applies soft attention for weighted aggregation.
    
    Args:
        tokens: [N, D] patch tokens
        target_tokens: number of tokens after compression
        
    Returns:
        compressed: [target_tokens, D]
    """
    N, D = tokens.shape
    
    if N <= target_tokens:
        return tokens
    
    # Try to find a reasonable spatial layout
    # Assume roughly square layout
    h = int(np.sqrt(N))
    w = N // h
    
    # If not perfectly divisible, pad or truncate
    if h * w < N:
        # Truncate to largest rectangle
        tokens_used = tokens[:h * w]
    elif h * w > N:
        # Reduce h
        h = N // w
        tokens_used = tokens[:h * w]
    else:
        tokens_used = tokens
    
    actual_n = tokens_used.shape[0]
    
    # Calculate target layout
    target_h = int(np.sqrt(target_tokens))
    target_w = target_tokens // target_h
    actual_target = target_h * target_w
    
    # Step 1: Create initial centroids using spatial pooling
    # Reshape to [1, D, h, w] for avg_pool2d
    x_img = tokens_used.view(h, w, D).permute(2, 0, 1).unsqueeze(0)  # [1, D, h, w]
    
    # Compute stride for spatial pooling
    stride_h = max(1, h // target_h)
    stride_w = max(1, w // target_w)
    kernel_h = h - (target_h - 1) * stride_h
    kernel_w = w - (target_w - 1) * stride_w
    
    if kernel_h <= 0 or kernel_w <= 0:
        # Fallback: use adaptive pooling
        centers = F.adaptive_avg_pool2d(x_img, (target_h, target_w))
    else:
        centers = F.avg_pool2d(x_img, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
    
    # Reshape centers: [1, D, th, tw] -> [th*tw, D]
    centers = centers.squeeze(0).permute(1, 2, 0).reshape(-1, D)  # [K, D]
    
    # Step 2: Compute similarity (attention scores)
    # tokens_used: [N', D], centers: [K, D]
    # sim: [N', K]
    sim = torch.mm(tokens_used, centers.T)
    sim = sim * (D ** -0.5)  # Scale
    
    # Step 3: Soft assignment
    attn = F.softmax(sim, dim=0)  # [N', K] - how much each token contributes to each center
    
    # Step 4: Weighted aggregation
    # out = attn.T @ tokens_used -> [K, D]
    out = torch.mm(attn.T, tokens_used)
    
    return out


def direct_pass(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """No compression, return tokens as-is."""
    return tokens


# ============================================================
# Feature Extraction Strategies (Pooling)
# ============================================================

def mean_pooling(tokens: torch.Tensor) -> torch.Tensor:
    """Standard mean pooling. tokens: [N, D] -> [D]"""
    return tokens.mean(dim=0)


def max_pooling(tokens: torch.Tensor) -> torch.Tensor:
    """Max pooling - captures most salient features. tokens: [N, D] -> [D]"""
    return tokens.max(dim=0)[0]


def hybrid_pooling(tokens: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and max. tokens: [N, D] -> [2*D]"""
    mean_feat = tokens.mean(dim=0)
    max_feat = tokens.max(dim=0)[0]
    return torch.cat([mean_feat, max_feat], dim=0)


def gem_pooling(tokens: torch.Tensor, p: float = 3.0) -> torch.Tensor:
    """
    Generalized Mean Pooling.
    p=1: mean pooling
    p->inf: max pooling
    tokens: [N, D] -> [D]
    """
    # Clamp to avoid numerical issues
    tokens_p = tokens.clamp(min=1e-6).pow(p)
    return tokens_p.mean(dim=0).pow(1.0 / p)


def compute_patchwise_similarity(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> float:
    """
    Compute patch-wise similarity without pooling.
    Compares corresponding patches directly.
    tokens_a, tokens_b: [N, D]
    Returns: average similarity across all patches
    """
    # Handle different sizes by truncating to smaller
    min_n = min(tokens_a.shape[0], tokens_b.shape[0])
    tokens_a = tokens_a[:min_n]
    tokens_b = tokens_b[:min_n]
    
    # Normalize
    tokens_a_norm = F.normalize(tokens_a, dim=-1)
    tokens_b_norm = F.normalize(tokens_b, dim=-1)
    
    # Compute similarity for each patch
    sim_per_patch = (tokens_a_norm * tokens_b_norm).sum(dim=-1)  # [N]
    
    return sim_per_patch.mean().item()


def center_features(features: torch.Tensor, mean_feat: torch.Tensor) -> torch.Tensor:
    """Center features by subtracting mean. features: [D], mean_feat: [D]"""
    return features - mean_feat


# ============================================================
# Saliency Metrics
# ============================================================

@dataclass
class SaliencyMetrics:
    """Metrics for evaluating feature saliency."""
    strategy_name: str
    
    # Preprocessing info
    preprocess: str = "direct"
    original_tokens: int = 0
    compressed_tokens: int = 0
    
    # Surprise distribution
    surprise_mean: float = 0.0
    surprise_std: float = 0.0
    surprise_min: float = 0.0
    surprise_max: float = 0.0
    surprise_range: float = 0.0  # max - min
    
    # Similarity distribution
    sim_mean: float = 0.0
    sim_min: float = 0.0
    sim_max: float = 0.0
    
    # Action-based analysis
    turn_surprise_mean: float = 0.0  # avg surprise when action is turn (2 or 3)
    forward_surprise_mean: float = 0.0  # avg surprise when action is forward (1)
    turn_forward_ratio: float = 0.0  # turn / forward (higher = better discrimination)
    
    # Feature dimension
    feature_dim: int = 0


def compute_saliency_metrics(
    similarities: List[float],
    actions: List[int],
    strategy_name: str,
    feature_dim: int,
    preprocess: str = "direct",
    original_tokens: int = 0,
    compressed_tokens: int = 0
) -> SaliencyMetrics:
    """Compute saliency metrics from similarity scores."""
    
    # Surprise = 1 - similarity
    surprises = [1 - s for s in similarities]
    
    # Basic statistics
    metrics = SaliencyMetrics(
        strategy_name=strategy_name,
        preprocess=preprocess,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        surprise_mean=np.mean(surprises),
        surprise_std=np.std(surprises),
        surprise_min=np.min(surprises),
        surprise_max=np.max(surprises),
        surprise_range=np.max(surprises) - np.min(surprises),
        sim_mean=np.mean(similarities),
        sim_min=np.min(similarities),
        sim_max=np.max(similarities),
        feature_dim=feature_dim
    )
    
    # Action-based analysis (skip first frame which has no previous)
    turn_surprises = []
    forward_surprises = []
    
    for i, (surprise, action) in enumerate(zip(surprises, actions[:-1])):
        # action is the action AFTER frame i
        if action in [2, 3]:  # Turn
            turn_surprises.append(surprise)
        elif action == 1:  # Forward
            forward_surprises.append(surprise)
    
    if turn_surprises:
        metrics.turn_surprise_mean = np.mean(turn_surprises)
    if forward_surprises:
        metrics.forward_surprise_mean = np.mean(forward_surprises)
    
    if metrics.forward_surprise_mean > 0:
        metrics.turn_forward_ratio = metrics.turn_surprise_mean / metrics.forward_surprise_mean
    
    return metrics


# ============================================================
# Model Loading
# ============================================================

def load_dinov2_model(device: torch.device):
    """Load DINOv2 model."""
    from transformers import AutoModel, AutoImageProcessor
    
    modelscope_cache = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
    model_path = os.path.join(modelscope_cache, 'facebook/dinov2-base')
    
    if not os.path.exists(model_path):
        from modelscope import snapshot_download
        model_path = snapshot_download('facebook/dinov2-base', cache_dir=modelscope_cache)
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    return model, processor


def load_qwen_model(model_path: str, device: torch.device):
    """Load Qwen2.5-VL model."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    modelscope_cache = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
    cache_path = os.path.join(modelscope_cache, 'models', model_path)
    
    load_path = cache_path if os.path.exists(cache_path) else model_path
    
    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    
    return model, processor


# ============================================================
# Feature Extraction
# ============================================================

def extract_patch_tokens_dinov2(
    model, processor, images: List[Image.Image], device: torch.device
) -> List[torch.Tensor]:
    """Extract patch tokens (before pooling) from DINOv2."""
    all_tokens = []
    
    for img in images:
        inputs = processor(images=[img], return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # [1, num_patches + 1, hidden_size]
            # Skip CLS token (index 0), keep only patch tokens
            patch_tokens = outputs.last_hidden_state[0, 1:, :]  # [N, D]
            all_tokens.append(patch_tokens.float())
    
    return all_tokens


def extract_patch_tokens_qwen(
    model, processor, images: List[Image.Image], device: torch.device
) -> List[torch.Tensor]:
    """Extract patch tokens from Qwen2.5-VL ViT."""
    all_tokens = []
    
    for img in images:
        media_inputs = processor.image_processor(images=[img], return_tensors='pt')
        pixel_values = media_inputs['pixel_values'].to(device, torch.bfloat16)
        image_grid_thw = media_inputs['image_grid_thw'].to(device)
        
        with torch.no_grad():
            vit_features = model.visual(pixel_values, grid_thw=image_grid_thw)
            all_tokens.append(vit_features.float())  # [N, D]
    
    return all_tokens


# ============================================================
# Evaluation
# ============================================================

def evaluate_strategy(
    tokens_list: List[torch.Tensor],
    actions: List[int],
    strategy_name: str,
    pooling_fn=None,
    use_centering: bool = False,
    use_patchwise: bool = False,
    preprocess_fn=None,
    target_tokens: int = 64,
    preprocess_name: str = "direct"
) -> Tuple[SaliencyMetrics, List[float]]:
    """
    Evaluate a feature extraction strategy.
    
    Returns:
        metrics: SaliencyMetrics
        similarities: List of similarity scores between adjacent frames
    """
    num_frames = len(tokens_list)
    similarities = []
    
    original_tokens = tokens_list[0].shape[0]
    
    # Apply preprocessing (compression) if specified
    if preprocess_fn is not None:
        processed_list = [preprocess_fn(tokens, target_tokens) for tokens in tokens_list]
    else:
        processed_list = tokens_list
    
    compressed_tokens = processed_list[0].shape[0]
    
    if use_patchwise:
        # Patch-wise similarity (no pooling)
        feature_dim = processed_list[0].shape[1]
        
        for t in range(1, num_frames):
            sim = compute_patchwise_similarity(processed_list[t], processed_list[t-1])
            similarities.append(sim)
    else:
        # Apply pooling
        features = [pooling_fn(tokens) for tokens in processed_list]
        feature_dim = features[0].shape[0]
        
        # Optionally center features
        if use_centering:
            mean_feat = torch.stack(features).mean(dim=0)
            features = [center_features(f, mean_feat) for f in features]
        
        # Normalize
        features = [F.normalize(f, dim=0) for f in features]
        
        # Compute similarities
        for t in range(1, num_frames):
            sim = torch.dot(features[t], features[t-1]).item()
            similarities.append(sim)
    
    metrics = compute_saliency_metrics(
        similarities, actions, strategy_name, feature_dim,
        preprocess=preprocess_name,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens
    )
    return metrics, similarities


def load_sequence(sequence_path: str) -> Tuple[List[Image.Image], List[int]]:
    """Load images and actions from a sequence directory."""
    images = []
    actions = []
    
    files = sorted([f for f in os.listdir(sequence_path) if f.endswith(('.png', '.jpg'))])
    
    for filename in files:
        # Parse action from filename: XXXXXX_A.png
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        action = int(parts[1])
        
        img = Image.open(os.path.join(sequence_path, filename)).convert('RGB')
        images.append(img)
        actions.append(action)
    
    return images, actions


# ============================================================
# Visualization
# ============================================================

def visualize_comparison(
    results: Dict[str, Tuple[SaliencyMetrics, List[float]]],
    actions: List[int],
    sequence_name: str,
    output_dir: str
):
    """Visualize comparison of different strategies."""
    if not HAS_MATPLOTLIB:
        return
    
    strategies = list(results.keys())
    num_strategies = len(strategies)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Surprise curves comparison
    ax1 = axes[0, 0]
    for strategy, (metrics, sims) in results.items():
        surprises = [1 - s for s in sims]
        label = f'{strategy} (range={metrics.surprise_range:.4f})'
        ax1.plot(range(1, len(surprises) + 1), surprises, label=label, alpha=0.8)
    
    # Mark turn frames
    turn_indices = [i for i, a in enumerate(actions[:-1]) if a in [2, 3]]
    for idx in turn_indices:
        ax1.axvline(x=idx + 1, color='red', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Surprise Score (1 - Cosine Sim)')
    ax1.set_title(f'{sequence_name} - Surprise Curves by Strategy')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Surprise range bar chart
    ax2 = axes[0, 1]
    ranges = [results[s][0].surprise_range for s in strategies]
    colors = plt.cm.tab20(np.linspace(0, 1, num_strategies))
    bars = ax2.bar(range(len(strategies)), ranges, color=colors)
    ax2.set_ylabel('Surprise Range (max - min)')
    ax2.set_title('Feature Discrimination: Surprise Range (Higher = Better)')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars, ranges):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Plot 3: Turn vs Forward ratio
    ax3 = axes[1, 0]
    ratios = [results[s][0].turn_forward_ratio for s in strategies]
    bars = ax3.bar(range(len(strategies)), ratios, color=colors)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal (ratio=1)')
    ax3.set_ylabel('Turn/Forward Surprise Ratio')
    ax3.set_title('Action Discrimination: Turn vs Forward (Higher = Better)')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars, ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary = "Strategy Comparison Summary\n" + "=" * 50 + "\n\n"
    summary += f"{'Strategy':<20} {'Preproc':<8} {'Tokens':<8} {'Range':<8} {'T/F':<6}\n"
    summary += "-" * 50 + "\n"
    
    for strategy in strategies:
        m = results[strategy][0]
        tokens_str = f"{m.compressed_tokens}" if m.compressed_tokens < m.original_tokens else "all"
        summary += f"{strategy:<20} {m.preprocess:<8} {tokens_str:<8} {m.surprise_range:<8.4f} {m.turn_forward_ratio:<6.2f}\n"
    
    # Best strategy
    best_range = max(strategies, key=lambda s: results[s][0].surprise_range)
    best_ratio = max(strategies, key=lambda s: results[s][0].turn_forward_ratio)
    
    summary += "\n" + "=" * 50 + "\n"
    summary += f"Best Range: {best_range}\n"
    summary += f"Best T/F Ratio: {best_ratio}\n"
    
    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sequence_name}_saliency.png'), dpi=150)
    plt.close()


def visualize_summary(
    all_results: Dict[str, Dict[str, SaliencyMetrics]],
    output_dir: str
):
    """Visualize overall summary across all sequences."""
    if not HAS_MATPLOTLIB:
        return
    
    strategies = list(list(all_results.values())[0].keys())
    seq_types = list(all_results.keys())
    
    # Separate direct and tome strategies for cleaner visualization
    direct_strategies = [s for s in strategies if not s.startswith('tome_')]
    tome_strategies = [s for s in strategies if s.startswith('tome_')]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Surprise Range comparison (Direct vs ToMe)
    ax1 = axes[0, 0]
    x = np.arange(len(seq_types))
    width = 0.08
    
    all_strats = direct_strategies[:6] + tome_strategies[:6]  # Limit for readability
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_strats)))
    
    for i, strategy in enumerate(all_strats):
        if strategy in strategies:
            ranges = [all_results[t].get(strategy, SaliencyMetrics(strategy)).surprise_range for t in seq_types]
            offset = (i - len(all_strats)/2 + 0.5) * width
            ax1.bar(x + offset, ranges, width, label=strategy, color=colors[i])
    
    ax1.set_xlabel('Sequence Type')
    ax1.set_ylabel('Surprise Range')
    ax1.set_title('Surprise Range: Direct Pooling vs ToMe Preprocessing')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_types)
    ax1.legend(fontsize=7, loc='upper left', ncol=2)
    
    # Plot 2: Turn/Forward Ratio comparison
    ax2 = axes[0, 1]
    
    for i, strategy in enumerate(all_strats):
        if strategy in strategies:
            ratios = [all_results[t].get(strategy, SaliencyMetrics(strategy)).turn_forward_ratio for t in seq_types]
            offset = (i - len(all_strats)/2 + 0.5) * width
            ax2.bar(x + offset, ratios, width, label=strategy, color=colors[i])
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Sequence Type')
    ax2.set_ylabel('Turn/Forward Ratio')
    ax2.set_title('Turn vs Forward Discrimination: Direct vs ToMe')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_types)
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    
    # Plot 3: Direct vs ToMe comparison (paired)
    ax3 = axes[1, 0]
    
    # Compare same pooling methods: direct vs tome
    pooling_methods = ['mean', 'max', 'mean_centered', 'max_centered']
    x_pool = np.arange(len(pooling_methods))
    width_pool = 0.35
    
    direct_ranges = []
    tome_ranges = []
    
    for pm in pooling_methods:
        direct_key = pm
        tome_key = f'tome_{pm}'
        
        if direct_key in strategies and tome_key in strategies:
            direct_avg = np.mean([all_results[t].get(direct_key, SaliencyMetrics(direct_key)).surprise_range for t in seq_types])
            tome_avg = np.mean([all_results[t].get(tome_key, SaliencyMetrics(tome_key)).surprise_range for t in seq_types])
            direct_ranges.append(direct_avg)
            tome_ranges.append(tome_avg)
        else:
            direct_ranges.append(0)
            tome_ranges.append(0)
    
    bars1 = ax3.bar(x_pool - width_pool/2, direct_ranges, width_pool, label='Direct', color='steelblue')
    bars2 = ax3.bar(x_pool + width_pool/2, tome_ranges, width_pool, label='ToMe', color='coral')
    
    ax3.set_ylabel('Avg Surprise Range')
    ax3.set_title('Direct Pooling vs ToMe Preprocessing (Same Pooling Method)')
    ax3.set_xticks(x_pool)
    ax3.set_xticklabels(pooling_methods)
    ax3.legend()
    
    for bar, val in zip(bars1, direct_ranges):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}', 
                     ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, tome_ranges):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}', 
                     ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Aggregate across all types
    avg_range = {s: np.mean([all_results[t].get(s, SaliencyMetrics(s)).surprise_range for t in seq_types]) for s in strategies}
    avg_ratio = {s: np.mean([all_results[t].get(s, SaliencyMetrics(s)).turn_forward_ratio for t in seq_types]) for s in strategies}
    
    summary = "Overall Best Strategies\n" + "=" * 45 + "\n\n"
    summary += f"{'Strategy':<20} {'Avg Range':<12} {'Avg T/F':<10}\n"
    summary += "-" * 42 + "\n"
    
    # Sort by avg range
    sorted_strategies = sorted(strategies, key=lambda s: avg_range[s], reverse=True)
    
    for s in sorted_strategies[:12]:  # Top 12
        summary += f"{s:<20} {avg_range[s]:<12.4f} {avg_ratio[s]:<10.2f}\n"
    
    best_range = max(strategies, key=lambda s: avg_range[s])
    best_ratio = max(strategies, key=lambda s: avg_ratio[s])
    
    summary += "\n" + "=" * 45 + "\n"
    summary += f"Best Avg Range: {best_range} ({avg_range[best_range]:.4f})\n"
    summary += f"Best Avg T/F:   {best_ratio} ({avg_ratio[best_ratio]:.2f})\n"
    
    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'saliency_summary.png'), dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Feature Saliency Evaluation')
    parser.add_argument('--model_type', type=str, default='dinov2', choices=['qwen', 'dinov2'])
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'test_data', 'habitat'))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_tokens', type=int, default=64,
                        help='Target number of tokens after ToMe compression')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum sequences per type (None = all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Feature Saliency Evaluation with Preprocessing Methods")
    print("=" * 70)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), 'results', f'saliency_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Config] Model: {args.model_type}")
    print(f"[Config] Device: {device}")
    print(f"[Config] Target tokens for ToMe: {args.target_tokens}")
    print(f"[Config] Output: {output_dir}")
    
    # Load model
    print("\n[Step 1] Loading model...")
    if args.model_type == 'dinov2':
        model, processor = load_dinov2_model(device)
        extract_fn = extract_patch_tokens_dinov2
    else:
        model, processor = load_qwen_model(args.model_path, device)
        extract_fn = extract_patch_tokens_qwen
    
    # Define strategies to compare
    # Format: (pooling_fn, use_centering, use_patchwise, preprocess_fn, preprocess_name)
    target_tokens = args.target_tokens
    
    strategies = {
        # Direct pooling (no preprocessing)
        'mean': (mean_pooling, False, False, None, 'direct'),
        'max': (max_pooling, False, False, None, 'direct'),
        'hybrid': (hybrid_pooling, False, False, None, 'direct'),
        'gem_p3': (lambda t: gem_pooling(t, p=3.0), False, False, None, 'direct'),
        'patchwise': (None, False, True, None, 'direct'),
        'mean_centered': (mean_pooling, True, False, None, 'direct'),
        'max_centered': (max_pooling, True, False, None, 'direct'),
        
        # ToMe preprocessing + pooling
        'tome_mean': (mean_pooling, False, False, tome_compress, 'tome'),
        'tome_max': (max_pooling, False, False, tome_compress, 'tome'),
        'tome_hybrid': (hybrid_pooling, False, False, tome_compress, 'tome'),
        'tome_gem_p3': (lambda t: gem_pooling(t, p=3.0), False, False, tome_compress, 'tome'),
        'tome_patchwise': (None, False, True, tome_compress, 'tome'),
        'tome_mean_centered': (mean_pooling, True, False, tome_compress, 'tome'),
        'tome_max_centered': (max_pooling, True, False, tome_compress, 'tome'),
    }
    
    # Find sequences
    print("\n[Step 2] Finding sequences...")
    seq_types = ['simple', 'turns', 'spin']
    all_sequences = {}
    
    for seq_type in seq_types:
        type_dir = os.path.join(args.data_dir, seq_type)
        if os.path.exists(type_dir):
            seqs = [os.path.join(type_dir, d) for d in sorted(os.listdir(type_dir)) 
                    if os.path.isdir(os.path.join(type_dir, d))]
            all_sequences[seq_type] = seqs
            print(f"  {seq_type}: {len(seqs)} sequences")
    
    # Evaluate
    print("\n[Step 3] Evaluating strategies...")
    
    aggregated_results = {t: {s: [] for s in strategies} for t in seq_types}
    detailed_results = []
    
    for seq_type, sequences in all_sequences.items():
        print(f"\n{'='*60}")
        print(f"  {seq_type.upper()}")
        print(f"{'='*60}")
        
        # Use all sequences or limit based on args
        seqs_to_process = sequences if args.max_sequences is None else sequences[:args.max_sequences]
        
        for seq_idx, seq_path in enumerate(seqs_to_process):
            seq_name = os.path.basename(seq_path)
            print(f"\n  [{seq_idx+1}/{len(seqs_to_process)}] Sequence: {seq_name}")
            
            # Load data
            images, actions = load_sequence(seq_path)
            print(f"    Frames: {len(images)}, Turns: {sum(1 for a in actions if a in [2,3])}")
            
            # Extract tokens
            tokens_list = extract_fn(model, processor, images, device)
            print(f"    Original tokens: {tokens_list[0].shape[0]}, Dim: {tokens_list[0].shape[1]}")
            
            # Evaluate each strategy
            seq_results = {}
            for strategy_name, (pooling_fn, use_centering, use_patchwise, preprocess_fn, preprocess_name) in strategies.items():
                metrics, sims = evaluate_strategy(
                    tokens_list, actions, strategy_name,
                    pooling_fn, use_centering, use_patchwise,
                    preprocess_fn, target_tokens, preprocess_name
                )
                seq_results[strategy_name] = (metrics, sims)
                aggregated_results[seq_type][strategy_name].append(metrics)
            
            # Print comparison for key strategies
            print(f"    {'Strategy':<20} {'Preproc':<8} {'Tokens':<8} {'Range':<10} {'T/F Ratio':<10}")
            print(f"    {'-'*56}")
            for sn in ['mean', 'tome_mean', 'max_centered', 'tome_max_centered']:
                if sn in seq_results:
                    m = seq_results[sn][0]
                    tokens_str = f"{m.compressed_tokens}" if m.compressed_tokens < m.original_tokens else "all"
                    print(f"    {sn:<20} {m.preprocess:<8} {tokens_str:<8} {m.surprise_range:<10.4f} {m.turn_forward_ratio:<10.2f}")
            
            # Visualize (only for first few sequences to save time)
            if seq_idx < 2:
                visualize_comparison(seq_results, actions, seq_name, output_dir)
            
            detailed_results.append({
                'sequence': seq_name,
                'type': seq_type,
                'num_frames': len(images),
                'original_tokens': int(tokens_list[0].shape[0]),
                'results': {s: asdict(m) for s, (m, _) in seq_results.items()}
            })
    
    # Aggregate results
    print("\n[Step 4] Aggregating results...")
    
    summary_results = {}
    for seq_type in seq_types:
        summary_results[seq_type] = {}
        for strategy_name in strategies:
            metrics_list = aggregated_results[seq_type][strategy_name]
            if metrics_list:
                avg_metrics = SaliencyMetrics(
                    strategy_name=strategy_name,
                    preprocess=metrics_list[0].preprocess,
                    original_tokens=int(np.mean([m.original_tokens for m in metrics_list])),
                    compressed_tokens=int(np.mean([m.compressed_tokens for m in metrics_list])),
                    surprise_range=np.mean([m.surprise_range for m in metrics_list]),
                    surprise_std=np.mean([m.surprise_std for m in metrics_list]),
                    surprise_mean=np.mean([m.surprise_mean for m in metrics_list]),
                    turn_forward_ratio=np.mean([m.turn_forward_ratio for m in metrics_list]),
                    feature_dim=metrics_list[0].feature_dim
                )
                summary_results[seq_type][strategy_name] = avg_metrics
    
    # Visualize summary
    visualize_summary(summary_results, output_dir)
    
    # Save results
    print("\n[Step 5] Saving results...")
    
    results_dict = {
        'model_type': args.model_type,
        'target_tokens': args.target_tokens,
        'timestamp': timestamp,
        'summary': {
            seq_type: {s: asdict(m) for s, m in type_results.items()}
            for seq_type, type_results in summary_results.items()
        },
        'detailed': detailed_results
    }
    
    with open(os.path.join(output_dir, 'saliency_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Direct Pooling vs ToMe Preprocessing")
    print("=" * 80)
    
    # Group by preprocessing method
    print(f"\n{'Strategy':<22} {'Preproc':<8}", end='')
    for t in seq_types:
        print(f" {t:>8}", end='')
    print(f" {'AVG':>8}")
    print("-" * 80)
    
    for strategy_name in strategies:
        preproc = strategies[strategy_name][4]
        print(f"{strategy_name:<22} {preproc:<8}", end='')
        vals = []
        for t in seq_types:
            if strategy_name in summary_results.get(t, {}):
                val = summary_results[t][strategy_name].surprise_range
                vals.append(val)
                print(f" {val:>8.4f}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        if vals:
            print(f" {np.mean(vals):>8.4f}")
        else:
            print()
    
    # Best strategies
    avg_ranges = {s: np.mean([summary_results[t][s].surprise_range for t in seq_types 
                              if s in summary_results.get(t, {})]) 
                  for s in strategies}
    
    best_direct = max([s for s in strategies if strategies[s][4] == 'direct'], 
                      key=lambda s: avg_ranges[s])
    best_tome = max([s for s in strategies if strategies[s][4] == 'tome'], 
                    key=lambda s: avg_ranges[s])
    best_overall = max(avg_ranges, key=avg_ranges.get)
    
    print("\n" + "=" * 80)
    print(f"Best Direct:  {best_direct:<20} (avg range = {avg_ranges[best_direct]:.4f})")
    print(f"Best ToMe:    {best_tome:<20} (avg range = {avg_ranges[best_tome]:.4f})")
    print(f"Best Overall: {best_overall:<20} (avg range = {avg_ranges[best_overall]:.4f})")
    print(f"\nOutput: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
