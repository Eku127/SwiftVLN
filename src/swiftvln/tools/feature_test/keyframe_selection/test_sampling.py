#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cosine Surprise-based Keyframe Selection Test (v4 - Diversity Constrained)

This script compares multiple frame sampling strategies for VLN history:
1. Uniform Sampling (Baseline): Sample frames at fixed intervals
2. Pure Top-K Surprise: Top-K sampling based on cosine distance (original)
3. Segmented Top-K: Divide sequence into K segments, pick highest surprise in each
4. Gap-Constrained Top-K: Top-K with minimum frame gap constraint
5. Hybrid: Uniform base + surprise-weighted adjustment
6. Diversity-Constrained Top-K: Greedy selection balancing surprise and diversity

Feature Processing (from saliency test findings):
- ToMe preprocessing: Soft K-Means token compression (256 -> 64)
- Max Pooling: Capture most salient features
- Centering: Subtract global mean to increase discrimination

Metrics:
- FRE (Feature Reconstruction Error): Average min distance to selected frames
- FRE Max: Maximum reconstruction error (worst case)
- Semantic Diversity Score: Average pairwise dissimilarity of selected frames
- Feature Discrimination Analysis: Analyze ViT feature quality

Test Data Categories:
- simple: Straight walking (low information density)
- turns: Walking with turns (high surprise, core test)
- spin: In-place rotation (stress test)

Usage:
    python test_sampling.py --model_type dinov2 --budget 4 --use_tome --use_centering
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

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not available, skipping visualization")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class FrameInfo:
    """Information about a single frame."""
    index: int
    filename: str
    action: int  # 0=stop, 1=forward, 2=left, 3=right
    path: str


@dataclass
class MethodResult:
    """Result for a single sampling method."""
    method_name: str
    selected_indices: List[int]
    fre: float
    fre_max: float = 0.0  # Maximum reconstruction error (worst case)
    diversity_score: float = 0.0  # Semantic diversity of selected frames


@dataclass
class SequenceResult:
    """Result for a single test sequence."""
    sequence_name: str
    sequence_type: str  # simple, turns, spin
    num_frames: int
    budget: int
    
    # Surprise scores (shared across methods)
    surprise_scores: List[float] = field(default_factory=list)
    
    # Results for each method
    method_results: Dict[str, MethodResult] = field(default_factory=dict)
    
    # Feature discrimination metrics
    avg_cosine_sim: float = 0.0
    min_cosine_sim: float = 0.0
    max_cosine_sim: float = 0.0
    std_cosine_sim: float = 0.0


@dataclass
class TestReport:
    """Complete test report."""
    model_name: str
    budget: int
    timestamp: str
    results: List[SequenceResult] = field(default_factory=list)
    
    # Aggregated metrics by sequence type
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Feature discrimination summary
    feature_discrimination: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Token Compression (ToMe) - From Saliency Test Findings
# ============================================================

def tome_compress(tokens: torch.Tensor, target_tokens: int = 64) -> torch.Tensor:
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
    h = int(np.sqrt(N))
    w = N // h
    
    if h * w < N:
        tokens_used = tokens[:h * w]
    elif h * w > N:
        h = N // w
        tokens_used = tokens[:h * w]
    else:
        tokens_used = tokens
    
    actual_n = tokens_used.shape[0]
    
    # Calculate target layout
    target_h = int(np.sqrt(target_tokens))
    target_w = target_tokens // target_h
    
    # Create initial centroids using spatial pooling
    x_img = tokens_used.view(h, w, D).permute(2, 0, 1).unsqueeze(0)  # [1, D, h, w]
    
    # Compute stride for spatial pooling
    stride_h = max(1, h // target_h)
    stride_w = max(1, w // target_w)
    kernel_h = h - (target_h - 1) * stride_h
    kernel_w = w - (target_w - 1) * stride_w
    
    if kernel_h <= 0 or kernel_w <= 0:
        centers = F.adaptive_avg_pool2d(x_img, (target_h, target_w))
    else:
        centers = F.avg_pool2d(x_img, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
    
    centers = centers.squeeze(0).permute(1, 2, 0).reshape(-1, D)  # [K, D]
    
    # Compute similarity (attention scores)
    sim = torch.mm(tokens_used, centers.T)
    sim = sim * (D ** -0.5)
    
    # Soft assignment
    attn = F.softmax(sim, dim=0)
    
    # Weighted aggregation
    out = torch.mm(attn.T, tokens_used)
    
    return out


# ============================================================
# Sampling Strategies
# ============================================================

def uniform_sampling(num_frames: int, budget: int) -> List[int]:
    """
    Uniform sampling strategy (baseline).
    """
    if budget >= num_frames:
        return list(range(num_frames))
    
    indices = np.linspace(0, num_frames - 1, budget, dtype=int)
    return sorted(set(indices.tolist()))


def compute_surprise_scores(features: torch.Tensor) -> List[float]:
    """
    Compute surprise scores for all frames.
    Score_t = 1 - CosineSim(f_t, f_{t-1})
    First frame gets inf score (must be selected).
    """
    num_frames = features.shape[0]
    features_norm = F.normalize(features, dim=-1)
    
    scores = [float('inf')]  # First frame must be selected
    
    for t in range(1, num_frames):
        sim = torch.dot(features_norm[t], features_norm[t - 1]).item()
        surprise = 1.0 - sim
        scores.append(surprise)
    
    return scores


def pure_topk_sampling(scores: List[float], budget: int) -> List[int]:
    """
    Pure Top-K selection by surprise score (original method).
    Problem: May cause temporal clustering.
    """
    num_frames = len(scores)
    if budget >= num_frames:
        return list(range(num_frames))
    
    score_array = np.array(scores)
    score_array[0] = np.inf  # Ensure first frame is always selected
    
    top_k_indices = np.argsort(score_array)[-budget:]
    return sorted(top_k_indices.tolist())


def segmented_topk_sampling(scores: List[float], budget: int) -> List[int]:
    """
    Segmented Top-K: Divide sequence into K segments, pick highest surprise in each.
    Guarantees temporal coverage.
    """
    num_frames = len(scores)
    if budget >= num_frames:
        return list(range(num_frames))
    
    segment_size = num_frames / budget
    selected = []
    
    for i in range(budget):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size) if i < budget - 1 else num_frames
        
        # Find highest surprise in segment
        segment_scores = scores[start:end]
        # Handle inf in first segment
        finite_scores = [s if s != float('inf') else -1 for s in segment_scores]
        best_offset = np.argmax(finite_scores)
        
        # If first segment, prefer frame 0 (anchor)
        if i == 0 and scores[0] == float('inf'):
            best_offset = 0
        
        selected.append(start + best_offset)
    
    return sorted(set(selected))


def gap_constrained_topk_sampling(scores: List[float], budget: int, min_gap: int = 3) -> List[int]:
    """
    Top-K with minimum gap constraint between selected frames.
    Prevents temporal clustering while prioritizing high surprise.
    """
    num_frames = len(scores)
    if budget >= num_frames:
        return list(range(num_frames))
    
    selected = [0]  # First frame always selected
    
    # Create score array, excluding already selected
    remaining_budget = budget - 1
    
    while remaining_budget > 0:
        best_idx = -1
        best_score = -1
        
        for i in range(num_frames):
            if i in selected:
                continue
            
            # Check gap constraint
            if all(abs(i - s) >= min_gap for s in selected):
                score = scores[i] if scores[i] != float('inf') else 0
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        if best_idx == -1:
            # No valid candidate with gap constraint, relax constraint
            for i in range(num_frames):
                if i not in selected:
                    score = scores[i] if scores[i] != float('inf') else 0
                    if score > best_score:
                        best_score = score
                        best_idx = i
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        remaining_budget -= 1
    
    return sorted(selected)


def hybrid_sampling(scores: List[float], budget: int, window_ratio: float = 0.3) -> List[int]:
    """
    Hybrid: Use uniform points as anchors, then adjust within local window based on surprise.
    Combines temporal coverage with surprise-awareness.
    """
    num_frames = len(scores)
    if budget >= num_frames:
        return list(range(num_frames))
    
    # Get uniform anchor points
    uniform_anchors = np.linspace(0, num_frames - 1, budget)
    
    # Window size based on segment size
    segment_size = num_frames / budget
    window = max(1, int(segment_size * window_ratio))
    
    selected = []
    for anchor in uniform_anchors:
        anchor = int(anchor)
        start = max(0, anchor - window)
        end = min(num_frames, anchor + window + 1)
        
        # Find highest surprise in window
        best_idx = anchor
        best_score = -1
        
        for i in range(start, end):
            score = scores[i] if scores[i] != float('inf') else (np.max([s for s in scores if s != float('inf')]) * 1.5 if i == 0 else 0)
            if score > best_score and i not in selected:
                best_score = score
                best_idx = i
        
        selected.append(best_idx)
    
    # Ensure frame 0 is included
    if 0 not in selected:
        selected[0] = 0
    
    return sorted(set(selected))


def diversity_constrained_topk_sampling(
    features: torch.Tensor, 
    scores: List[float], 
    budget: int, 
    diversity_weight: float = 0.3
) -> List[int]:
    """
    Diversity-Constrained Top-K: Greedy selection balancing surprise and diversity.
    
    At each step, select the frame that maximizes:
        combined_score = (1 - α) * surprise + α * diversity
    
    where:
        - surprise = normalized surprise score
        - diversity = min cosine distance to already selected frames
        - α = diversity_weight
    
    This prevents selecting multiple similar frames (e.g., consecutive turn frames).
    
    Args:
        features: [T, D] frame features (normalized recommended)
        scores: Surprise scores for each frame
        budget: Number of frames to select
        diversity_weight: Weight for diversity term (0 = pure surprise, 1 = pure diversity)
        
    Returns:
        List of selected frame indices
    """
    num_frames = len(scores)
    if budget >= num_frames:
        return list(range(num_frames))
    
    # Normalize features for cosine similarity
    features_norm = F.normalize(features, dim=-1)
    
    # Normalize surprise scores to [0, 1] (excluding inf)
    finite_scores = [s for s in scores if s != float('inf')]
    if finite_scores:
        score_min = min(finite_scores)
        score_max = max(finite_scores)
        score_range = score_max - score_min if score_max > score_min else 1.0
    else:
        score_min, score_range = 0, 1.0
    
    normalized_scores = []
    for s in scores:
        if s == float('inf'):
            normalized_scores.append(1.0)  # First frame gets max normalized score
        else:
            normalized_scores.append((s - score_min) / score_range)
    
    # Greedy selection
    selected = []
    
    # First frame always selected
    selected.append(0)
    
    while len(selected) < budget:
        best_idx = -1
        best_combined = -float('inf')
        
        for t in range(num_frames):
            if t in selected:
                continue
            
            # Surprise component
            surprise_score = normalized_scores[t]
            
            # Diversity component: min distance (1 - max similarity) to selected frames
            if selected:
                selected_features = features_norm[selected]
                similarities = torch.mv(selected_features, features_norm[t])
                max_sim = similarities.max().item()
                diversity_score = 1.0 - max_sim  # Higher when more different
            else:
                diversity_score = 1.0
            
            # Combined score
            combined = (1 - diversity_weight) * surprise_score + diversity_weight * diversity_score
            
            if combined > best_combined:
                best_combined = combined
                best_idx = t
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
    
    return sorted(selected)


# ============================================================
# Feature Extraction
# ============================================================

def load_model(model_path: str, device: torch.device, model_type: str = 'qwen'):
    """
    Load model for feature extraction.
    
    Args:
        model_path: Path to model
        device: CUDA device
        model_type: 'qwen' for Qwen2.5-VL, 'dinov2' for DINOv2
    """
    if model_type == 'dinov2':
        return load_dinov2_model(model_path, device)
    else:
        return load_qwen_model(model_path, device)


def load_qwen_model(model_path: str, device: torch.device):
    """Load Qwen2.5-VL model for ViT feature extraction."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print(f"[Loading] Qwen model from {model_path}...")
    
    # Check ModelScope cache first
    modelscope_cache = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
    model_cache_path = os.path.join(modelscope_cache, 'models', model_path)
    
    if os.path.exists(model_cache_path):
        print(f"[Loading] Found model in ModelScope cache: {model_cache_path}")
        load_path = model_cache_path
    else:
        load_path = model_path
    
    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"[Loading] Qwen model loaded successfully, dtype: {model.dtype}")
    return model, processor


def load_dinov2_model(model_path: str, device: torch.device):
    """Load DINOv2 model for feature extraction."""
    from transformers import AutoModel, AutoImageProcessor
    
    print(f"[Loading] DINOv2 model from {model_path}...")
    
    # Check ModelScope cache first
    modelscope_cache = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
    model_cache_path = os.path.join(modelscope_cache, model_path)
    
    if os.path.exists(model_cache_path):
        print(f"[Loading] Found DINOv2 in ModelScope cache: {model_cache_path}")
        load_path = model_cache_path
    else:
        # Try to download from ModelScope
        try:
            from modelscope import snapshot_download
            print(f"[Loading] Downloading DINOv2 from ModelScope...")
            load_path = snapshot_download(model_path, cache_dir=modelscope_cache)
            print(f"[Loading] Downloaded to: {load_path}")
        except Exception as e:
            print(f"[Warning] ModelScope download failed: {e}")
            print(f"[Loading] Trying direct path: {model_path}")
            load_path = model_path
    
    processor = AutoImageProcessor.from_pretrained(load_path)
    model = AutoModel.from_pretrained(load_path)
    model = model.to(device)
    model.eval()
    
    # Get model info
    hidden_size = model.config.hidden_size
    patch_size = model.config.patch_size
    num_patches_per_side = 224 // patch_size  # for 224x224 input
    total_tokens = num_patches_per_side ** 2 + 1  # patches + CLS
    print(f"[Loading] DINOv2 loaded successfully:")
    print(f"          - hidden_size: {hidden_size}")
    print(f"          - patch_size: {patch_size}")
    print(f"          - tokens per image: {total_tokens} ({num_patches_per_side}x{num_patches_per_side} patches + 1 CLS)")
    
    return model, processor


def extract_features(
    model,
    processor,
    images: List[Image.Image],
    device: torch.device,
    batch_size: int = 8,
    pooling: str = 'mean',
    model_type: str = 'qwen',
    use_tome: bool = False,
    tome_target: int = 64,
    use_centering: bool = False
) -> torch.Tensor:
    """
    Extract features for a list of images.
    
    Args:
        pooling: Feature aggregation method
            - 'mean': Global Average Pooling (default)
            - 'max': Global Max Pooling
            - 'cls': Use CLS token
        model_type: 'qwen' or 'dinov2'
        use_tome: Apply ToMe compression before pooling
        tome_target: Target number of tokens after ToMe
        use_centering: Apply centering (subtract global mean) after pooling
    """
    if model_type == 'dinov2':
        features = extract_dinov2_features(model, processor, images, device, batch_size, pooling, use_tome, tome_target)
    else:
        features = extract_qwen_features(model, processor, images, device, batch_size, pooling, use_tome, tome_target)
    
    # Apply centering if requested
    if use_centering:
        mean_feat = features.mean(dim=0, keepdim=True)
        features = features - mean_feat
    
    return features


def extract_qwen_features(
    model,
    processor,
    images: List[Image.Image],
    device: torch.device,
    batch_size: int = 8,
    pooling: str = 'mean',
    use_tome: bool = False,
    tome_target: int = 64
) -> torch.Tensor:
    """Extract Qwen2.5-VL ViT features with optional ToMe compression."""
    all_features = []
    
    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]
        
        # Process images
        media_inputs = processor.image_processor(
            images=batch_images,
            return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(device, torch.bfloat16)
        image_grid_thw = media_inputs['image_grid_thw'].to(device)
        
        # Extract ViT features
        with torch.no_grad():
            vit_features = model.visual(pixel_values, grid_thw=image_grid_thw)
        
        # Split features by image and compute global feature
        merge_size = processor.image_processor.merge_size
        merge_length = merge_size ** 2
        
        embed_idx = 0
        for i in range(len(batch_images)):
            num_tokens = int(image_grid_thw[i].prod() // merge_length)
            frame_features = vit_features[embed_idx:embed_idx + num_tokens].float()
            embed_idx += num_tokens
            
            # Apply ToMe compression if requested
            if use_tome:
                frame_features = tome_compress(frame_features, tome_target)
            
            # Apply pooling strategy
            if pooling == 'mean':
                global_feature = frame_features.mean(dim=0)
            elif pooling == 'max':
                global_feature = frame_features.max(dim=0)[0]
            elif pooling == 'cls':
                global_feature = frame_features[0]
            else:
                global_feature = frame_features.mean(dim=0)
            
            all_features.append(global_feature)
    
    return torch.stack(all_features)


def extract_dinov2_features(
    model,
    processor,
    images: List[Image.Image],
    device: torch.device,
    batch_size: int = 8,
    pooling: str = 'mean',
    use_tome: bool = False,
    tome_target: int = 64
) -> torch.Tensor:
    """
    Extract DINOv2 features with optional ToMe compression.
    
    DINOv2 output:
    - last_hidden_state: [batch, num_patches + 1, hidden_size]
      - First token is CLS token
      - Remaining tokens are patch tokens
    - For dinov2-base: hidden_size=768, patch_size=14
    - For 224x224 image: (224/14)^2 = 256 patches + 1 CLS = 257 tokens
    """
    all_features = []
    
    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]
        
        # Process images
        inputs = processor(images=batch_images, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state: [batch, num_tokens, hidden_size]
            hidden_states = outputs.last_hidden_state
        
        for i in range(len(batch_images)):
            frame_hidden = hidden_states[i]  # [num_tokens, hidden_size]
            cls_token = frame_hidden[0]       # [hidden_size] - CLS token
            patch_tokens = frame_hidden[1:]   # [num_patches, hidden_size]
            
            # Apply ToMe compression if requested
            if use_tome:
                patch_tokens = tome_compress(patch_tokens, tome_target)
            
            # Apply pooling strategy
            if pooling == 'cls':
                global_feature = cls_token
            elif pooling == 'mean':
                global_feature = patch_tokens.mean(dim=0)
            elif pooling == 'max':
                global_feature = patch_tokens.max(dim=0)[0]
            else:
                global_feature = patch_tokens.mean(dim=0)
            
            all_features.append(global_feature)
    
    return torch.stack(all_features)


def analyze_feature_discrimination(features: torch.Tensor) -> Dict[str, float]:
    """
    Analyze how discriminative the ViT features are between adjacent frames.
    
    Returns statistics about cosine similarity distribution.
    """
    features_norm = F.normalize(features, dim=-1)
    
    # Compute cosine similarity between all adjacent frames
    similarities = []
    for t in range(1, len(features)):
        sim = torch.dot(features_norm[t], features_norm[t - 1]).item()
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        'avg_cosine_sim': float(np.mean(similarities)),
        'min_cosine_sim': float(np.min(similarities)),
        'max_cosine_sim': float(np.max(similarities)),
        'std_cosine_sim': float(np.std(similarities)),
        'surprise_range': float(1 - np.min(similarities)) - float(1 - np.max(similarities)),
    }


# ============================================================
# Metrics
# ============================================================

def compute_fre(
    all_features: torch.Tensor,
    selected_indices: List[int]
) -> Tuple[float, float]:
    """
    Compute Feature Reconstruction Error (FRE).
    
    Returns:
        fre_avg: Average min distance to selected frames (lower = better)
        fre_max: Maximum min distance (worst case, lower = better)
    """
    T = all_features.shape[0]
    selected_features = all_features[selected_indices]
    
    min_distances = []
    
    for t in range(T):
        frame_feature = all_features[t]
        distances = ((selected_features - frame_feature) ** 2).sum(dim=-1)
        min_distance = distances.min().item()
        min_distances.append(min_distance)
    
    fre_avg = np.mean(min_distances)
    fre_max = np.max(min_distances)
    
    return fre_avg, fre_max


def compute_diversity_score(
    all_features: torch.Tensor,
    selected_indices: List[int]
) -> float:
    """
    Compute Semantic Diversity Score of selected frames.
    
    Higher diversity = selected frames are more different from each other.
    
    Formula:
        Score = (2 / K(K-1)) * Σ_{i<j} (1 - CosineSim(f_i, f_j))
    
    This measures average pairwise dissimilarity.
    - Uniform sampling in corridors: low diversity (similar frames)
    - Surprise sampling: high diversity (diverse frames)
    """
    K = len(selected_indices)
    if K < 2:
        return 0.0
    
    selected_features = all_features[selected_indices]
    selected_features_norm = F.normalize(selected_features, dim=-1)
    
    # Compute pairwise cosine similarity
    sim_matrix = torch.mm(selected_features_norm, selected_features_norm.T)
    
    # Sum of upper triangular (excluding diagonal)
    total_dissimilarity = 0.0
    pair_count = 0
    
    for i in range(K):
        for j in range(i + 1, K):
            dissimilarity = 1.0 - sim_matrix[i, j].item()
            total_dissimilarity += dissimilarity
            pair_count += 1
    
    diversity_score = total_dissimilarity / pair_count if pair_count > 0 else 0.0
    
    return diversity_score


# ============================================================
# Data Loading
# ============================================================

def parse_frame_filename(filename: str) -> Tuple[int, int]:
    """Parse frame filename to get index and action."""
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    frame_idx = int(parts[0])
    action = int(parts[1])
    return frame_idx, action


def load_sequence(sequence_path: str) -> List[FrameInfo]:
    """Load all frames from a sequence directory."""
    frames = []
    
    for filename in sorted(os.listdir(sequence_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            frame_idx, action = parse_frame_filename(filename)
            frames.append(FrameInfo(
                index=frame_idx,
                filename=filename,
                action=action,
                path=os.path.join(sequence_path, filename)
            ))
    
    frames.sort(key=lambda f: f.index)
    return frames


def get_test_sequences(data_dir: str) -> Dict[str, List[str]]:
    """Get all test sequences organized by type."""
    sequences = {
        'simple': [],
        'turns': [],
        'spin': []
    }
    
    for seq_type in sequences.keys():
        type_dir = os.path.join(data_dir, seq_type)
        if os.path.exists(type_dir):
            for seq_name in sorted(os.listdir(type_dir)):
                seq_path = os.path.join(type_dir, seq_name)
                if os.path.isdir(seq_path):
                    sequences[seq_type].append(seq_path)
    
    return sequences


# ============================================================
# Visualization
# ============================================================

def visualize_sequence_result(
    result: SequenceResult,
    frames: List[FrameInfo],
    output_dir: str
):
    """Generate visualization for a single sequence result."""
    if not HAS_MATPLOTLIB:
        return
    
    methods = list(result.method_results.keys())
    num_methods = len(methods)
    
    fig, axes = plt.subplots(3 + num_methods, 1, figsize=(14, 4 + num_methods * 2))
    
    # Plot 1: Surprise scores with action annotations
    ax1 = axes[0]
    x = list(range(len(result.surprise_scores)))
    
    # Replace inf with visible value for plotting
    plot_scores = []
    max_finite = max([s for s in result.surprise_scores if s != float('inf')], default=0.1)
    for s in result.surprise_scores:
        if s == float('inf'):
            plot_scores.append(max_finite * 1.5)
        else:
            plot_scores.append(s)
    
    ax1.plot(x, plot_scores, 'b-', linewidth=1.5, label='Surprise Score')
    ax1.fill_between(x, plot_scores, alpha=0.3)
    
    # Mark turn points (action 2 or 3)
    turn_indices = [f.index for f in frames if f.action in [2, 3]]
    for idx in turn_indices:
        if idx < len(result.surprise_scores):
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Surprise Score')
    ax1.set_title(f'{result.sequence_name} - Surprise Scores (Red = turn actions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add feature discrimination info
    ax1.text(0.02, 0.98, f'Avg Sim: {result.avg_cosine_sim:.4f}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Action sequence
    ax2 = axes[1]
    actions = [f.action for f in frames]
    action_colors = {0: 'gray', 1: 'green', 2: 'orange', 3: 'red'}
    action_labels = {0: 'Stop', 1: 'Forward', 2: 'Left', 3: 'Right'}
    
    for i, action in enumerate(actions):
        ax2.bar(i, 1, color=action_colors.get(action, 'black'), alpha=0.7)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=action_colors[a], label=action_labels[a]) for a in [1, 2, 3, 0]]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=4)
    ax2.set_xlabel('Frame Index')
    ax2.set_title('Action Sequence')
    ax2.set_yticks([])
    
    # Plot 3: Method comparison - FRE bar chart
    ax3 = axes[2]
    method_names = list(result.method_results.keys())
    fre_values = [result.method_results[m].fre for m in method_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))
    
    bars = ax3.bar(method_names, fre_values, color=colors)
    ax3.set_ylabel('FRE (Lower is Better)')
    ax3.set_title('Feature Reconstruction Error by Method')
    ax3.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for bar, val in zip(bars, fre_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Best method highlight
    best_method = min(method_names, key=lambda m: result.method_results[m].fre)
    best_idx = method_names.index(best_method)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # Plot 4+: Frame selection for each method
    for i, method in enumerate(methods):
        ax = axes[3 + i]
        selection = result.method_results[method].selected_indices
        
        # Create selection bar
        selection_bar = np.zeros(result.num_frames)
        for idx in selection:
            selection_bar[idx] = 1
        
        ax.bar(range(result.num_frames), selection_bar, alpha=0.7, color=colors[i], label=method)
        
        # Mark turn points
        for idx in turn_indices:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_ylabel('Selected')
        ax.set_title(f'{method}: {selection}')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        
        # Coverage indicator
        if len(selection) > 1:
            coverage = (max(selection) - min(selection)) / (result.num_frames - 1) * 100
            ax.text(0.98, 0.98, f'Coverage: {coverage:.0f}%', 
                    transform=ax.transAxes, fontsize=9, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if coverage > 80 else 'lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{result.sequence_name}_analysis.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  [Viz] Saved: {output_path}")


def visualize_summary(report: TestReport, output_dir: str):
    """Generate summary visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    methods = ['uniform', 'pure_topk', 'segmented_topk', 'gap_constrained', 'hybrid', 'diversity_topk']
    seq_types = list(report.metrics_by_type.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: FRE comparison by method (grouped by sequence type)
    ax1 = axes[0, 0]
    x = np.arange(len(seq_types))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        fre_values = []
        for seq_type in seq_types:
            key = f'avg_fre_{method}'
            if key in report.metrics_by_type.get(seq_type, {}):
                fre_values.append(report.metrics_by_type[seq_type][key])
            else:
                fre_values.append(0)
        
        offset = (i - len(methods)/2 + 0.5) * width
        ax1.bar(x + offset, fre_values, width, label=method, color=colors[i])
    
    ax1.set_xlabel('Sequence Type')
    ax1.set_ylabel('Average FRE (Lower is Better)')
    ax1.set_title('FRE by Method and Sequence Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_types)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: FRE improvement vs uniform (percentage)
    ax2 = axes[0, 1]
    
    for i, method in enumerate(methods[1:], 1):  # Skip uniform
        improvements = []
        for seq_type in seq_types:
            uniform_key = 'avg_fre_uniform'
            method_key = f'avg_fre_{method}'
            if uniform_key in report.metrics_by_type.get(seq_type, {}) and \
               method_key in report.metrics_by_type.get(seq_type, {}):
                uniform_fre = report.metrics_by_type[seq_type][uniform_key]
                method_fre = report.metrics_by_type[seq_type][method_key]
                improvement = (uniform_fre - method_fre) / uniform_fre * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        offset = (i - 1 - (len(methods)-1)/2 + 0.5) * width
        ax2.bar(x + offset, improvements, width, label=method, color=colors[i])
    
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% target')
    ax2.set_xlabel('Sequence Type')
    ax2.set_ylabel('FRE Improvement vs Uniform (%)')
    ax2.set_title('FRE Improvement (Higher is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_types)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Feature discrimination analysis
    ax3 = axes[1, 0]
    
    discrimination_metrics = ['avg_cosine_sim', 'min_cosine_sim', 'max_cosine_sim']
    for i, metric in enumerate(discrimination_metrics):
        values = []
        for seq_type in seq_types:
            key = f'avg_{metric}'
            if key in report.metrics_by_type.get(seq_type, {}):
                values.append(report.metrics_by_type[seq_type][key])
            else:
                values.append(0)
        
        offset = (i - len(discrimination_metrics)/2 + 0.5) * 0.25
        ax3.bar(x + offset, values, 0.25, label=metric.replace('_', ' ').title())
    
    ax3.set_xlabel('Sequence Type')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Feature Discrimination: Adjacent Frame Similarity\n(Lower = More Discriminative)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(seq_types)
    ax3.legend()
    ax3.set_ylim(0.9, 1.0)  # Focus on high similarity range
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Best method summary
    ax4 = axes[1, 1]
    
    best_methods = {}
    for seq_type in seq_types:
        best_fre = float('inf')
        best_method = 'uniform'
        for method in methods:
            key = f'avg_fre_{method}'
            if key in report.metrics_by_type.get(seq_type, {}):
                fre = report.metrics_by_type[seq_type][key]
                if fre < best_fre:
                    best_fre = fre
                    best_method = method
        best_methods[seq_type] = best_method
    
    # Create text summary
    summary_text = "Best Method by Sequence Type:\n\n"
    for seq_type, method in best_methods.items():
        uniform_fre = report.metrics_by_type[seq_type].get('avg_fre_uniform', 0)
        best_fre = report.metrics_by_type[seq_type].get(f'avg_fre_{method}', 0)
        improvement = (uniform_fre - best_fre) / uniform_fre * 100 if uniform_fre > 0 else 0
        summary_text += f"• {seq_type}: {method}\n"
        summary_text += f"  FRE: {best_fre:.2f} ({improvement:+.1f}% vs uniform)\n\n"
    
    # Feature discrimination summary
    summary_text += "\nFeature Discrimination:\n"
    overall_sim = report.feature_discrimination.get('overall_avg_sim', 0)
    summary_text += f"• Overall Avg Similarity: {overall_sim:.4f}\n"
    summary_text += f"• Surprise Range: {1-overall_sim:.4f}\n"
    
    if overall_sim > 0.98:
        summary_text += "\n⚠️ Warning: Very high similarity!\n"
        summary_text += "ViT features may lack discrimination."
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Summary')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'summary_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[Viz] Summary saved: {output_path}")


def visualize_feature_discrimination(report: TestReport, output_dir: str):
    """Generate feature discrimination analysis visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    # Collect all similarity data from results
    all_sims_by_type = {t: [] for t in ['simple', 'turns', 'spin']}
    
    for result in report.results:
        # Compute similarities from surprise scores
        sims = [1 - s for s in result.surprise_scores[1:] if s != float('inf')]
        all_sims_by_type[result.sequence_type].extend(sims)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (seq_type, sims) in enumerate(all_sims_by_type.items()):
        ax = axes[i]
        if sims:
            ax.hist(sims, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(sims), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sims):.4f}')
            ax.axvline(x=np.median(sims), color='green', linestyle='--',
                       label=f'Median: {np.median(sims):.4f}')
        
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_title(f'{seq_type.upper()} - Adjacent Frame Similarity Distribution')
        ax.legend()
        ax.set_xlim(0.9, 1.0)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_discrimination.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[Viz] Feature discrimination saved: {output_path}")


# ============================================================
# Main Test Logic
# ============================================================

def test_sequence(
    sequence_path: str,
    model,
    processor,
    device: torch.device,
    budget: int,
    pooling: str = 'mean',
    model_type: str = 'qwen',
    use_tome: bool = False,
    tome_target: int = 64,
    use_centering: bool = False
) -> Tuple[SequenceResult, List[FrameInfo]]:
    """Test all sampling strategies on a single sequence."""
    sequence_name = os.path.basename(sequence_path)
    parent_dir = os.path.basename(os.path.dirname(sequence_path))
    
    # Load frames
    frames = load_sequence(sequence_path)
    num_frames = len(frames)
    
    print(f"  Sequence: {sequence_name} ({num_frames} frames)")
    
    if num_frames == 0:
        print(f"    [Warning] No frames found, skipping")
        return None, []
    
    # Load images
    images = []
    for frame in frames:
        try:
            img = Image.open(frame.path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"    [Warning] Failed to load {frame.filename}: {e}")
            images.append(Image.new('RGB', (640, 480), color='black'))
    
    # Extract features with optional ToMe and centering
    features = extract_features(
        model, processor, images, device, 
        pooling=pooling, model_type=model_type,
        use_tome=use_tome, tome_target=tome_target,
        use_centering=use_centering
    )
    
    feature_config = f"pooling={pooling}, model={model_type}"
    if use_tome:
        feature_config += f", ToMe={tome_target}"
    if use_centering:
        feature_config += ", centered"
    print(f"    Features extracted: {features.shape}, {feature_config}")
    
    # Analyze feature discrimination
    discrimination = analyze_feature_discrimination(features)
    print(f"    Feature discrimination: avg_sim={discrimination['avg_cosine_sim']:.4f}, "
          f"surprise_range={discrimination['surprise_range']:.4f}")
    
    # Compute surprise scores
    surprise_scores = compute_surprise_scores(features)
    
    # Apply all sampling strategies
    sampling_methods = {
        'uniform': lambda: uniform_sampling(num_frames, budget),
        'pure_topk': lambda: pure_topk_sampling(surprise_scores, budget),
        'segmented_topk': lambda: segmented_topk_sampling(surprise_scores, budget),
        'gap_constrained': lambda: gap_constrained_topk_sampling(surprise_scores, budget, min_gap=max(2, num_frames // budget // 2)),
        'hybrid': lambda: hybrid_sampling(surprise_scores, budget),
        'diversity_topk': lambda: diversity_constrained_topk_sampling(features, surprise_scores, budget, diversity_weight=0.3),
    }
    
    result = SequenceResult(
        sequence_name=sequence_name,
        sequence_type=parent_dir,
        num_frames=num_frames,
        budget=budget,
        surprise_scores=surprise_scores,
        avg_cosine_sim=discrimination['avg_cosine_sim'],
        min_cosine_sim=discrimination['min_cosine_sim'],
        max_cosine_sim=discrimination['max_cosine_sim'],
        std_cosine_sim=discrimination['std_cosine_sim'],
    )
    
    print(f"    {'Method':<18} {'FRE':>8} {'FRE_Max':>8} {'Diversity':>10} {'Coverage':>8}")
    print(f"    {'-'*56}")
    
    for method_name, sampler in sampling_methods.items():
        indices = sampler()
        fre_avg, fre_max = compute_fre(features, indices)
        diversity = compute_diversity_score(features, indices)
        
        result.method_results[method_name] = MethodResult(
            method_name=method_name,
            selected_indices=indices,
            fre=fre_avg,
            fre_max=fre_max,
            diversity_score=diversity
        )
        
        # Calculate coverage
        coverage = (max(indices) - min(indices)) / (num_frames - 1) * 100 if len(indices) > 1 else 0
        print(f"    {method_name:<18} {fre_avg:>8.2f} {fre_max:>8.2f} {diversity:>10.4f} {coverage:>7.1f}%")
    
    return result, frames


def main():
    parser = argparse.ArgumentParser(description='Cosine Surprise-based Keyframe Selection Test')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Path to model (Qwen or DINOv2)')
    parser.add_argument('--model_type', type=str, default='qwen', choices=['qwen', 'dinov2'],
                        help='Model type: qwen for Qwen2.5-VL, dinov2 for DINOv2')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'test_data', 'habitat'),
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: results/<timestamp>)')
    parser.add_argument('--budget', type=int, default=4,
                        help='Number of frames to select (K)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for feature extraction')
    parser.add_argument('--pooling', type=str, default='max', choices=['mean', 'max', 'cls'],
                        help='Feature pooling method (mean, max, cls)')
    parser.add_argument('--use_tome', action='store_true',
                        help='Use ToMe compression before pooling')
    parser.add_argument('--tome_target', type=int, default=64,
                        help='Target number of tokens after ToMe compression')
    parser.add_argument('--use_centering', action='store_true',
                        help='Apply feature centering (subtract global mean)')
    args = parser.parse_args()
    
    # Set default model path based on model type
    if args.model_type == 'dinov2' and args.model_path == 'Qwen/Qwen2.5-VL-3B-Instruct':
        args.model_path = 'facebook/dinov2-base'
    
    print("=" * 70)
    print("Cosine Surprise-based Keyframe Selection Test (v3 - Enhanced)")
    print("=" * 70)
    
    # Setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(__file__), 'results', timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Config] Device: {device}")
    print(f"[Config] Model: {args.model_path}")
    print(f"[Config] Model Type: {args.model_type}")
    print(f"[Config] Budget (K): {args.budget}")
    print(f"[Config] Pooling: {args.pooling}")
    print(f"[Config] ToMe: {args.use_tome} (target={args.tome_target})")
    print(f"[Config] Centering: {args.use_centering}")
    print(f"[Config] Data dir: {args.data_dir}")
    print(f"[Config] Output dir: {output_dir}")

    if not os.path.isdir(args.data_dir):
        print(f"[Error] Test data directory not found: {args.data_dir}")
        print("[Hint] Provide --data_dir pointing to external feature_test fixtures.")
        return
    
    # Load model
    print("\n[Step 1] Loading model...")
    model, processor = load_model(args.model_path, device, model_type=args.model_type)
    
    # Get test sequences
    print("\n[Step 2] Finding test sequences...")
    sequences = get_test_sequences(args.data_dir)
    
    for seq_type, seq_list in sequences.items():
        print(f"  {seq_type}: {len(seq_list)} sequences")
    
    # Run tests
    print("\n[Step 3] Running tests...")
    report = TestReport(
        model_name=args.model_path,
        budget=args.budget,
        timestamp=timestamp
    )
    
    all_similarities = []
    
    for seq_type, seq_list in sequences.items():
        print(f"\n--- Testing {seq_type.upper()} sequences ---")
        
        for seq_path in seq_list:
            result, frames = test_sequence(
                seq_path, model, processor, device, args.budget, 
                args.pooling, args.model_type,
                args.use_tome, args.tome_target, args.use_centering
            )
            
            if result is not None:
                report.results.append(result)
                all_similarities.append(result.avg_cosine_sim)
                
                # Generate per-sequence visualization
                visualize_sequence_result(result, frames, output_dir)
    
    # Aggregate metrics
    print("\n[Step 4] Aggregating metrics...")
    methods = ['uniform', 'pure_topk', 'segmented_topk', 'gap_constrained', 'hybrid', 'diversity_topk']
    
    for seq_type in sequences.keys():
        type_results = [r for r in report.results if r.sequence_type == seq_type]
        
        if type_results:
            metrics = {
                'num_sequences': len(type_results),
            }
            
            # Metrics for each method
            for method in methods:
                method_results = [r.method_results[method] for r in type_results if method in r.method_results]
                if method_results:
                    metrics[f'avg_fre_{method}'] = np.mean([mr.fre for mr in method_results])
                    metrics[f'std_fre_{method}'] = np.std([mr.fre for mr in method_results])
                    metrics[f'avg_fre_max_{method}'] = np.mean([mr.fre_max for mr in method_results])
                    metrics[f'avg_diversity_{method}'] = np.mean([mr.diversity_score for mr in method_results])
            
            # Feature discrimination
            metrics['avg_avg_cosine_sim'] = np.mean([r.avg_cosine_sim for r in type_results])
            metrics['avg_min_cosine_sim'] = np.mean([r.min_cosine_sim for r in type_results])
            metrics['avg_max_cosine_sim'] = np.mean([r.max_cosine_sim for r in type_results])
            
            report.metrics_by_type[seq_type] = metrics
    
    # Overall feature discrimination
    report.feature_discrimination = {
        'overall_avg_sim': np.mean(all_similarities) if all_similarities else 0,
        'overall_std_sim': np.std(all_similarities) if all_similarities else 0,
    }
    
    # Generate summary visualization
    print("\n[Step 5] Generating visualizations...")
    visualize_summary(report, output_dir)
    visualize_feature_discrimination(report, output_dir)
    
    # Save results
    print("\n[Step 6] Saving results...")
    
    # Helper to convert numpy types to Python native types
    def to_python_type(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(v) for v in obj]
        return obj
    
    # Convert results to JSON-serializable format
    results_dict = {
        'model_name': report.model_name,
        'model_type': args.model_type,
        'budget': report.budget,
        'timestamp': report.timestamp,
        'pooling': args.pooling,
        'use_tome': args.use_tome,
        'tome_target': args.tome_target,
        'use_centering': args.use_centering,
        'feature_discrimination': to_python_type(report.feature_discrimination),
        'metrics_by_type': to_python_type(report.metrics_by_type),
        'detailed_results': []
    }
    
    for r in report.results:
        result_dict = {
            'sequence_name': r.sequence_name,
            'sequence_type': r.sequence_type,
            'num_frames': r.num_frames,
            'budget': r.budget,
            'surprise_scores': [float(s) if s != float('inf') else 'inf' for s in r.surprise_scores],
            'avg_cosine_sim': float(r.avg_cosine_sim),
            'min_cosine_sim': float(r.min_cosine_sim),
            'max_cosine_sim': float(r.max_cosine_sim),
            'method_results': {
                name: {
                    'selected_indices': [int(i) for i in mr.selected_indices],
                    'fre': float(mr.fre),
                    'fre_max': float(mr.fre_max),
                    'diversity_score': float(mr.diversity_score)
                }
                for name, mr in r.method_results.items()
            }
        }
        results_dict['detailed_results'].append(result_dict)
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"[Results] Saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nModel: {report.model_name} ({args.model_type})")
    print(f"Budget (K): {report.budget}")
    print(f"Pooling: {args.pooling}")
    print(f"ToMe: {args.use_tome} (target={args.tome_target})")
    print(f"Centering: {args.use_centering}")
    print(f"Output: {output_dir}")
    
    # Feature discrimination warning
    overall_sim = report.feature_discrimination.get('overall_avg_sim', 0)
    print(f"\n[Feature Discrimination]")
    print(f"  Overall Average Cosine Similarity: {overall_sim:.4f}")
    print(f"  Average Surprise Range: {1 - overall_sim:.4f}")
    
    if overall_sim > 0.98:
        print(f"  ⚠️  WARNING: Very high similarity ({overall_sim:.4f})!")
        print(f"      ViT features may lack sufficient discrimination between frames.")
        print(f"      Consider: (1) Different pooling method  (2) Different feature extractor")
    
    print(f"\n[Metric 1: FRE - Feature Reconstruction Error] (Lower = Better)")
    print("  定义: 所有帧到最近选中帧的平均距离，衡量整体覆盖程度")
    print("-" * 80)
    header = f"{'Type':<10}"
    for method in methods:
        header += f" {method:>13}"
    print(header)
    print("-" * 80)
    
    for seq_type, metrics in report.metrics_by_type.items():
        row = f"{seq_type:<10}"
        for method in methods:
            key = f'avg_fre_{method}'
            if key in metrics:
                row += f" {metrics[key]:>13.2f}"
            else:
                row += f" {'N/A':>13}"
        print(row)
    
    print("-" * 80)
    
    print(f"\n[Metric 2: FRE Max - Maximum Reconstruction Error] (Lower = Better)")
    print("  定义: 最坏情况下的重建误差，衡量是否有帧被严重遗漏")
    print("-" * 80)
    header = f"{'Type':<10}"
    for method in methods:
        header += f" {method:>13}"
    print(header)
    print("-" * 80)
    
    for seq_type, metrics in report.metrics_by_type.items():
        row = f"{seq_type:<10}"
        for method in methods:
            key = f'avg_fre_max_{method}'
            if key in metrics:
                row += f" {metrics[key]:>13.2f}"
            else:
                row += f" {'N/A':>13}"
        print(row)
    
    print("-" * 80)
    
    print(f"\n[Metric 3: Diversity Score - Semantic Diversity] (Higher = Better)")
    print("  定义: 选中帧之间的平均成对相异度，衡量是否存在信息冗余")
    print("  公式: (2/K(K-1)) * Σ(1 - CosineSim(f_i, f_j)) for i<j")
    print("-" * 80)
    header = f"{'Type':<10}"
    for method in methods:
        header += f" {method:>13}"
    print(header)
    print("-" * 80)
    
    for seq_type, metrics in report.metrics_by_type.items():
        row = f"{seq_type:<10}"
        for method in methods:
            key = f'avg_diversity_{method}'
            if key in metrics:
                row += f" {metrics[key]:>13.4f}"
            else:
                row += f" {'N/A':>13}"
        print(row)
    
    print("-" * 80)
    
    # Find best method for each type (considering all metrics)
    print(f"\n[Best Method Analysis by Sequence Type]")
    print("-" * 80)
    
    for seq_type, metrics in report.metrics_by_type.items():
        print(f"\n  {seq_type.upper()}:")
        
        uniform_fre = metrics.get('avg_fre_uniform', float('inf'))
        uniform_fre_max = metrics.get('avg_fre_max_uniform', float('inf'))
        uniform_div = metrics.get('avg_diversity_uniform', 0)
        
        # Best FRE
        best_fre = float('inf')
        best_fre_method = 'uniform'
        for method in methods:
            key = f'avg_fre_{method}'
            if key in metrics and metrics[key] < best_fre:
                best_fre = metrics[key]
                best_fre_method = method
        fre_imp = (uniform_fre - best_fre) / uniform_fre * 100 if uniform_fre > 0 else 0
        
        # Best FRE Max
        best_fre_max = float('inf')
        best_fre_max_method = 'uniform'
        for method in methods:
            key = f'avg_fre_max_{method}'
            if key in metrics and metrics[key] < best_fre_max:
                best_fre_max = metrics[key]
                best_fre_max_method = method
        fre_max_imp = (uniform_fre_max - best_fre_max) / uniform_fre_max * 100 if uniform_fre_max > 0 else 0
        
        # Best Diversity
        best_div = 0
        best_div_method = 'uniform'
        for method in methods:
            key = f'avg_diversity_{method}'
            if key in metrics and metrics[key] > best_div:
                best_div = metrics[key]
                best_div_method = method
        div_imp = (best_div - uniform_div) / uniform_div * 100 if uniform_div > 0 else 0
        
        print(f"    Best FRE:       {best_fre_method:<18} ({best_fre:.2f}, {fre_imp:+.1f}% vs uniform)")
        print(f"    Best FRE Max:   {best_fre_max_method:<18} ({best_fre_max:.2f}, {fre_max_imp:+.1f}% vs uniform)")
        print(f"    Best Diversity: {best_div_method:<18} ({best_div:.4f}, {div_imp:+.1f}% vs uniform)")
    
    print("-" * 80)
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
