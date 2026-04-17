#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Feature Analysis Tests

Test 1: Random negative sample cosine distribution
- Compute cosine similarity between frames from different episodes
- If random unrelated frames have 0.7-0.9 cosine, the embedding space is anisotropic

Test 2: Layer/Pooling sensitivity
- Test different VIT layers (shallow=geometric, deep=semantic)
- Test different pooling methods (mean, first token, attention-weighted)

Usage:
    python test_feature_analysis.py \
        --checkpoint /path/to/checkpoint \
        --data_path /path/to/data \
        --output_dir ./analysis_results
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths - ensure we can import swiftvln.model
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mem_test_dir = _current_dir
_feature_test_dir = os.path.dirname(_mem_test_dir)
_vln_dir = os.path.dirname(_feature_test_dir)
_examples_dir = os.path.dirname(_vln_dir)
_msswift_root = os.path.dirname(_examples_dir)

for path in [_msswift_root, _vln_dir, _mem_test_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


class FeatureAnalyzer:
    """Analyze VIT feature properties."""
    
    def __init__(
        self,
        checkpoint_path: str,
        data_path: str,
        output_dir: str,
        device: str = 'cuda',
        num_episodes: int = 20,  # Number of episodes to sample
        frames_per_episode: int = 5,  # Frames per episode
    ):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device
        self.num_episodes = num_episodes
        self.frames_per_episode = frames_per_episode
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.processor = None
        self.nav_data = None
    
    def setup(self):
        """Load model and data."""
        print("=" * 60)
        print("Feature Analysis Setup")
        print("=" * 60)
        
        self._load_model()
        self._load_data()
        print("Setup complete!\n")
    
    def _load_model(self):
        """Load model."""
        print(f"\nLoading model from: {self.checkpoint_path}")
        
        # Register custom models
        try:
            import swiftvln.model
        except ImportError:
            pass
        
        from swift.llm import get_model_tokenizer
        
        self.model, self.processor = get_model_tokenizer(
            model_id_or_path=self.checkpoint_path,
            model_type='overlapvln_qwen2_5_vl',
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_impl='flash_attn',
        )
        self.model.eval()
        print("  Model loaded successfully")
    
    def _load_data(self):
        """Load navigation data."""
        print(f"\nLoading data from: {self.data_path}")
        
        anno_path = os.path.join(self.data_path, 'annotations.json')
        with open(anno_path, 'r') as f:
            self.nav_data = json.load(f)
        
        # Add video path prefix
        for item in self.nav_data:
            item['video'] = os.path.join(self.data_path, item['video'])
        
        print(f"  Loaded {len(self.nav_data)} episodes")
    
    @torch.no_grad()
    def encode_frames(
        self,
        images: List[Image.Image],
        return_all_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Encode frames through VIT.
        
        Returns:
            features: Final layer features [num_images, tokens, hidden]
            grid_thw: Grid dimensions
            all_layer_features: If return_all_layers, list of features from each layer
        """
        if not images:
            return None, None, None
        
        # Process images
        media_inputs = self.processor.image_processor(
            images=images, return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(self.device).type(torch.bfloat16)
        image_grid_thw = media_inputs['image_grid_thw'].to(self.device)
        
        # Get VIT encoder - model.visual is the VIT encoder itself
        vit = self.model.visual
        
        # Extract features with optional intermediate layers
        if return_all_layers:
            # Forward through VIT with output_hidden_states
            vit_outputs = vit(
                pixel_values, 
                grid_thw=image_grid_thw,
                output_hidden_states=True
            )
            if hasattr(vit_outputs, 'hidden_states'):
                all_layer_features = [h.float() for h in vit_outputs.hidden_states]
            else:
                all_layer_features = [vit_outputs.float()]
            features = all_layer_features[-1]
        else:
            features = vit(pixel_values, grid_thw=image_grid_thw).float()
            all_layer_features = None
        
        return features, image_grid_thw, all_layer_features
    
    def _sample_frames_from_episodes(self) -> Dict[int, List[Image.Image]]:
        """Sample frames from multiple episodes."""
        print(f"\nSampling {self.frames_per_episode} frames from {self.num_episodes} episodes...")
        
        # Randomly select episodes
        episode_indices = random.sample(
            range(len(self.nav_data)), 
            min(self.num_episodes, len(self.nav_data))
        )
        
        episode_frames = {}
        
        for ep_idx in episode_indices:
            item = self.nav_data[ep_idx]
            video_path = item['video']
            rgb_path = os.path.join(video_path, 'rgb')
            
            if not os.path.exists(rgb_path):
                continue
            
            frame_files = sorted(os.listdir(rgb_path))
            if len(frame_files) < self.frames_per_episode:
                continue
            
            # Uniformly sample frames
            indices = np.linspace(0, len(frame_files) - 1, self.frames_per_episode, dtype=int)
            
            frames = []
            for idx in indices:
                frame_path = os.path.join(rgb_path, frame_files[idx])
                try:
                    img = Image.open(frame_path).convert('RGB')
                    frames.append(img)
                except Exception as e:
                    print(f"  Warning: Failed to load {frame_path}")
                    continue
            
            if len(frames) == self.frames_per_episode:
                episode_frames[ep_idx] = frames
        
        print(f"  Successfully sampled from {len(episode_frames)} episodes")
        return episode_frames
    
    def test1_negative_sample_distribution(self) -> Dict:
        """
        Test 1: Random negative sample cosine distribution.
        
        Compute cosine similarity between frames from DIFFERENT episodes.
        If random unrelated frames have 0.7-0.9 cosine, the embedding space is anisotropic.
        """
        print("\n" + "=" * 60)
        print("Test 1: Random Negative Sample Cosine Distribution")
        print("=" * 60)
        
        episode_frames = self._sample_frames_from_episodes()
        
        if len(episode_frames) < 2:
            print("Not enough episodes to compute negative samples")
            return {}
        
        # Encode all frames
        print("\nEncoding frames...")
        episode_features = {}
        
        for ep_idx, frames in episode_frames.items():
            features, grid_thw, _ = self.encode_frames(frames)
            if features is not None:
                # Mean pool each frame
                merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
                merge_length = merge_size ** 2
                
                frame_features = []
                embed_idx = 0
                for i in range(len(frames)):
                    num_tokens = int(grid_thw[i].prod() // merge_length)
                    frame_feat = features[embed_idx:embed_idx + num_tokens]
                    # Mean pool to get frame representation
                    frame_repr = frame_feat.mean(dim=0)
                    frame_features.append(frame_repr)
                    embed_idx += num_tokens
                
                episode_features[ep_idx] = torch.stack(frame_features)
        
        print(f"  Encoded {len(episode_features)} episodes")
        
        # Compute inter-episode cosine similarities (negative samples)
        print("\nComputing inter-episode cosine similarities...")
        inter_episode_sims = []
        intra_episode_sims = []
        
        ep_ids = list(episode_features.keys())
        
        for i, ep1 in enumerate(ep_ids):
            feats1 = episode_features[ep1]
            feats1_norm = F.normalize(feats1, p=2, dim=-1)
            
            # Intra-episode similarities
            intra_sim = torch.mm(feats1_norm, feats1_norm.t())
            # Exclude diagonal
            mask = ~torch.eye(len(feats1), dtype=torch.bool, device=intra_sim.device)
            intra_episode_sims.extend(intra_sim[mask].cpu().tolist())
            
            # Inter-episode similarities
            for j, ep2 in enumerate(ep_ids):
                if i >= j:  # Only upper triangle
                    continue
                
                feats2 = episode_features[ep2]
                feats2_norm = F.normalize(feats2, p=2, dim=-1)
                
                # All pairwise similarities between episodes
                inter_sim = torch.mm(feats1_norm, feats2_norm.t())
                inter_episode_sims.extend(inter_sim.cpu().flatten().tolist())
        
        inter_episode_sims = np.array(inter_episode_sims)
        intra_episode_sims = np.array(intra_episode_sims)
        
        # Statistics
        results = {
            'inter_episode': {
                'mean': float(np.mean(inter_episode_sims)),
                'std': float(np.std(inter_episode_sims)),
                'min': float(np.min(inter_episode_sims)),
                'max': float(np.max(inter_episode_sims)),
                'p25': float(np.percentile(inter_episode_sims, 25)),
                'p50': float(np.percentile(inter_episode_sims, 50)),
                'p75': float(np.percentile(inter_episode_sims, 75)),
                'p90': float(np.percentile(inter_episode_sims, 90)),
                'num_samples': len(inter_episode_sims),
            },
            'intra_episode': {
                'mean': float(np.mean(intra_episode_sims)),
                'std': float(np.std(intra_episode_sims)),
                'min': float(np.min(intra_episode_sims)),
                'max': float(np.max(intra_episode_sims)),
                'p25': float(np.percentile(intra_episode_sims, 25)),
                'p50': float(np.percentile(intra_episode_sims, 50)),
                'p75': float(np.percentile(intra_episode_sims, 75)),
                'p90': float(np.percentile(intra_episode_sims, 90)),
                'num_samples': len(intra_episode_sims),
            }
        }
        
        # Print results
        print("\n" + "-" * 40)
        print("Inter-Episode Similarity (Negative Samples):")
        print(f"  Mean: {results['inter_episode']['mean']:.4f}")
        print(f"  Std:  {results['inter_episode']['std']:.4f}")
        print(f"  Range: [{results['inter_episode']['min']:.4f}, {results['inter_episode']['max']:.4f}]")
        print(f"  P25/P50/P75/P90: {results['inter_episode']['p25']:.4f} / {results['inter_episode']['p50']:.4f} / {results['inter_episode']['p75']:.4f} / {results['inter_episode']['p90']:.4f}")
        
        print("\nIntra-Episode Similarity (Same Episode):")
        print(f"  Mean: {results['intra_episode']['mean']:.4f}")
        print(f"  Std:  {results['intra_episode']['std']:.4f}")
        print(f"  Range: [{results['intra_episode']['min']:.4f}, {results['intra_episode']['max']:.4f}]")
        print(f"  P25/P50/P75/P90: {results['intra_episode']['p25']:.4f} / {results['intra_episode']['p50']:.4f} / {results['intra_episode']['p75']:.4f} / {results['intra_episode']['p90']:.4f}")
        
        # Interpretation
        print("\n" + "-" * 40)
        print("Interpretation:")
        inter_mean = results['inter_episode']['mean']
        if inter_mean > 0.8:
            print(f"  ⚠️  HIGH anisotropy: inter-episode mean={inter_mean:.3f}")
            print("  → Embedding space is highly concentrated")
            print("  → Cosine similarity has limited dynamic range")
        elif inter_mean > 0.7:
            print(f"  ⚠️  MODERATE anisotropy: inter-episode mean={inter_mean:.3f}")
            print("  → Some concentration in embedding space")
            print("  → Consider using different metrics or features")
        else:
            print(f"  ✓  LOW anisotropy: inter-episode mean={inter_mean:.3f}")
            print("  → Embedding space is well-distributed")
            print("  → Cosine similarity should work well")
        
        # Plot distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(inter_episode_sims, bins=50, alpha=0.7, label='Inter-episode (negative)', color='blue')
        ax.hist(intra_episode_sims, bins=50, alpha=0.7, label='Intra-episode (same)', color='red')
        ax.axvline(results['inter_episode']['mean'], color='blue', linestyle='--', label=f"Inter mean={results['inter_episode']['mean']:.3f}")
        ax.axvline(results['intra_episode']['mean'], color='red', linestyle='--', label=f"Intra mean={results['intra_episode']['mean']:.3f}")
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Cosine Similarity Distribution')
        ax.legend()
        
        # Box plot
        ax = axes[1]
        ax.boxplot([inter_episode_sims, intra_episode_sims], labels=['Inter-episode\n(negative)', 'Intra-episode\n(same)'])
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'test1_cosine_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {plot_path}")
        
        return results
    
    def test2_pooling_sensitivity(self) -> Dict:
        """
        Test 2: Pooling sensitivity.
        
        Compare different pooling methods:
        - Mean pooling (current)
        - First token (if available)
        - Max pooling
        - Attention-weighted pooling
        """
        print("\n" + "=" * 60)
        print("Test 2: Pooling Method Sensitivity")
        print("=" * 60)
        
        episode_frames = self._sample_frames_from_episodes()
        
        if len(episode_frames) < 2:
            print("Not enough episodes")
            return {}
        
        # Take first few episodes for detailed analysis
        test_episodes = list(episode_frames.keys())[:5]
        
        print("\nEncoding frames with different pooling methods...")
        
        pooling_methods = ['mean', 'max', 'first', 'last']
        method_features = {method: [] for method in pooling_methods}
        
        for ep_idx in test_episodes:
            frames = episode_frames[ep_idx]
            features, grid_thw, _ = self.encode_frames(frames)
            
            if features is None:
                continue
            
            merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
            merge_length = merge_size ** 2
            
            embed_idx = 0
            for i in range(len(frames)):
                num_tokens = int(grid_thw[i].prod() // merge_length)
                frame_feat = features[embed_idx:embed_idx + num_tokens]  # [tokens, hidden]
                embed_idx += num_tokens
                
                # Different pooling methods
                method_features['mean'].append(frame_feat.mean(dim=0))
                method_features['max'].append(frame_feat.max(dim=0)[0])
                method_features['first'].append(frame_feat[0])
                method_features['last'].append(frame_feat[-1])
        
        # Stack features for each method
        for method in pooling_methods:
            method_features[method] = torch.stack(method_features[method])
        
        # Compute redundancy-like metrics for each pooling method
        print("\nComputing metrics for each pooling method...")
        
        results = {}
        
        for method in pooling_methods:
            feats = method_features[method]
            feats_norm = F.normalize(feats, p=2, dim=-1)
            
            # Pairwise similarity
            sim = torch.mm(feats_norm, feats_norm.t())
            
            # Exclude diagonal
            n = len(feats)
            mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
            pairwise_sims = sim[mask].cpu().numpy()
            
            # NN similarity (max excluding self)
            sim_masked = sim.masked_fill(~mask, float('-inf'))
            nn_sims = sim_masked.max(dim=1)[0].cpu().numpy()
            
            results[method] = {
                'pairwise_mean': float(np.mean(pairwise_sims)),
                'pairwise_std': float(np.std(pairwise_sims)),
                'pairwise_min': float(np.min(pairwise_sims)),
                'pairwise_max': float(np.max(pairwise_sims)),
                'nn_mean': float(np.mean(nn_sims)),
                'nn_std': float(np.std(nn_sims)),
            }
        
        # Print results
        print("\n" + "-" * 40)
        print("Pooling Method Comparison:")
        print("-" * 40)
        print(f"{'Method':<10} {'Pairwise Mean':<15} {'Pairwise Std':<15} {'NN Mean':<15}")
        print("-" * 55)
        
        for method in pooling_methods:
            r = results[method]
            print(f"{method:<10} {r['pairwise_mean']:<15.4f} {r['pairwise_std']:<15.4f} {r['nn_mean']:<15.4f}")
        
        # Sensitivity analysis
        print("\n" + "-" * 40)
        print("Sensitivity Analysis:")
        
        pairwise_means = [results[m]['pairwise_mean'] for m in pooling_methods]
        nn_means = [results[m]['nn_mean'] for m in pooling_methods]
        
        pairwise_range = max(pairwise_means) - min(pairwise_means)
        nn_range = max(nn_means) - min(nn_means)
        
        print(f"  Pairwise mean range: {pairwise_range:.4f}")
        print(f"  NN mean range: {nn_range:.4f}")
        
        if pairwise_range > 0.1:
            print("  ⚠️  HIGH sensitivity to pooling method")
            print("  → Feature selection significantly affects metrics")
        else:
            print("  ✓  LOW sensitivity to pooling method")
            print("  → Metrics are relatively stable across pooling methods")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(pooling_methods))
        width = 0.35
        
        # Pairwise similarity
        ax = axes[0]
        means = [results[m]['pairwise_mean'] for m in pooling_methods]
        stds = [results[m]['pairwise_std'] for m in pooling_methods]
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.8)
        ax.set_ylabel('Pairwise Cosine Similarity')
        ax.set_title('Pairwise Similarity by Pooling Method')
        ax.set_xticks(x)
        ax.set_xticklabels(pooling_methods)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        # NN similarity
        ax = axes[1]
        means = [results[m]['nn_mean'] for m in pooling_methods]
        stds = [results[m]['nn_std'] for m in pooling_methods]
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.8, color='orange')
        ax.set_ylabel('NN Cosine Similarity')
        ax.set_title('Nearest Neighbor Similarity by Pooling Method')
        ax.set_xticks(x)
        ax.set_xticklabels(pooling_methods)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'test2_pooling_sensitivity.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {plot_path}")
        
        return results
    
    def run_all_tests(self):
        """Run all analysis tests."""
        self.setup()
        
        results = {}
        
        # Test 1
        results['test1_negative_samples'] = self.test1_negative_sample_distribution()
        
        # Test 2
        results['test2_pooling_sensitivity'] = self.test2_pooling_sensitivity()
        
        # Save results
        results_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        # Summary
        print("\n" + "=" * 60)
        print("Analysis Summary")
        print("=" * 60)
        
        if 'test1_negative_samples' in results and results['test1_negative_samples']:
            inter_mean = results['test1_negative_samples']['inter_episode']['mean']
            intra_mean = results['test1_negative_samples']['intra_episode']['mean']
            print(f"\nTest 1 - Anisotropy Check:")
            print(f"  Inter-episode (negative) mean: {inter_mean:.4f}")
            print(f"  Intra-episode (same) mean: {intra_mean:.4f}")
            print(f"  Gap: {intra_mean - inter_mean:.4f}")
        
        if 'test2_pooling_sensitivity' in results and results['test2_pooling_sensitivity']:
            print(f"\nTest 2 - Pooling Sensitivity:")
            for method, r in results['test2_pooling_sensitivity'].items():
                print(f"  {method}: pairwise_mean={r['pairwise_mean']:.4f}, nn_mean={r['nn_mean']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Feature Analysis Tests')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to trajectory data')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes to sample')
    parser.add_argument('--frames_per_episode', type=int, default=5,
                       help='Frames per episode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    analyzer = FeatureAnalyzer(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        frames_per_episode=args.frames_per_episode,
    )
    
    analyzer.run_all_tests()


if __name__ == '__main__':
    main()
