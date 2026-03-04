# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Build Fixed Test Set for Memory Evaluation

This script creates a fixed test set from R2R/RxR trajectory data for
consistent evaluation across different memory strategies.

Test set categories:
- common: Random samples with history length 16-64
- long_trajectory: Samples with history length > 64 (multiple window slides)
- early_segment: Samples with history length < 16 (early in trajectory)

Usage:
    python build_testset.py \
        --data_path /path/to/R2R \
        --output testset/r2r_testset.json \
        --num_common 30 --num_long 10 --num_early 10 \
        --seed 42
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directories to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_vln_dir = os.path.dirname(os.path.dirname(_current_dir))
if _vln_dir not in sys.path:
    sys.path.insert(0, _vln_dir)


@dataclass
class TestSample:
    """A single test sample definition."""
    id: str
    category: str  # 'common', 'long_trajectory', 'early_segment'
    episode_id: int
    instruction_id: int
    start_idx: int
    history_length: int
    total_actions: int
    video_path: str


class TestSetBuilder:
    """
    Builds fixed test sets from trajectory data.
    
    The test set is saved as JSON and can be loaded later for consistent
    evaluation across different memory strategies.
    """
    
    def __init__(
        self,
        data_path: str,
        num_frames: int = 32,
        num_future_steps: int = 4,
        seed: int = 42,
    ):
        """
        Initialize the test set builder.
        
        Args:
            data_path: Path to trajectory data folder (contains annotations.json)
            num_frames: Number of frames per window (default: 32)
            num_future_steps: Actions per turn (default: 4, for K=4 chunks)
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.num_frames = num_frames
        self.num_future_steps = num_future_steps
        self.seed = seed
        
        # Load annotation data
        self.nav_data = self._load_annotations()
        
        # Build all possible sample points
        self.all_samples = self._build_sample_index()
        
        print(f"[TestSetBuilder] Loaded {len(self.nav_data)} episodes")
        print(f"[TestSetBuilder] Found {len(self.all_samples)} valid sample points")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from data path."""
        video_folders = [p.strip() for p in self.data_path.split(',') if p.strip()]
        nav_data = []
        
        for vf in video_folders:
            anno_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_path):
                print(f"Warning: {anno_path} not found, skipping...")
                continue
            
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
                tdata['_source_folder'] = vf
            
            nav_data.extend(anno_json)
            print(f"Loaded {len(anno_json)} episodes from {vf}")
        
        return nav_data
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all valid sample points.
        
        Each sample point represents a position in a trajectory where
        we can evaluate the model's decision quality.
        
        Uses fine-grained stride (num_future_steps) to ensure we have samples
        with various history lengths, especially for early_segment category.
        
        Returns:
            List of sample dictionaries with metadata
        """
        samples = []
        # Use fine-grained stride to capture early segments (0 < history < 16)
        stride = self.num_future_steps  # 4 steps per sample point
        
        for ep_id, item in enumerate(self.nav_data):
            instructions = item.get('instructions', [])
            actions = item.get('actions', [])
            actions_len = len(actions)
            video_path = item.get('video', '')
            
            if actions_len < self.num_future_steps:
                continue
            
            if not isinstance(instructions, list):
                instructions = [instructions]
            
            for ins_id in range(len(instructions)):
                # Generate sample points with fine-grained stride
                for start_idx in range(0, actions_len, stride):
                    # Calculate history length (number of steps before this point)
                    history_length = start_idx
                    
                    # Skip if not enough actions remaining
                    remaining_actions = actions_len - start_idx
                    if remaining_actions < self.num_future_steps:
                        continue
                    
                    sample = {
                        'episode_id': ep_id,
                        'instruction_id': ins_id,
                        'start_idx': start_idx,
                        'history_length': history_length,
                        'total_actions': actions_len,
                        'video_path': video_path,
                    }
                    samples.append(sample)
        
        return samples
    
    def _categorize_samples(self) -> Dict[str, List[Dict]]:
        """
        Categorize samples by history length.
        
        Categories:
        - common: history 16-64 (normal navigation scenarios)
        - long_trajectory: history > 64 (multiple window slides)
        - early_segment: 8 <= history < 16 (early in trajectory, has some history)
        
        Returns:
            Dictionary mapping category names to sample lists
        """
        categories = {
            'common': [],           # history 16-64
            'long_trajectory': [],  # history > 64
            'early_segment': [],    # 8 <= history < 16
        }
        
        for sample in self.all_samples:
            history_len = sample['history_length']
            
            if 8 <= history_len < 16:
                # Early segment: has 8-15 steps of history
                categories['early_segment'].append(sample)
            elif history_len > 64:
                categories['long_trajectory'].append(sample)
            elif 16 <= history_len <= 64:
                categories['common'].append(sample)
        
        for cat, samples in categories.items():
            print(f"[TestSetBuilder] Category '{cat}': {len(samples)} available samples")
        
        return categories
    
    def build_testset(
        self,
        num_common: int = 30,
        num_long: int = 10,
        num_early: int = 10,
    ) -> Dict:
        """
        Build a fixed test set by sampling from each category.
        
        Args:
            num_common: Number of common samples (history 16-64)
            num_long: Number of long trajectory samples (history > 64)
            num_early: Number of early segment samples (history < 16)
            
        Returns:
            Test set dictionary ready for JSON serialization
        """
        random.seed(self.seed)
        
        categories = self._categorize_samples()
        samples = []
        
        # Sample from each category
        category_counts = {
            'common': num_common,
            'long_trajectory': num_long,
            'early_segment': num_early,
        }
        
        for cat_name, count in category_counts.items():
            available = categories[cat_name]
            
            if len(available) < count:
                print(f"Warning: Only {len(available)} samples available for "
                      f"'{cat_name}', requested {count}")
                selected = available
            else:
                selected = random.sample(available, count)
            
            # Create TestSample objects with IDs
            for i, sample in enumerate(selected):
                test_sample = TestSample(
                    id=f"{cat_name}_{i+1:03d}",
                    category=cat_name,
                    episode_id=sample['episode_id'],
                    instruction_id=sample['instruction_id'],
                    start_idx=sample['start_idx'],
                    history_length=sample['history_length'],
                    total_actions=sample['total_actions'],
                    video_path=sample['video_path'],
                )
                samples.append(asdict(test_sample))
        
        # Build final test set structure
        testset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'source': self.data_path,
                'total_samples': len(samples),
                'seed': self.seed,
                'num_frames': self.num_frames,
                'num_future_steps': self.num_future_steps,
                'category_counts': {
                    'common': sum(1 for s in samples if s['category'] == 'common'),
                    'long_trajectory': sum(1 for s in samples if s['category'] == 'long_trajectory'),
                    'early_segment': sum(1 for s in samples if s['category'] == 'early_segment'),
                }
            },
            'samples': samples,
        }
        
        return testset
    
    def save_testset(self, testset: Dict, output_path: str):
        """Save test set to JSON file."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(testset, f, indent=2, ensure_ascii=False)
        
        print(f"[TestSetBuilder] Saved test set to: {output_path}")
        print(f"[TestSetBuilder] Total samples: {testset['metadata']['total_samples']}")


def main():
    parser = argparse.ArgumentParser(
        description='Build fixed test set for memory evaluation'
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to trajectory data folder (contains annotations.json)'
    )
    parser.add_argument(
        '--output', type=str, default='testset/r2r_testset.json',
        help='Output path for test set JSON'
    )
    parser.add_argument(
        '--num_common', type=int, default=30,
        help='Number of common samples (history 16-64)'
    )
    parser.add_argument(
        '--num_long', type=int, default=10,
        help='Number of long trajectory samples (history > 64)'
    )
    parser.add_argument(
        '--num_early', type=int, default=10,
        help='Number of early segment samples (history < 16)'
    )
    parser.add_argument(
        '--num_frames', type=int, default=32,
        help='Number of frames per window'
    )
    parser.add_argument(
        '--num_future_steps', type=int, default=4,
        help='Actions per turn (chunk size K)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Build test set
    builder = TestSetBuilder(
        data_path=args.data_path,
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        seed=args.seed,
    )
    
    testset = builder.build_testset(
        num_common=args.num_common,
        num_long=args.num_long,
        num_early=args.num_early,
    )
    
    builder.save_testset(testset, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Set Summary")
    print("=" * 60)
    for cat, count in testset['metadata']['category_counts'].items():
        print(f"  {cat}: {count} samples")
    print(f"  Total: {testset['metadata']['total_samples']} samples")
    print("=" * 60)


if __name__ == '__main__':
    main()
