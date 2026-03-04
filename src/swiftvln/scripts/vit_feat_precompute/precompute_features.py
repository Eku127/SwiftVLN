# Copyright (c) Alibaba, Inc. and its affiliates.
"""
UniNaVid Feature Precomputation Script

Precomputes ViT features for all frames in VLN dataset episodes.
Supports multi-GPU distributed processing for faster computation.

Usage:
    # Single GPU
    python src/swiftvln/scripts/vit_feat_precompute/precompute_features.py \
        --data_path /path/to/dataset \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct
    
    # Multi-GPU (distributed)
    torchrun --nproc_per_node=8 src/swiftvln/scripts/vit_feat_precompute/precompute_features.py \
        --data_path /path/to/dataset \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct

Output:
    {data_path}/features/{episode_id}.pt
    
    Each .pt file contains:
    {
        'vit_features': List[torch.Tensor],  # [num_tokens, 2048], bfloat16
        'grid_thw': List[torch.Tensor],      # [3] per frame
    }
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
import torch.distributed as dist
from PIL import Image

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_msswift_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
if _msswift_root not in sys.path:
    sys.path.insert(0, _msswift_root)


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(model_path: str, device: torch.device):
    """Load Qwen2.5-VL model for ViT feature extraction."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print(f"Loading model from {model_path}...")
    
    # Load processor for image preprocessing
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model (we only need the visual encoder)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    return model, processor


def get_episodes(data_path: str) -> List[Dict[str, Any]]:
    """Load episode metadata from annotations.json."""
    anno_path = os.path.join(data_path, 'annotations.json')
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotations not found: {anno_path}")
    
    with open(anno_path, 'r') as f:
        episodes = json.load(f)
    
    # Add full video path
    for ep in episodes:
        ep['video'] = os.path.join(data_path, ep['video'])
    
    return episodes


def get_episode_id(episode: Dict[str, Any]) -> str:
    """Extract episode ID from video path."""
    video_path = episode['video']
    return os.path.basename(video_path.rstrip('/'))


def process_episode(
    episode: Dict[str, Any],
    model: torch.nn.Module,
    processor,
    device: torch.device,
    output_dir: str,
    batch_size: int = 8,
) -> bool:
    """
    Process all frames in an episode and save features.
    
    Args:
        episode: Episode metadata
        model: Qwen2.5-VL model
        processor: Image processor
        device: CUDA device
        output_dir: Directory to save features
        batch_size: Number of frames to process at once
        
    Returns:
        True if successful, False if skipped or failed
    """
    episode_id = get_episode_id(episode)
    output_path = os.path.join(output_dir, f'{episode_id}.pt')
    
    # Skip if already processed
    if os.path.exists(output_path):
        return False
    
    video_path = episode['video']
    rgb_path = os.path.join(video_path, 'rgb')
    
    if not os.path.exists(rgb_path):
        print(f"Warning: RGB path not found: {rgb_path}")
        return False
    
    frame_files = sorted(os.listdir(rgb_path))
    if not frame_files:
        print(f"Warning: No frames found in {rgb_path}")
        return False
    
    features_list = []
    grid_thw_list = []
    
    # Process frames in batches
    # Note: do NOT use do_resize=False - it causes reshape errors in fast image processor
    for batch_start in range(0, len(frame_files), batch_size):
        batch_end = min(batch_start + batch_size, len(frame_files))
        batch_frames = frame_files[batch_start:batch_end]
        
        # Load images
        images = []
        for frame_file in batch_frames:
            frame_path = os.path.join(rgb_path, frame_file)
            try:
                image = Image.open(frame_path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load {frame_path}: {e}")
                # Use black image as fallback
                images.append(Image.new('RGB', (640, 480), color='black'))
        
        # Process batch through image processor (let it resize to standard format)
        media_inputs = processor.image_processor(
            images=images, return_tensors='pt'
        )
        
        pixel_values = media_inputs['pixel_values'].to(device, torch.bfloat16)
        image_grid_thw = media_inputs['image_grid_thw'].to(device)
        
        # Extract ViT features
        with torch.no_grad():
            vit_features = model.visual(pixel_values, grid_thw=image_grid_thw)
        
        # Split features by frame
        merge_size = processor.image_processor.merge_size
        merge_length = merge_size ** 2
        
        embed_idx = 0
        for i in range(len(images)):
            num_tokens = int(image_grid_thw[i].prod() // merge_length)
            frame_features = vit_features[embed_idx:embed_idx + num_tokens]
            embed_idx += num_tokens
            
            # Save features to CPU with bfloat16
            features_list.append(frame_features.cpu().bfloat16())
            grid_thw_list.append(image_grid_thw[i].cpu())
    
    # Save to file
    torch.save({
        'vit_features': features_list,
        'grid_thw': grid_thw_list,
    }, output_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Precompute ViT features for UniNaVid')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to VLN dataset (contains annotations.json)')
    parser.add_argument('--model_path', type=str, 
                        default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Path to Qwen2.5-VL model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for features (default: {data_path}/features)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing frames')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    is_main = (rank == 0)
    
    # Setup output directory
    output_dir = args.output_dir or os.path.join(args.data_path, 'features')
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    
    # Sync before checking directory
    if world_size > 1:
        dist.barrier()
    
    # Load all episodes
    all_episodes = get_episodes(args.data_path)
    
    # === Phase 1: Analyze missing episodes (main process only) ===
    if is_main:
        print(f"=" * 60)
        print(f"Analyzing dataset: {args.data_path}")
        print(f"Output directory: {output_dir}")
        print(f"=" * 60)
        
        existing_features = set(f[:-3] for f in os.listdir(output_dir) if f.endswith('.pt'))
        
        missing_episodes = []
        for ep in all_episodes:
            episode_id = get_episode_id(ep)
            if episode_id not in existing_features:
                missing_episodes.append(ep)
        
        print(f"Total episodes: {len(all_episodes)}")
        print(f"Already processed: {len(existing_features)}")
        print(f"Missing (to process): {len(missing_episodes)}")
        print(f"=" * 60)
        
        if len(missing_episodes) == 0:
            print("All episodes already processed! Nothing to do.")
    else:
        missing_episodes = None
    
    # Sync and broadcast missing episodes list
    if world_size > 1:
        import pickle
        if is_main:
            # Serialize and broadcast size
            data = pickle.dumps(missing_episodes)
            size_tensor = torch.tensor([len(data)], dtype=torch.long, device=device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
        
        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()
        
        if size == 0:
            cleanup_distributed()
            return
        
        # Broadcast data
        if is_main:
            data_tensor = torch.ByteTensor(list(data)).to(device)
        else:
            data_tensor = torch.ByteTensor(size).to(device)
        
        dist.broadcast(data_tensor, src=0)
        
        if not is_main:
            missing_episodes = pickle.loads(bytes(data_tensor.cpu().tolist()))
    else:
        if len(missing_episodes) == 0:
            return
    
    # Load model (only after we know there's work to do)
    if is_main:
        print(f"\nLoading model...")
    model, processor = load_model(args.model_path, device)
    
    # === Phase 2: Distribute missing episodes across GPUs ===
    episodes_per_gpu = []
    for i, ep in enumerate(missing_episodes):
        if i % world_size == rank:
            episodes_per_gpu.append(ep)
    
    if is_main:
        print(f"\nStarting processing with {world_size} GPUs")
        print(f"Episodes per GPU: ~{len(missing_episodes) // world_size}")
        print(f"=" * 60)
    
    # Sync before processing
    if world_size > 1:
        dist.barrier()
    
    # Process episodes
    pbar = tqdm(episodes_per_gpu, desc=f'GPU {rank}', disable=not is_main)
    processed = 0
    errors = 0
    
    for episode in pbar:
        try:
            process_episode(episode, model, processor, device, output_dir, args.batch_size)
            processed += 1
            pbar.set_postfix({'processed': processed, 'errors': errors})
        except Exception as e:
            episode_id = get_episode_id(episode)
            print(f"[GPU {rank}] Error processing {episode_id}: {e}")
            errors += 1
    
    # Sync and report
    if world_size > 1:
        dist.barrier()
    
    if is_main:
        # Count total features
        feature_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
        print(f"\n" + "=" * 60)
        print(f"Precomputation complete!")
        print(f"Total feature files: {len(feature_files)}/{len(all_episodes)}")
        print(f"Output directory: {output_dir}")
        print(f"=" * 60)
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
