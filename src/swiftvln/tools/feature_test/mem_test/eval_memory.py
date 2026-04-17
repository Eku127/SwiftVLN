# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Memory Strategy Evaluation Script

This script evaluates different memory construction strategies on a fixed
test set, computing NLL, redundancy, and coverage metrics.

Usage:
    python eval_memory.py \
        --checkpoint /path/to/checkpoint \
        --testset testset/r2r_testset.json \
        --data_path /path/to/R2R \
        --memory_strategy uniform \
        --output_dir ./results
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
import yaml
from PIL import Image

# Setup paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mem_test_dir = _current_dir
_feature_test_dir = os.path.dirname(_mem_test_dir)
_vln_dir = os.path.dirname(_feature_test_dir)
_msswift_root = os.path.dirname(os.path.dirname(_vln_dir))

for path in [_msswift_root, _vln_dir, _mem_test_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import memory strategies
from memory_strategies import get_strategy, STRATEGY_REGISTRY
from memory_strategies.base import MemoryOutput

# Import metrics
from metrics import NLLComputer, RedundancyComputer, CoverageComputer

# Import output utilities
from output.visualizer import ResultVisualizer, EvalResult


# Constants (from OverlapVLN)
HISTORY_IMAGE_TOKEN = "<history_image>"  # Legacy per-frame token (deprecated)
HISTORY_MEMORY_TOKEN = "<history_memory>"  # Unified memory token
CURRENT_IMAGE_TOKEN = "<current_image>"

UNIFIED_PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Based on your observations, output a sequence of actions using: "
    "↑ (forward), ← (turn left), → (turn right), or STOP (when goal is reached). "
    "Output actions directly without explanation."
)

CONJUNCTIONS = [
    'you can see ',
    'in front of you is ',
    'there is ',
    'you can spot ',
    'you are toward the ',
    'ahead of you is ',
    'in your sight is '
]


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with inheritance support.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Merged configuration dictionary
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance (_base_ field)
    if '_base_' in config:
        base_path = config.pop('_base_')
        if not os.path.isabs(base_path):
            base_path = os.path.join(config_dir, base_path)
        
        # Load base config recursively
        base_config = load_config(base_path)
        
        # Deep merge: config overrides base_config
        config = deep_merge(base_config, config)
    
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


class MemoryEvaluator:
    """
    Evaluator for memory strategies.
    
    This class handles:
    1. Loading the model and test set
    2. Building prompts and memory
    3. Computing evaluation metrics
    4. Outputting results
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        testset_path: str,
        data_path: str,
        memory_strategy: str = 'uniform',
        memory_strategy_params: Optional[Dict[str, Any]] = None,
        output_dir: str = './results',
        device: str = 'cuda',
        num_history: int = 8,
        compress_stride: int = 2,
        num_future_steps: int = 4,
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            testset_path: Path to fixed test set JSON
            data_path: Path to trajectory data folder
            memory_strategy: Name of memory strategy to use
            memory_strategy_params: Additional params for memory strategy
            output_dir: Directory for output files
            device: Device for computation
            num_history: Number of history frames
            compress_stride: Compression stride
            num_future_steps: Actions per chunk
        """
        self.checkpoint_path = checkpoint_path
        self.testset_path = testset_path
        self.data_path = data_path
        self.memory_strategy_name = memory_strategy
        self.memory_strategy_params = memory_strategy_params or {}
        self.output_dir = output_dir
        self.device = device
        self.num_history = num_history
        self.compress_stride = compress_stride
        self.num_future_steps = num_future_steps
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Will be initialized in setup()
        self.model = None
        self.processor = None
        self.testset = None
        self.nav_data = None
        self.memory_strategy = None
        self.nll_computer = None
        self.redundancy_computer = None
        self.coverage_computer = None
        self.visualizer = None
    
    def setup(self):
        """Setup model, data, and strategy."""
        print("=" * 60)
        print("Memory Strategy Evaluation Setup")
        print("=" * 60)
        
        # Load model
        self._load_model()
        
        # Load test set
        self._load_testset()
        
        # Load navigation data
        self._load_nav_data()
        
        # Initialize memory strategy
        self._init_memory_strategy()
        
        # Initialize metric computers
        self._init_metrics()
        
        # Initialize visualizer with strategy-specific subdirectory
        self.visualizer = ResultVisualizer(self.output_dir, self.memory_strategy_name)
        
        print("Setup complete!\n")
    
    def _load_model(self):
        """Load model and processor."""
        print(f"\nLoading model from: {self.checkpoint_path}")
        
        # Register OverlapVLN model
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
        
        # Get special token IDs
        self.history_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(HISTORY_IMAGE_TOKEN)
        self.history_memory_token_id = self.processor.tokenizer.convert_tokens_to_ids(HISTORY_MEMORY_TOKEN)
        self.current_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(CURRENT_IMAGE_TOKEN)
        
        print(f"  Model loaded successfully")
        print(f"  History memory token ID: {self.history_memory_token_id} (unified)")
        print(f"  History image token ID: {self.history_image_token_id} (legacy)")
        print(f"  Current token ID: {self.current_image_token_id}")
    
    def _load_testset(self):
        """Load fixed test set."""
        print(f"\nLoading test set from: {self.testset_path}")
        
        with open(self.testset_path, 'r') as f:
            self.testset = json.load(f)
        
        print(f"  Total samples: {self.testset['metadata']['total_samples']}")
        for cat, count in self.testset['metadata'].get('category_counts', {}).items():
            print(f"    {cat}: {count}")
    
    def _load_nav_data(self):
        """Load navigation data."""
        print(f"\nLoading navigation data from: {self.data_path}")
        
        video_folders = [p.strip() for p in self.data_path.split(',') if p.strip()]
        self.nav_data = []
        
        for vf in video_folders:
            anno_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_path):
                print(f"  Warning: {anno_path} not found, skipping...")
                continue
            
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            
            self.nav_data.extend(anno_json)
            print(f"  Loaded {len(anno_json)} episodes from {vf}")
    
    def _init_memory_strategy(self):
        """Initialize memory strategy."""
        print(f"\nInitializing memory strategy: {self.memory_strategy_name}")
        
        # Build strategy kwargs
        strategy_kwargs = {
            'num_history': self.num_history,
            'compress_stride': self.compress_stride,
            'device': self.device,
        }
        # Add any strategy-specific params
        strategy_kwargs.update(self.memory_strategy_params)
        
        self.memory_strategy = get_strategy(
            self.memory_strategy_name,
            **strategy_kwargs
        )
        
        # Set encoder
        self.memory_strategy.set_encoder(self.model, self.processor)
        
        print(f"  Strategy: {self.memory_strategy.name}")
        if self.memory_strategy_params:
            print(f"  Extra params: {self.memory_strategy_params}")
    
    def _init_metrics(self):
        """Initialize metric computers."""
        print("\nInitializing metric computers...")
        
        self.nll_computer = NLLComputer(
            self.model, self.processor, self.device, self.num_future_steps
        )
        self.redundancy_computer = RedundancyComputer(self.device)
        self.coverage_computer = CoverageComputer(self.device)
    
    def _load_sample_data(self, sample: Dict) -> Tuple[List[Image.Image], List[int], str]:
        """
        Load data for a single sample.
        
        Args:
            sample: Sample definition from test set
            
        Returns:
            Tuple of (history_frames, target_actions, instruction)
        """
        episode_id = sample['episode_id']
        instruction_id = sample['instruction_id']
        start_idx = sample['start_idx']
        
        data = self.nav_data[episode_id]
        video_path = data['video']
        rgb_path = os.path.join(video_path, 'rgb')
        
        # Get instruction
        instructions = data.get('instructions', [])
        if not isinstance(instructions, list):
            instructions = [instructions]
        instruction = instructions[instruction_id] if instruction_id < len(instructions) else instructions[0]
        
        # Get actions
        actions = data['actions'][1:] + [0]  # Shift by 1
        
        # Get target actions for this chunk
        target_actions = actions[start_idx:start_idx + self.num_future_steps]
        if len(target_actions) < self.num_future_steps:
            target_actions = target_actions + [0] * (self.num_future_steps - len(target_actions))
        
        # Load history frames
        video_frames = sorted(os.listdir(rgb_path))
        history_frames = []
        
        if start_idx > 0:
            for i in range(start_idx):
                frame_path = os.path.join(rgb_path, video_frames[min(i, len(video_frames) - 1)])
                try:
                    img = Image.open(frame_path).convert('RGB')
                    history_frames.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load {frame_path}: {e}")
                    history_frames.append(Image.new('RGB', (640, 480), color='black'))
        
        # Load current frame
        current_frame_path = os.path.join(rgb_path, video_frames[min(start_idx, len(video_frames) - 1)])
        try:
            current_frame = Image.open(current_frame_path).convert('RGB')
        except Exception:
            current_frame = Image.new('RGB', (640, 480), color='black')
        
        return history_frames, current_frame, target_actions, instruction
    
    def _build_prompt_embeds(
        self,
        instruction: str,
        memory_output: MemoryOutput,
        current_frame: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build complete prompt embeddings.
        
        Args:
            instruction: Navigation instruction
            memory_output: Output from memory strategy
            current_frame: Current observation image
            
        Returns:
            Tuple of (inputs_embeds, attention_mask)
        """
        # Encode current frame
        current_features, current_grid_thw = self.memory_strategy.encode_frames([current_frame])
        current_vit = current_features[0] if current_features else torch.empty(0)
        current_token_count = current_vit.shape[0] if current_vit.numel() > 0 else 0
        
        # Build system prompt
        system_prompt = UNIFIED_PROMPT_TEMPLATE.format(instruction=instruction)
        
        memory_tokens = memory_output.memory_tokens
        num_memory_tokens = memory_tokens.shape[0] if memory_tokens.numel() > 0 else 0
        
        # Unified memory mode: single <history_memory> block with all tokens
        if num_memory_tokens > 0:
            # All history embeddings are packed into a single unified block
            history_str = f'<|vision_start|>{HISTORY_MEMORY_TOKEN * num_memory_tokens}<|vision_end|>'
            system_prompt += f" These are your historical observations: {history_str}."
        
        # Build user turn
        conjunction = random.choice(CONJUNCTIONS)
        current_tokens = CURRENT_IMAGE_TOKEN * current_token_count
        user_content = f"{conjunction}<|vision_start|>{current_tokens}<|vision_end|>."
        
        # Format messages
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ]
        
        # Tokenize
        inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        
        # Get text embeddings
        text_embeds = self._get_text_embeddings(input_ids)
        
        # Replace unified history memory tokens
        if num_memory_tokens > 0:
            history_positions = (input_ids[0] == self.history_memory_token_id).nonzero(as_tuple=True)[0]
            if len(history_positions) >= num_memory_tokens:
                positions = history_positions[:num_memory_tokens]
                mem_embeds = memory_tokens.to(text_embeds.device, text_embeds.dtype)
                text_embeds[0, positions] = mem_embeds
        
        # Replace current image tokens
        if current_token_count > 0:
            current_positions = (input_ids[0] == self.current_image_token_id).nonzero(as_tuple=True)[0]
            if len(current_positions) >= current_token_count:
                positions = current_positions[:current_token_count]
                cur_embeds = current_vit.to(text_embeds.device, text_embeds.dtype)
                text_embeds[0, positions] = cur_embeds
        
        # Create attention mask
        attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.long, device=self.device)
        
        return text_embeds, attention_mask
    
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from input_ids."""
        base_model = self.model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            return base_model.model.embed_tokens(input_ids)
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
            return base_model.model.language_model.embed_tokens(input_ids)
        else:
            raise ValueError("Cannot find embed_tokens in model")
    
    @torch.no_grad()
    def evaluate_sample(self, sample: Dict) -> EvalResult:
        """
        Evaluate a single sample.
        
        Args:
            sample: Sample definition from test set
            
        Returns:
            EvalResult with computed metrics
        """
        # Load sample data
        history_frames, current_frame, target_actions, instruction = self._load_sample_data(sample)
        
        # Build memory
        memory_output = self.memory_strategy.build_memory(
            history_frames=history_frames,
            current_start_idx=sample['start_idx'],
        )
        
        # Build prompt embeddings
        inputs_embeds, attention_mask = self._build_prompt_embeds(
            instruction=instruction,
            memory_output=memory_output,
            current_frame=current_frame,
        )
        
        # Compute NLL
        nll_result = self.nll_computer.compute_nll(
            inputs_embeds=inputs_embeds,
            target_actions=target_actions,
            attention_mask=attention_mask,
        )
        
        # Compute redundancy at both frame and token level
        # Frame-level: measures similarity between the 8 selected frames (more discriminative)
        # Token-level: measures similarity between all 704 tokens
        memory_tokens = memory_output.memory_tokens
        if memory_tokens.numel() > 0:
            # Get tokens per frame from metadata
            # compressed_tokens_per_frame is a list like [88, 88, 88, ...], use first value or default
            compressed_tokens_list = memory_output.metadata.get('compressed_tokens_per_frame', [])
            if compressed_tokens_list and len(compressed_tokens_list) > 0:
                # Use the first frame's token count (assuming uniform compression)
                tokens_per_frame = compressed_tokens_list[0]
            else:
                tokens_per_frame = 88  # Default for stride=2 compression
            
            redundancy_result = self.redundancy_computer.compute(
                memory_tokens, 
                tokens_per_frame=tokens_per_frame
            )
            redundancy_frame = redundancy_result.redundancy_frame
            redundancy_token = redundancy_result.redundancy_token
        else:
            redundancy_frame = 0.0
            redundancy_token = 0.0
        
        # Compute coverage using ALL history frames (not just selected ones)
        # This measures how well the memory represents the entire history
        coverage = 0.0
        if len(history_frames) > 0 and memory_tokens.numel() > 0:
            # Encode ALL history frames to get their VIT features
            all_history_features, _ = self.memory_strategy.encode_frames(history_frames)
            
            if all_history_features:
                # Stack all history frame features: [num_frames, tokens_per_frame, hidden]
                all_features = torch.stack(all_history_features)
                
                # Compute coverage: how well memory_tokens cover all history frames
                coverage_result = self.coverage_computer.compute(all_features, memory_tokens)
                coverage = coverage_result.coverage
        
        return EvalResult(
            sample_id=sample['id'],
            category=sample['category'],
            history_length=sample['history_length'],
            nll=nll_result.nll,
            ce=nll_result.ce,
            redundancy_frame=redundancy_frame,
            redundancy_token=redundancy_token,
            coverage=coverage,
            metadata={
                'num_memory_tokens': memory_tokens.shape[0] if memory_tokens.numel() > 0 else 0,
                'num_history_frames': len(history_frames),
                'num_selected_frames': len(memory_output.selected_indices) if memory_output.selected_indices else 0,
                'target_text': nll_result.target_text,
            }
        )
    
    def run(self) -> List[EvalResult]:
        """
        Run evaluation on all samples.
        
        Returns:
            List of EvalResult for all samples
        """
        print("\n" + "=" * 60)
        print(f"Running Evaluation: {self.memory_strategy_name}")
        print("=" * 60)
        
        results = []
        samples = self.testset['samples']
        
        for sample in tqdm(samples, desc="Evaluating"):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
            except Exception as e:
                print(f"\nError evaluating sample {sample['id']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def save_results(self, results: List[EvalResult]):
        """Save results to all output formats."""
        # JSON
        json_path = self.visualizer.save_json(
            results,
            self.memory_strategy_name,
            metadata={
                'checkpoint': self.checkpoint_path,
                'testset': self.testset_path,
                'num_history': self.num_history,
                'compress_stride': self.compress_stride,
            }
        )
        
        # CSV
        csv_path = self.visualizer.save_csv(results, self.memory_strategy_name)
        
        # Charts
        chart_paths = self.visualizer.generate_charts(results, self.memory_strategy_name)
        
        # Print summary
        self.visualizer.print_summary(results, self.memory_strategy_name)
        
        return {
            'json': json_path,
            'csv': csv_path,
            'charts': chart_paths,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate memory strategies for VLN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended):
  python eval_memory.py --config configs/uniform.yaml
  python eval_memory.py --config configs/uniform_tome.yaml
  
  # Using command line args (legacy):
  python eval_memory.py --checkpoint /path/to/checkpoint --testset ./testset/r2r_testset.json --data_path /path/to/R2R
        """
    )
    
    # Config file argument
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file (recommended)'
    )
    
    # Command line arguments (can override config)
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--testset', type=str, default=None,
        help='Path to fixed test set JSON'
    )
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to trajectory data folder'
    )
    parser.add_argument(
        '--memory_strategy', type=str, default=None,
        help='Memory strategy to evaluate'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--num_history', type=int, default=None,
        help='Number of history frames'
    )
    parser.add_argument(
        '--compress_stride', type=int, default=None,
        help='Compression stride for features'
    )
    parser.add_argument(
        '--num_future_steps', type=int, default=None,
        help='Actions per chunk (K)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device for computation'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        config = {}
    
    # Helper to get value with priority: CLI args > config > default
    def get_value(cli_val, config_section, config_key, default=None):
        if cli_val is not None:
            return cli_val
        if config_section in config and config_key in config[config_section]:
            return config[config_section][config_key]
        return default
    
    # Extract values
    checkpoint = get_value(args.checkpoint, 'model', 'checkpoint')
    testset = get_value(args.testset, 'data', 'testset_path', './testset/r2r_testset.json')
    data_path = get_value(args.data_path, 'data', 'data_path')
    memory_strategy = get_value(args.memory_strategy, 'memory_strategy', 'name', 'uniform')
    output_dir = get_value(args.output_dir, 'output', 'output_dir', './results')
    num_history = get_value(args.num_history, 'evaluation', 'num_history', 8)
    compress_stride = get_value(args.compress_stride, 'evaluation', 'compress_stride', 2)
    num_future_steps = get_value(args.num_future_steps, 'evaluation', 'num_future_steps', 4)
    seed = get_value(args.seed, 'evaluation', 'seed', 42)
    device = get_value(args.device, 'model', 'device', 'cuda')
    
    # Get strategy-specific params from config
    memory_strategy_params = config.get('memory_strategy', {}).get('params', {})
    
    # Validate required arguments
    if not checkpoint:
        parser.error("--checkpoint is required (or set in config)")
    if not data_path:
        parser.error("--data_path is required (or set in config)")
    
    # Set seed
    set_seed(seed)
    
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Testset: {testset}")
    print(f"  Data path: {data_path}")
    print(f"  Memory strategy: {memory_strategy}")
    if memory_strategy_params:
        print(f"  Strategy params: {memory_strategy_params}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num history: {num_history}")
    print(f"  Compress stride: {compress_stride}")
    print(f"  Seed: {seed}")
    print()
    
    # Create evaluator
    evaluator = MemoryEvaluator(
        checkpoint_path=checkpoint,
        testset_path=testset,
        data_path=data_path,
        memory_strategy=memory_strategy,
        memory_strategy_params=memory_strategy_params,
        output_dir=output_dir,
        device=device,
        num_history=num_history,
        compress_stride=compress_stride,
        num_future_steps=num_future_steps,
    )
    
    # Setup
    evaluator.setup()
    
    # Run evaluation
    results = evaluator.run()
    
    # Save results
    output_paths = evaluator.save_results(results)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to:")
    for key, path in output_paths.items():
        if isinstance(path, list):
            for p in path:
                print(f"  {key}: {p}")
        else:
            print(f"  {key}: {path}")


if __name__ == '__main__':
    main()
