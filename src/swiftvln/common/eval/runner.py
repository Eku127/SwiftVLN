# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base VLN Evaluation Entry Point

This module provides a base class for VLN evaluation across different models
(StreamVLN, MonoVLN, CompressVLN) and environments (Habitat, SatNav).

Subclasses need to override:
    - model_type: str
    - template_type: str
    - evaluator_class: Type
    - get_model_specific_args(): Add model-specific arguments
    - get_summary_extras(): Return model-specific summary fields

Usage:
    class MyVLNEval(BaseVLNEval):
        model_type = 'my_vln_model'
        template_type = 'my_vln_template'
        ...
    
    if __name__ == "__main__":
        eval = MyVLNEval()
        eval.run()
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import tqdm
from abc import ABC, abstractmethod
from typing import Type, Optional, Dict, Any, List

from .reporting import (
    clean_results_for_output,
    compute_weighted_trajectory_type_metrics,
    compute_trajectory_type_stats,
    get_swanlab_url,
    get_swanlab_url_from_train_metadata,
    load_satnav_reference_distribution,
    save_timing_stats,
)

# ============================================================================
# Default random seed for reproducible evaluation
# ============================================================================
DEFAULT_EVAL_SEED = 42

# ============================================================================
# Debug logging for specific rank
# ============================================================================
_DEBUG_RANK = int(os.environ.get('SATNAV_DEBUG_RANK', '-1'))
_DEBUG_LOG_FILE = os.environ.get('SATNAV_DEBUG_LOG', None)
_debug_file_handle = None

def _debug_log(rank: int, msg: str):
    """Log debug message if debugging is enabled for this rank."""
    global _debug_file_handle
    if rank == _DEBUG_RANK:
        log_msg = f"[Rank {rank}][BaseEval] {msg}"
        if _DEBUG_LOG_FILE:
            if _debug_file_handle is None:
                _debug_file_handle = open(_DEBUG_LOG_FILE, 'a')
            _debug_file_handle.write(log_msg + '\n')
            _debug_file_handle.flush()
        else:
            print(log_msg, file=sys.stderr, flush=True)


class BaseVLNEval(ABC):
    """Base class for VLN evaluation.
    
    Provides common functionality for:
    - Distributed initialization
    - Model loading
    - Environment configuration
    - Evaluation loop
    - Metrics aggregation
    - Results saving
    """
    
    # Subclass must override these
    model_type: str = None
    template_type: str = None
    evaluator_class: Type = None
    model_description: str = "VLN"
    
    # Optional: Set to True if model uses compression
    uses_compression: bool = False
    # Optional: Set to True if model uses num_frames (streaming)
    uses_num_frames: bool = False
    
    def __init__(self):
        self.args = None
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        
        # Paths
        self._current_dir = os.path.dirname(os.path.abspath(__file__))
        self._common_root = os.path.dirname(self._current_dir)
        self._package_root = os.path.dirname(self._common_root)
        # package_root: <repo>/src/swiftvln -> repo_root: <repo>
        self._repo_root = os.path.dirname(os.path.dirname(self._package_root))
    
    @staticmethod
    def set_seed(seed: int = DEFAULT_EVAL_SEED):
        """Set random seeds for reproducible evaluation.
        
        Args:
            seed: Random seed value (default: 42)
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Note: We don't set torch.backends.cudnn.deterministic = True
        # as it significantly slows down evaluation
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with common arguments."""
        parser = argparse.ArgumentParser(description=f"{self.model_description} Multi-Environment Evaluation")
        
        # Model and data
        parser.add_argument("--model_path", type=str, required=True, 
                            help="Path to trained checkpoint")
        parser.add_argument("--env-type", type=str, default='habitat', choices=['habitat', 'satnav'],
                            help="Environment type: habitat (indoor) or satnav (satellite map)")
        parser.add_argument("--habitat_config_path", type=str, default='configs/vln_r2r.yaml', 
                            help="Path to habitat yaml (relative to package root)")
        parser.add_argument("--satnav-config", type=str, default='configs/satnav_task.yaml',
                            help="Path to satnav config yaml (relative to package root)")
        parser.add_argument("--eval_split", type=str, default='val_unseen', 
                            help="Dataset split to evaluate")
        
        # VLN parameters (common)
        parser.add_argument("--num_history", type=int, default=8, 
                            help="Number of history frames to sample")
        parser.add_argument("--num_future_steps", type=int, default=4, 
                            help="Number of actions to predict per step")
        
        # Optional: num_frames for streaming models
        if self.uses_num_frames:
            parser.add_argument("--num_frames", type=int, default=32, 
                                help="Streaming window size")
        
        # Optional: compression parameters
        if self.uses_compression:
            parser.add_argument("--compress_stride", type=int, default=2,
                                help="Pooling stride for history frame compression")
        
        # Output
        parser.add_argument("--output_dir", type=str, default=f'./results/eval/{self.model_type}', 
                            help="Output directory for results")
        parser.add_argument("--save_video", action="store_true", 
                            help="Save visualization videos")
        parser.add_argument("--video_compression", action="store_true",
                            help="Compress videos into zip files")
        
        # Execution mode
        parser.add_argument("--distributed", action="store_true",
                            help="Enable distributed evaluation (use with torchrun)")
        parser.add_argument("--max_episodes", type=int, default=None,
                            help="Maximum number of episodes to evaluate (for debugging)")
        parser.add_argument("--debug_timing", action="store_true",
                            help="Print detailed timing statistics for each episode")
        
        # Add model-specific arguments
        self.add_model_specific_args(parser)
        
        return parser
    
    def add_model_specific_args(self, parser: argparse.ArgumentParser):
        """Override to add model-specific arguments."""
        pass
    
    def init_distributed(self):
        """Initialize distributed training."""
        try:
            from swiftvln.common import init_distributed as _init_distributed
            return _init_distributed()
        except ImportError:
            try:
                from ..utils import init_distributed as _init_distributed
                return _init_distributed()
            except ImportError:
                return 0, 1, 0
    
    def register_module(self):
        """Override to import/register model module."""
        pass
    
    def load_model(self):
        """Load model and processor."""
        from swift.llm import get_model_tokenizer
        
        # Device mapping based on mode
        if self.world_size > 1:
            device_map = {'': self.local_rank}
        else:
            device_map = 'auto'
        
        model, processor = get_model_tokenizer(
            model_id_or_path=self.args.model_path,
            model_type=self.model_type,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_impl='flash_attn',
        )
        
        return model, processor
    
    def load_template(self, processor):
        """Load template for inference."""
        from swift.llm import get_template
        
        template = get_template(
            template_type=self.template_type,
            processor=processor
        )
        
        return template
    
    def configure_template(self, template):
        """Configure template with model-specific settings. Override if needed."""
        if self.uses_compression and hasattr(self.args, 'compress_stride'):
            if hasattr(template, 'compress_stride'):
                template.compress_stride = self.args.compress_stride
                if hasattr(template, 'compressor'):
                    template.compressor.stride = self.args.compress_stride
                if self.is_main:
                    print(f"[{self.model_description}] Template compress_stride set to {self.args.compress_stride}")
    
    def initialize_model(self, model):
        """Initialize model state. Override if needed."""
        if hasattr(model, 'reset'):
            model.reset(env_num=1)
    
    def resolve_config_path(self) -> str:
        """Resolve config path based on environment type."""
        if self.args.env_type == "habitat":
            config_path = self.args.habitat_config_path
        else:
            config_path = getattr(self.args, 'satnav_config', self.args.__dict__.get('satnav-config', 'configs/satnav_task.yaml'))
        
        if not os.path.isabs(config_path):
            path_wrt_pkg = os.path.join(self._package_root, config_path)
            path_wrt_repo = os.path.join(self._repo_root, config_path)
            
            if os.path.exists(path_wrt_pkg):
                config_path = path_wrt_pkg
            elif os.path.exists(path_wrt_repo):
                config_path = path_wrt_repo
            else:
                config_path = path_wrt_pkg
        
        return config_path
    
    def create_evaluator(self, config_path, model, processor, template):
        """Create evaluator instance."""
        return self.evaluator_class(
            config_path=config_path,
            model=model,
            processor=processor,
            template=template,
            args=self.args,
            env_type=self.args.env_type,
        )
    
    def get_episodes(self, env_wrapper) -> List:
        """Get episodes from environment."""
        if self.args.env_type == "habitat":
            all_episodes = env_wrapper.env.episodes
        else:
            all_episodes = env_wrapper.env._dataset.episodes
        
        if self.args.max_episodes is not None:
            all_episodes = all_episodes[:self.args.max_episodes]
        
        return all_episodes
    
    def distribute_episodes(self, all_episodes) -> List:
        """Distribute episodes across processes."""
        # Group episodes by scene
        scene_episode_dict = {}
        for episode in all_episodes:
            if hasattr(episode, 'scene_id'):
                scene_id = episode.scene_id
            else:
                scene_id = "default_scene"
            if scene_id not in scene_episode_dict:
                scene_episode_dict[scene_id] = []
            scene_episode_dict[scene_id].append(episode)
        
        # Build episode list for this rank
        my_episodes = []
        for scene_id in sorted(scene_episode_dict.keys()):
            scene_episodes = scene_episode_dict[scene_id]
            if self.world_size > 1:
                my_episodes.extend(scene_episodes[self.rank::self.world_size])
            else:
                my_episodes.extend(scene_episodes)
        
        return my_episodes, scene_episode_dict
    
    def evaluate_episode(self, evaluator, env_wrapper, episode) -> Dict:
        """Evaluate a single episode. Returns result dict."""
        if hasattr(episode, 'scene_id'):
            scene_name = os.path.basename(episode.scene_id).replace('.glb', '').replace('.tif', '')
        else:
            scene_name = "unknown"
        
        instruction_text = env_wrapper.get_instruction(episode)
        
        # Get trajectory_type for SatNav episodes
        trajectory_type = getattr(episode, 'trajectory_type', None)
        
        # Debug logging
        _debug_log(self.rank, f"=" * 70)
        _debug_log(self.rank, f"evaluate_episode() called for episode {episode.episode_id}")
        _debug_log(self.rank, f"  trajectory_type: {trajectory_type}")
        _debug_log(self.rank, f"  scene_id: {scene_name}")
        if hasattr(episode, 'start_position'):
            _debug_log(self.rank, f"  start_position: {episode.start_position}")
            _debug_log(self.rank, f"  start_rotation: {episode.start_rotation}")
        
        try:
            _debug_log(self.rank, f"  Calling evaluator.eval_episode()...")
            metrics = evaluator.eval_episode(env_wrapper, episode, env_idx=0)
            _debug_log(self.rank, f"  eval_episode() succeeded")
            _debug_log(self.rank, f"  metrics: success={metrics.get('success')}, spl={metrics.get('spl')}, steps={metrics.get('_step_count')}")
            
            result = {
                "episode_id": episode.episode_id,
                "scene_id": scene_name,
                "success": float(metrics.get("success", 0)),
                "spl": float(metrics.get("spl", 0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0)),
                "oracle_success": float(metrics.get("oracle_success", 0)),
                "steps": int(metrics.get("_step_count", 0)),
                "instruction": instruction_text,
            }
            
            # Add trajectory_type for SatNav episodes
            if trajectory_type is not None:
                result["trajectory_type"] = trajectory_type
            
            # Store timing stats internally for timing summary (not saved to all_results)
            if '_timing_stats' in metrics:
                result['_timing_stats'] = metrics.get('_timing_stats', {})
                result['_total_time'] = metrics.get('_total_time', 0.0)
                result['_step_count'] = metrics.get('_step_count', 0)
            
            # Add error info if evaluator caught an exception internally (e.g., out-of-bounds)
            if '_error' in metrics:
                result['error'] = metrics.get('_error')
            
            # Add error tags if available
            if "error_tags" in metrics:
                result["error_tags"] = metrics.get("error_tags", [])
                result["had_deviation"] = metrics.get("had_deviation", False)
                result["deviation_recovered"] = metrics.get("deviation_recovered", False)
            
            if self.world_size > 1:
                result["rank"] = self.rank
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"\n[Rank {self.rank}] Error on episode {episode.episode_id}: {e}")
            _debug_log(self.rank, f"  eval_episode() FAILED!")
            _debug_log(self.rank, f"  Error: {e}")
            _debug_log(self.rank, f"  Traceback:\n{error_traceback}")
            result = {
                "episode_id": episode.episode_id,
                "scene_id": scene_name,
                "success": 0.0,
                "spl": 0.0,
                "distance_to_goal": float('inf'),
                "oracle_success": 0.0,
                "steps": 0,
                "error": str(e),
                "instruction": instruction_text,
            }
            # Add trajectory_type even for error cases
            if trajectory_type is not None:
                result["trajectory_type"] = trajectory_type
            if self.args.env_type == "habitat":
                result["error_tags"] = []
                result["had_deviation"] = False
                result["deviation_recovered"] = False
        
        _debug_log(self.rank, f"evaluate_episode() finished for episode {episode.episode_id}")
        return result
    
    def gather_results(self, results: List[Dict]):
        """Gather results from all processes."""
        try:
            from swiftvln.common import gather_metrics
        except ImportError:
            try:
                from ..utils import gather_metrics
            except ImportError:
                gather_metrics = None
        
        if gather_metrics is not None:
            sucs_all, spls_all, oss_all, nes_all, all_results_merged = gather_metrics(
                results=results,
                rank=self.rank,
                world_size=self.world_size,
                local_rank=self.local_rank,
                is_main=self.is_main
            )
        else:
            # Fallback for single process
            sucs_all = torch.tensor([r["success"] for r in results])
            spls_all = torch.tensor([r["spl"] for r in results])
            oss_all = torch.tensor([r["oracle_success"] for r in results])
            nes_all = torch.tensor([r["distance_to_goal"] for r in results])
            all_results_merged = results
        
        return sucs_all, spls_all, oss_all, nes_all, all_results_merged
    
    def get_summary_extras(self) -> Dict[str, Any]:
        """Override to add model-specific summary fields."""
        extras = {}
        if self.uses_num_frames and hasattr(self.args, 'num_frames'):
            extras['num_frames'] = self.args.num_frames
        if self.uses_compression and hasattr(self.args, 'compress_stride'):
            extras['compress_stride'] = self.args.compress_stride
        return extras
    
    def save_results(self, sucs_all, spls_all, oss_all, nes_all, all_results_merged, timing_stats_list):
        """Save evaluation results."""
        total_episodes = len(sucs_all)
        success_rate = (sucs_all.sum() / total_episodes).item() if total_episodes > 0 else 0
        mean_spl = (spls_all.sum() / total_episodes).item() if total_episodes > 0 else 0
        mean_os = (oss_all.sum() / total_episodes).item() if total_episodes > 0 else 0
        valid_nes = nes_all[nes_all < 1000]
        mean_ne = valid_nes.mean().item() if len(valid_nes) > 0 else 0
        
        # Compute average steps
        all_steps = [r.get("steps", 0) for r in all_results_merged]
        avg_steps = sum(all_steps) / len(all_steps) if all_steps else 0
        
        summary = {
            "eval_split": self.args.eval_split,
            "success_rate": success_rate,
            "mean_spl": mean_spl,
            "oracle_success": mean_os,
            "navigation_error": mean_ne,
            "avg_steps": round(avg_steps, 2),
            "total_episodes": total_episodes,
            "world_size": self.world_size,
            "model_path": self.args.model_path,
            "num_history": self.args.num_history,
        }
        
        # Add model-specific extras
        summary.update(self.get_summary_extras())
        
        # Try to get SwanLab run URL: active session first, then train_metadata.json
        swanlab_url = get_swanlab_url()
        if not swanlab_url:
            swanlab_url = get_swanlab_url_from_train_metadata(self.args.model_path)
        if swanlab_url:
            summary["swanlab_url"] = swanlab_url
        
        # For SatNav, compute per-trajectory_type statistics
        if self.args.env_type == "satnav":
            trajectory_type_stats = compute_trajectory_type_stats(all_results_merged)
            if trajectory_type_stats:
                summary["by_trajectory_type"] = trajectory_type_stats
                reference_distribution = load_satnav_reference_distribution(self.resolve_config_path())
                if reference_distribution:
                    weighted_metrics = compute_weighted_trajectory_type_metrics(
                        trajectory_type_stats,
                        reference_distribution,
                    )
                    if weighted_metrics:
                        summary["weighted_by_seen_unseen_distribution"] = weighted_metrics
        
        print(f"\n" + "="*60)
        print(f"{self.model_description} Evaluation Summary ({self.args.eval_split})")
        print(f"="*60)
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Mean SPL: {summary['mean_spl']:.4f}")
        print(f"Oracle Success: {summary['oracle_success']:.2%}")
        print(f"Navigation Error: {summary['navigation_error']:.2f}m")
        print(f"Average Steps: {summary['avg_steps']:.2f}")
        
        # Print per-trajectory_type stats for SatNav
        if self.args.env_type == "satnav" and "by_trajectory_type" in summary:
            print(f"\n--- By Trajectory Type ---")
            for ttype, tstats in summary["by_trajectory_type"].items():
                print(f"  [{ttype}] SR: {tstats['success_rate']:.2%}, SPL: {tstats['mean_spl']:.4f}, "
                      f"OS: {tstats['oracle_success']:.2%}, NE: {tstats['navigation_error']:.2f}m, "
                      f"Steps: {tstats['avg_steps']:.2f}, N: {tstats['total_episodes']}")
            weighted_stats = summary.get("weighted_by_seen_unseen_distribution")
            if weighted_stats:
                print(f"\n--- Weighted (val_seen+val_unseen) ---")
                print(f"  SR: {weighted_stats['success_rate']:.2%}, SPL: {weighted_stats['mean_spl']:.4f}, "
                      f"OS: {weighted_stats['oracle_success']:.2%}, NE: {weighted_stats['navigation_error']:.2f}m, "
                      f"Steps: {weighted_stats['avg_steps']:.2f}")
        
        if self.uses_compression and hasattr(self.args, 'compress_stride'):
            print(f"\nCompression: stride={self.args.compress_stride} ({self.args.compress_stride**2}x)")
        
        print(f"Total Episodes: {total_episodes}")
        if self.world_size > 1:
            print(f"Distributed: {self.world_size} GPUs")
        if swanlab_url:
            print(f"SwanLab URL: {swanlab_url}")
        print(f"="*60)
        
        with open(os.path.join(self.args.output_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Clean all_results: remove timing data, save one result per line (JSONL format)
        cleaned_results = clean_results_for_output(all_results_merged)
        # Sort by episode_id descending before saving
        cleaned_results = sorted(cleaned_results, key=lambda x: int(x.get('episode_id', 0)), reverse=True)
        with open(os.path.join(self.args.output_dir, "all_results.jsonl"), "w") as f:
            for result in cleaned_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # Save timing statistics
        if len(timing_stats_list) > 0:
            timing_summary_path = save_timing_stats(self.args.output_dir, timing_stats_list)
            print(f"Timing statistics summary saved to {timing_summary_path}")
        
        print(f"Results saved to {self.args.output_dir}")
        
        # Video compression
        if self.args.video_compression and self.args.save_video:
            try:
                from swiftvln.common import compress_videos
                compress_videos(self.args.output_dir, cleaned_results, chunk_size=400)
            except ImportError:
                pass
    
    def run(self):
        """Main evaluation entry point."""
        # Parse arguments
        parser = self.create_parser()
        self.args = parser.parse_args()
        
        # Set random seeds for reproducibility
        self.set_seed(DEFAULT_EVAL_SEED)
        
        # Initialize distributed
        if self.args.distributed:
            self.rank, self.world_size, self.local_rank = self.init_distributed()
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0
        
        self.is_main = (self.rank == 0)
        
        if self.is_main:
            os.makedirs(self.args.output_dir, exist_ok=True)
            if self.world_size > 1:
                print(f"[Distributed Mode] {self.world_size} processes")
            else:
                print("[Single Process Mode]")
            if self.uses_compression and hasattr(self.args, 'compress_stride'):
                print(f"[{self.model_description}] compress_stride={self.args.compress_stride}")
        
        if self.world_size > 1:
            torch.distributed.barrier()
        
        # Register model module
        self.register_module()
        
        # Load model and template
        if self.is_main:
            print(f"Loading model from {self.args.model_path}...")
        
        model, processor = self.load_model()
        template = self.load_template(processor)
        self.configure_template(template)
        self.initialize_model(model)
        
        # Create evaluator
        config_path = self.resolve_config_path()
        evaluator = self.create_evaluator(config_path, model, processor, template)
        
        # Setup environment
        env_wrapper = evaluator.config_env()
        all_episodes = self.get_episodes(env_wrapper)
        my_episodes, scene_episode_dict = self.distribute_episodes(all_episodes)
        
        if self.is_main:
            print(f"Environment: {self.args.eval_split}, Total: {len(all_episodes)}, "
                  f"This process: {len(my_episodes)}, Scenes: {len(scene_episode_dict)}")
        
        # Evaluation loop
        results = []
        timing_stats_list = []
        
        if self.world_size > 1:
            desc = f"Rank 0 ({len(my_episodes)} eps, {self.world_size} GPUs total)"
        else:
            desc = f"Evaluating ({self.model_description})"
        pbar = tqdm.tqdm(my_episodes, desc=desc, disable=not self.is_main)
        
        for i, episode in enumerate(pbar):
            result = self.evaluate_episode(evaluator, env_wrapper, episode)
            results.append(result)
            
            # Collect timing stats
            if self.rank == 0 and '_timing_stats' in result:
                timing_stats_list.append(result)
            
            # Save partial results
            if self.is_main and (i + 1) % 10 == 0:
                with open(os.path.join(self.args.output_dir, "results_partial.json"), "w") as f:
                    json.dump(results, f, indent=2)
        
        env_wrapper.close()
        
        # Gather and save results
        sucs_all, spls_all, oss_all, nes_all, all_results_merged = self.gather_results(results)
        
        if self.is_main:
            self.save_results(sucs_all, spls_all, oss_all, nes_all, all_results_merged, timing_stats_list)
        
        # Cleanup
        if self.world_size > 1:
            torch.distributed.destroy_process_group()
