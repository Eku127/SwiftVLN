"""
Evaluate StreamVLN on SatNav task.

Design:
  - Uses SatNavEnvWrapper (from SwiftVLN) for unified environment interface
  - Keeps StreamVLN's original KV-cache streaming + window reset logic
  - No depth / pose / intrinsics (not available in SatNav)
  - Supports save_video, trajectory_type statistics, resume from partial results

Run via scripts/eval_satnav.sh (torchrun multi-GPU or single GPU).
Requires: streamvln-baseline conda env (satnav installed via pip install -e /path/to/SatNav).
"""

import sys
import os

# ---------------------------------------------------------------
# sys.path setup — make StreamVLN packages importable
# ---------------------------------------------------------------
_STREAMVLN_ROOT = "/mnt/data1/home/jiangjiajun/workspace/StreamVLN"
if _STREAMVLN_ROOT not in sys.path:
    sys.path.insert(0, _STREAMVLN_ROOT)

_STREAMVLN_SUBPKG = os.path.join(_STREAMVLN_ROOT, "streamvln")
if _STREAMVLN_SUBPKG not in sys.path:
    sys.path.insert(0, _STREAMVLN_SUBPKG)

_BASELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_ROOT not in sys.path:
    sys.path.insert(0, _BASELINE_ROOT)

_REPO_ROOT = os.path.abspath(os.path.join(_BASELINE_ROOT, "..", "..", ".."))
_SWIFTVLN_SRC = os.path.join(_REPO_ROOT, "src")
if _SWIFTVLN_SRC not in sys.path:
    sys.path.insert(0, _SWIFTVLN_SRC)


import re
import copy
import json
import time
import random
import argparse
import itertools

import tqdm
import torch
import numpy as np
import transformers
import torch.distributed as dist
from PIL import Image, ImageFile
from collections import OrderedDict
from omegaconf import OmegaConf

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import (
    dict_to_cuda,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_MEMORY_TOKEN,
    MEMORY_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN,
)
from utils.dist import (
    init_distributed_mode,
    get_rank,
    get_world_size,
)

# SatNav environment
from satnav.core.env import Env as SatNavEnv
from satnav.dataset.satnav_dataset import SatNavDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fixed NE penalty for episodes that fail with runtime errors during evaluation.
ERROR_NE_PENALTY = 500.0


# =====================================================================
# SatNav Environment Wrapper
# (Inlined from SwiftVLN/src/swiftvln/common/env/satnav.py to avoid
#  importing swiftvln.common which depends on ms-swift)
# =====================================================================

class SatNavEnvWrapper:
    """Wrapper for SatNav VLN environment.

    Adapts satnav.core.env.Env to a unified interface for evaluation.
    """

    def __init__(self, satnav_env: SatNavEnv, config=None):
        self._env = satnav_env
        self._current_episode = None
        self._last_info = {}
        self._config = config

    def reset(self, episode) -> dict:
        self._current_episode = episode
        observations = self._env.reset_to_episode(episode)
        return observations

    def step(self, action: int):
        observations, done, info = self._env.step(action)
        self._last_info = info
        return observations, done

    def get_last_step_info(self) -> dict:
        return self._last_info

    def get_agent_state(self):
        return self._env._task._sim.get_agent_state()

    @property
    def config(self):
        return self._config

    def get_metrics(self) -> dict:
        return self._env.get_metrics()

    def get_rgb(self, obs: dict) -> np.ndarray:
        return obs["rgb"]

    def get_instruction(self, episode) -> str:
        instruction = episode.instruction
        if isinstance(instruction, dict):
            return instruction.get("text", instruction.get("instruction_text", ""))
        elif hasattr(instruction, "text"):
            return instruction.text
        elif hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        return str(instruction)

    @property
    def max_steps(self) -> int:
        return self._env.max_episode_steps

    @property
    def episode_over(self) -> bool:
        return self._env.episode_over

    def close(self):
        pass

    @property
    def env(self) -> SatNavEnv:
        return self._env


# =====================================================================
# Evaluator
# =====================================================================

class SatNavVLNEvaluator:
    """
    StreamVLN evaluator for SatNav environment.

    Mirrors the structure of VLNEvaluator in streamvln_eval.py but:
     - Uses SatNavEnvWrapper for unified env interface
     - Does NOT use depth / pose / intrinsics
     - Step size: 10 meters (vs 25 cm in R2R)
     - Supports save_video, trajectory_type statistics
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.device = torch.device("cuda")
        self.model = model
        self.tokenizer = tokenizer

        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

        self.image_processor = model.get_vision_tower().image_processor

        # SatNav navigation prompt (same as training)
        self.prompt_template = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 10 meters, or STOP."
        )
        self.conversation = [
            {"from": "human", "value": self.prompt_template},
            {"from": "gpt", "value": ""},
        ]

        # Action symbol → index mapping (same as streamvln_eval.py)
        self.actions2idx = OrderedDict({
            "STOP": [0],
            "↑": [1],
            "←": [2],
            "→": [3],
        })

        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert numpy RGB (H, W, 3) to model input tensor."""
        image = Image.fromarray(rgb).convert("RGB")
        return self.image_processor.preprocess(
            images=image, return_tensors="pt"
        )["pixel_values"][0]

    # ------------------------------------------------------------------
    # Prompt building (same logic as VLNEvaluator.preprocess_qwen)
    # ------------------------------------------------------------------

    def preprocess_qwen(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        max_len: int = 2048,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False,
    ):
        """Tokenize conversation sources for Qwen-style chat template."""
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids[:2]

        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] "
            "+ '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        tokenizer.chat_template = chat_template

        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_message}]
                )

            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except KeyError:
                    role = conv["from"]
                    content = conv["value"]
                role = roles.get(role, role)
                conv_msg = [{"role": role, "content": content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv_msg)
                input_id += encode_id

            for idx_tok, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx_tok] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx_tok] = MEMORY_TOKEN_INDEX

            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    # ------------------------------------------------------------------
    # Action parsing (identical to streamvln_eval.py)
    # ------------------------------------------------------------------

    def parse_actions(self, output: str):
        """Parse action symbols from model output string."""
        action_patterns = "|".join(
            re.escape(a) for a in self.actions2idx
        )
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[m] for m in matches]
        return list(itertools.chain.from_iterable(actions))

    # ------------------------------------------------------------------
    # Instruction extraction
    # ------------------------------------------------------------------

    @staticmethod
    def get_instruction(episode) -> str:
        """Extract instruction text from episode."""
        instruction = getattr(episode, "instruction", "")
        if isinstance(instruction, dict):
            return instruction.get("text", instruction.get("instruction_text", ""))
        elif hasattr(instruction, "text"):
            return instruction.text
        elif hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        return str(instruction)

    # ------------------------------------------------------------------
    # Single episode evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_episode(self, env_wrapper: SatNavEnvWrapper, episode, env_idx: int = 0):
        """
        Evaluate a single episode using StreamVLN's KV-cache streaming.

        This follows the exact same logic as streamvln_eval.py:
        1. Build prompt with instruction + current observation
        2. Generate action sequence using KV-cache
        3. Execute actions until action queue is empty or episode ends
        4. On window boundary (step % num_frames == 0), reset KV-cache

        Args:
            env_wrapper: SatNavEnvWrapper instance
            episode: VLNEpisode to evaluate
            env_idx: Environment index for multi-env model state

        Returns:
            Dict of metrics: success, spl, distance_to_goal, oracle_success, steps, etc.
        """
        self.model.eval()
        self.model.reset_for_env(env_idx)

        observations = env_wrapper.reset(episode)
        instruction = self.get_instruction(episode)

        rgb_list = []
        time_ids = []
        action_seq = []
        past_key_values = None
        output_ids = None
        step_id = 0

        max_steps = env_wrapper.max_steps

        while not env_wrapper.episode_over and step_id < max_steps:
            time_ids.append(step_id)
            rgb = env_wrapper.get_rgb(observations)
            image_tensor = self.process_rgb(rgb)
            rgb_list.append(image_tensor)

            if len(action_seq) == 0:
                # ---- Build prompt ----
                if output_ids is None:
                    # First prediction in this window
                    sources = copy.deepcopy(self.conversation)
                    sources[0]["value"] = sources[0]["value"].replace(
                        DEFAULT_VIDEO_TOKEN + "\n", ""
                    )
                    if step_id != 0:
                        sources[0]["value"] += (
                            f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
                        )
                    sources[0]["value"] = sources[0]["value"].replace(
                        "<instruction>.", instruction
                    )
                    add_system = True
                else:
                    # Subsequent prediction in the same window — append to existing context
                    sources = [
                        {"from": "human", "value": ""},
                        {"from": "gpt", "value": ""},
                    ]
                    add_system = False

                input_ids, _ = self.preprocess_qwen(
                    [sources], self.tokenizer, True, add_system=add_system
                )
                if output_ids is not None:
                    input_ids = torch.cat(
                        [output_ids, input_ids.to(output_ids.device)], dim=1
                    )

                # ---- Build image list: [history] + [current] ----
                # Use the same condition as memory-token insertion: first generate
                # call in a new window (output_ids is None) and not the very first
                # step. This fixes a crash when leftover actions from the previous
                # window carry execution past the boundary (step_id % num_frames != 0)
                # while output_ids is still None from the reset — in that case the
                # memory token was added to the prompt but history images were not
                # provided, causing memory_features[b] == None and a TypeError.
                images = rgb_list[-1:]
                if output_ids is None and step_id != 0:
                    cur_step = time_ids[0]  # == step_id (just appended above)
                    if self.num_history is None:
                        history_ids = slice(0, cur_step, self.num_future_steps)
                    else:
                        history_ids = slice(
                            0,
                            cur_step,
                            max(cur_step // self.num_history, 1),
                        )
                    images = rgb_list[history_ids] + images

                images_tensor = torch.stack(images).unsqueeze(0)  # [1, T, C, H, W]

                # ---- Forward (no depth/pose/intrinsics) ----
                input_dict = {
                    "images": images_tensor,
                    "depths": None,
                    "poses": None,
                    "intrinsics": None,
                    "inputs": input_ids,
                    "env_id": env_idx,
                    "time_ids": [time_ids],
                    "task_type": [0],
                }
                input_dict = dict_to_cuda(input_dict, self.device)
                if input_dict["images"] is not None:
                    input_dict["images"] = input_dict["images"].to(torch.bfloat16)

                outputs = self.model.generate(
                    **input_dict,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=10000,
                    use_cache=True,
                    return_dict_in_generate=True,
                    past_key_values=past_key_values,
                )

                output_ids = outputs.sequences
                past_key_values = outputs.past_key_values
                llm_outputs = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=False
                )[0].strip()
                action_seq = self.parse_actions(llm_outputs)
                if len(action_seq) == 0:
                    action_seq = [0]  # default STOP

            # ---- Execute action ----
            action = action_seq.pop(0)
            observations, _ = env_wrapper.step(action)
            step_id += 1

            # ---- Window reset (same as streamvln_eval.py) ----
            if step_id % self.num_frames == 0:
                self.model.reset_for_env(env_idx)
                output_ids = None
                past_key_values = None
                time_ids = []

        metrics = env_wrapper.get_metrics()
        metrics["_step_count"] = step_id
        return metrics


# =====================================================================
# Environment setup
# =====================================================================

def create_satnav_env(config_path: str, eval_split: str) -> SatNavEnvWrapper:
    """Create SatNav environment wrapped in SatNavEnvWrapper.

    Uses OmegaConf to load config (consistent with OverlapVLN evaluator),
    then creates SatNavDataset + SatNavEnv + SatNavEnvWrapper.
    """
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = eval_split
    OmegaConf.set_struct(config, True)

    dataset = SatNavDataset(config.DATASET)
    satnav_env = SatNavEnv(config, dataset=dataset, cycle=False)
    return SatNavEnvWrapper(satnav_env, config=config)


# =====================================================================
# Main evaluation loop
# =====================================================================

def evaluate(model, tokenizer, args):
    """Run evaluation on all episodes (distributed-aware)."""
    model.eval()
    world_size = get_world_size()
    rank = get_rank()
    is_main = (rank == 0)

    model.reset(world_size)

    evaluator = SatNavVLNEvaluator(args=args, model=model, tokenizer=tokenizer)

    # Create environment
    env_wrapper = create_satnav_env(args.satnav_config_path, args.eval_split)

    # Get all episodes
    all_episodes = env_wrapper.env._dataset.episodes
    if args.max_episodes is not None:
        all_episodes = all_episodes[:args.max_episodes]

    # Distribute episodes by scene
    scene_episode_dict = {}
    for episode in all_episodes:
        scene_id = getattr(episode, "scene_id", "default")
        if scene_id not in scene_episode_dict:
            scene_episode_dict[scene_id] = []
        scene_episode_dict[scene_id].append(episode)

    my_episodes = []
    for scene_id in sorted(scene_episode_dict.keys()):
        scene_episodes = scene_episode_dict[scene_id]
        my_episodes.extend(scene_episodes[rank::world_size])

    if is_main:
        print(f"[Eval] split={args.eval_split}, total={len(all_episodes)}, "
              f"this_rank={len(my_episodes)}, scenes={len(scene_episode_dict)}, "
              f"world_size={world_size}")

    # Resume: load already-done results
    result_file = os.path.join(args.output_path, "result.jsonl")
    existing_results = load_dedup_results(result_file)
    done_ids = {
        build_episode_key(r.get("episode_id", ""), r.get("scene_id", ""))
        for r in existing_results
    }
    done_ids.discard("")
    results = []
    if existing_results and is_main:
        print(f"[Resume] Loaded {len(done_ids)} done episodes from {result_file}")

    # Evaluation loop
    pbar = tqdm.tqdm(
        my_episodes,
        desc=f"Rank {rank}" if world_size > 1 else "Evaluating StreamVLN",
        disable=not is_main,
    )

    for episode in pbar:
        ep_id = str(episode.episode_id)
        scene_id = getattr(episode, "scene_id", "unknown")
        ep_key = build_episode_key(ep_id, scene_id)
        if ep_key in done_ids:
            continue

        instruction = evaluator.get_instruction(episode)
        trajectory_type = getattr(episode, "trajectory_type", None)

        try:
            metrics = evaluator.eval_episode(env_wrapper, episode, env_idx=rank)

            result = {
                "episode_id": ep_id,
                "scene_id": scene_id,
                "success": float(metrics.get("success", 0)),
                "spl": float(metrics.get("spl", 0)),
                "oracle_success": float(metrics.get("oracle_success", 0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0)),
                "steps": int(metrics.get("_step_count", 0)),
                "instruction": instruction[:200],
            }
            if trajectory_type is not None:
                result["trajectory_type"] = trajectory_type

        except Exception as e:
            import traceback
            print(f"[Rank {rank}] Error on episode {ep_id}: {e}")
            traceback.print_exc()
            result = {
                "episode_id": ep_id,
                "scene_id": scene_id,
                "success": 0.0,
                "spl": 0.0,
                "oracle_success": 0.0,
                "distance_to_goal": ERROR_NE_PENALTY,
                "steps": 0,
                "instruction": instruction[:200],
                "error": str(e),
            }
            if trajectory_type is not None:
                result["trajectory_type"] = trajectory_type

        results.append(result)
        done_ids.add(ep_key)

        # Append result immediately (for resume support)
        with open(result_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Update progress bar
        recent = results[-min(20, len(results)):]
        avg_sr = np.mean([r["success"] for r in recent])
        avg_ne = np.mean([r["distance_to_goal"] for r in recent if r["distance_to_goal"] < 1e6])
        pbar.set_postfix(SR=f"{avg_sr:.2%}", NE=f"{avg_ne:.1f}m", done=len(results))

    env_wrapper.close()

    # ---- Gather and summarize results ----
    if world_size > 1:
        aggregate_distributed(result_file, args.output_path, args)
    else:
        all_results = load_dedup_results(result_file)
        save_summary(all_results, args.output_path, args)


def build_episode_key(episode_id, scene_id):
    ep_id = str(episode_id) if episode_id is not None else ""
    scene = str(scene_id) if scene_id is not None else ""
    if ep_id == "":
        return ""
    if scene == "":
        return ep_id
    return f"{scene}::{ep_id}"


def load_dedup_results(result_file):
    """Load jsonl results and keep the latest record per scene_id+episode_id."""
    if not os.path.exists(result_file):
        return []

    results_by_ep = {}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep_key = build_episode_key(
                result.get("episode_id", ""),
                result.get("scene_id", ""),
            )
            if not ep_key:
                continue
            results_by_ep[ep_key] = result
    return list(results_by_ep.values())


def aggregate_distributed(result_file, output_path, args):
    """Wait all ranks, then summarize from deduplicated result.jsonl on rank 0."""
    rank = get_rank()
    dist.barrier()
    if rank == 0:
        all_results = load_dedup_results(result_file)
        save_summary(all_results, output_path, args)
    dist.barrier()


def save_summary(results, output_path, args):
    """Save evaluation summary for single-process mode."""
    if not results:
        print("[Warning] No results to summarize.")
        return

    sucs = [r["success"] for r in results]
    spls = [r["spl"] for r in results]
    oss = [r["oracle_success"] for r in results]
    nes = [r["distance_to_goal"] for r in results if r["distance_to_goal"] < 1e6]
    steps = [r["steps"] for r in results]

    total = len(results)
    summary = {
        "eval_split": args.eval_split,
        "SR": np.mean(sucs) if sucs else 0,
        "SPL": np.mean(spls) if spls else 0,
        "OS": np.mean(oss) if oss else 0,
        "NE": np.mean(nes) if nes else 0,
        "avg_steps": np.mean(steps) if steps else 0,
        "total_episodes": total,
        "model_path": args.model_path,
        "num_frames": args.num_frames,
        "num_history": args.num_history,
        "num_future_steps": args.num_future_steps,
    }

    # Per-trajectory_type statistics
    type_stats = {}
    for r in results:
        ttype = r.get("trajectory_type", "unknown")
        if ttype not in type_stats:
            type_stats[ttype] = {"sucs": [], "spls": [], "oss": [], "nes": [], "steps": []}
        type_stats[ttype]["sucs"].append(r["success"])
        type_stats[ttype]["spls"].append(r["spl"])
        type_stats[ttype]["oss"].append(r["oracle_success"])
        if r["distance_to_goal"] < 1e6:
            type_stats[ttype]["nes"].append(r["distance_to_goal"])
        type_stats[ttype]["steps"].append(r["steps"])

    if len(type_stats) > 1 or "unknown" not in type_stats:
        summary["by_trajectory_type"] = {}
        for ttype, ts in sorted(type_stats.items()):
            summary["by_trajectory_type"][ttype] = {
                "SR": np.mean(ts["sucs"]),
                "SPL": np.mean(ts["spls"]),
                "OS": np.mean(ts["oss"]),
                "NE": np.mean(ts["nes"]) if ts["nes"] else 0,
                "avg_steps": np.mean(ts["steps"]),
                "count": len(ts["sucs"]),
            }
    print("\n" + "=" * 60)
    print(f"StreamVLN SatNav Evaluation Summary ({args.eval_split})")
    print(f"=" * 60)
    print(f"Success Rate: {summary['SR']:.2%}")
    print(f"SPL:          {summary['SPL']:.4f}")
    print(f"Oracle Succ:  {summary['OS']:.2%}")
    print(f"Nav Error:    {summary['NE']:.2f}m")
    print(f"Avg Steps:    {summary['avg_steps']:.2f}")
    print(f"Total:        {summary['total_episodes']}")

    if "by_trajectory_type" in summary:
        print(f"\n--- By Trajectory Type ---")
        for ttype, ts in summary["by_trajectory_type"].items():
            print(f"  [{ttype}] SR: {ts['SR']:.2%}, SPL: {ts['SPL']:.4f}, "
                  f"OS: {ts['OS']:.2%}, NE: {ts['NE']:.2f}m, "
                  f"Steps: {ts['avg_steps']:.2f}, N: {ts['count']}")

    print(f"=" * 60)

    with open(os.path.join(output_path, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


# =====================================================================
# Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="StreamVLN SatNav Evaluation")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned StreamVLN checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--satnav_config_path", type=str, required=True,
                        help="Path to SatNav task config yaml")
    parser.add_argument("--eval_split", type=str, default="val_seen",
                        choices=["train", "val_seen", "val_unseen", "test"])
    parser.add_argument("--output_path", type=str, default="./results/satnav/streamvln")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=32768)
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Cap total episodes before distributed sharding (for debugging)")
    # Distributed args
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="12345")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Initialize distributed
    init_distributed_mode(args)

    # ---- Tokenizer ----
    tok_path = args.tokenizer_path or args.model_path
    # If no tokenizer in model_path, fall back to LLaVA-Video-7B-Qwen2
    if not os.path.exists(os.path.join(tok_path, "tokenizer_config.json")):
        tok_path = "lmms-lab/LLaVA-Video-7B-Qwen2"
        if get_rank() == 0:
            print(f"[INFO] No tokenizer in model_path, using: {tok_path}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tok_path,
        model_max_length=args.model_max_length,
        padding_side="right",
    )

    # ---- Model ----
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = args.num_history
    model.requires_grad_(False)

    vision_tower = model.get_vision_tower()
    if not getattr(vision_tower, "is_loaded", False):
        if get_rank() == 0:
            print(f"[INFO] Vision tower not loaded (delay_load). Loading from: {vision_tower.vision_tower_name}")
        vision_tower.load_model()
    vision_tower.to(device=args.local_rank, dtype=torch.bfloat16)

    model.to(args.local_rank)

    os.makedirs(args.output_path, exist_ok=True)
    evaluate(model, tokenizer, args)


if __name__ == "__main__":
    main()
