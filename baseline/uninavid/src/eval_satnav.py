"""
Evaluate Uni-NaVid fine-tuned model on SatNav task.

Design:
  - Uses load_pretrained_model from Uni-NaVid for model loading
  - Mirrors the token-injection inference logic in offline_eval_uninavid.py
  - Online eval strategy: predict 4 actions per query; model maintains internal
    navigation-feature cache across query calls (incremental frame processing)
  - No KV-cache streaming (unlike StreamVLN)
  - Actions: text-based "forward / left / right / stop"
  - Supports multi-GPU distributed eval (per-rank independent episodes)
  - Supports resume from partial result.jsonl

Run via:
  bash scripts/eval_satnav.sh <exp_name_or_checkpoint_path> [split] [gpus] [max_episodes]

Environment: conda env uninavid-baseline (satnav installed via pip install -e /path/to/SatNav)
"""

import sys
import os

# ---------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------
_UNINAVID_ROOT = "/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid"
if _UNINAVID_ROOT not in sys.path:
    sys.path.insert(0, _UNINAVID_ROOT)

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

import re
import json
import argparse

import tqdm
import torch
import numpy as np
import torch.distributed as dist
from PIL import Image, ImageFile
from omegaconf import OmegaConf

from uninavid.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from uninavid.model.builder import load_pretrained_model

# ------------------------------------------------------------------
# Flash-attention patch (mirrors train_satnav.py)
# Inject use_flash_attention_2=True into LlavaLlamaAttForCausalLM.from_pretrained
# so that load_pretrained_model (builder.py) picks up LlamaFlashAttention2.
# ------------------------------------------------------------------
from uninavid.model.language_model.llava_llama_vid import LlavaLlamaAttForCausalLM as _LlavaModel

_orig_from_pretrained = _LlavaModel.from_pretrained

@classmethod
def _flash_attn_from_pretrained(cls, *args, **kwargs):
    kwargs.setdefault("use_flash_attention_2", True)
    return _orig_from_pretrained(*args, **kwargs)

_LlavaModel.from_pretrained = _flash_attn_from_pretrained
# ------------------------------------------------------------------
from uninavid.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    NAVIGATION_IDENTIFIER,
    VIDEO_START_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    NAVIGATION_SPECIAL_TOKEN,
    IAMGE_SEPARATOR,
)
from uninavid.conversation import conv_templates, SeparatorStyle

from satnav.core.env import Env as SatNavEnv
from satnav.dataset.satnav_dataset import SatNavDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fixed NE penalty for episodes that error during evaluation.
ERROR_NE_PENALTY = 500.0

# Action text → SatNav action index
ACTION_TEXT_MAP = {"forward": 1, "left": 2, "right": 3, "stop": 0}


# =====================================================================
# SatNav Environment Wrapper
# (Inlined from SwiftVLN common/env/satnav.py to avoid importing
#  swiftvln which depends on ms-swift)
# =====================================================================


class SatNavEnvWrapper:
    """Wraps satnav.core.env.Env to a unified eval interface."""

    def __init__(self, satnav_env: SatNavEnv, config=None):
        self._env = satnav_env
        self._last_info = {}
        self._config = config

    def reset(self, episode) -> dict:
        return self._env.reset_to_episode(episode)

    def step(self, action: int):
        observations, done, info = self._env.step(action)
        self._last_info = info
        return observations, done

    def get_last_step_info(self) -> dict:
        return self._last_info

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


class UniNaVidEvaluator:
    """
    Online evaluator for Uni-NaVid on SatNav.

    Mirrors offline_eval_uninavid.py's inference logic but runs online
    against the SatNav simulator instead of replaying pre-recorded images.

    Query strategy:
      - Predict 4 actions per model call.
      - After executing all 4 actions, collect new observations and call model again.
      - Model internally caches navigation features across calls via
        initialize_online_inference_nav_feat_cache() + new_frames attribute.
    """

    def __init__(self, args, model, tokenizer, image_processor):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given {nav_id} {img_token}. "
            "Your assigned task is: '{instruction}'. "
            "Analyze this series of images to determine your next four actions. "
            "The predicted action should be one of the following: "
            "forward, left, right, or stop."
        )

        # Pre-tokenize navigation special tokens once (same as offline_eval_uninavid.py)
        device = next(model.parameters()).device

        def _tok(text):
            return tokenizer(text, return_tensors="pt").input_ids[0][1:].to(device)

        self._tok_video_start = _tok(VIDEO_START_SPECIAL_TOKEN)
        self._tok_video_end = _tok(VIDEO_END_SPECIAL_TOKEN)
        self._tok_image_start = _tok(IMAGE_START_TOKEN)
        self._tok_image_end = _tok(IMAGE_END_TOKEN)
        self._tok_nav = _tok(NAVIGATION_SPECIAL_TOKEN)
        self._tok_img_sep = _tok(IAMGE_SEPARATOR)

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def process_images(self, rgb_list: list) -> list:
        """Convert list of numpy RGB frames to model image input.

        Sets model.get_model().new_frames = len(rgb_list) so the model
        knows how many frames are NEW vs already cached.
        """
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = (
            self.image_processor
            .preprocess(batch_image, return_tensors="pt")["pixel_values"]
            .half()
            .to(next(self.model.parameters()).device)
        )
        return [video]

    # ------------------------------------------------------------------
    # Inference (mirrors offline_eval_uninavid.py predict_inference)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_inference(self, instruction: str, rgb_list: list) -> str:
        """Run one forward pass and return the raw output text.

        Args:
            instruction: Navigation instruction string.
            rgb_list: List of numpy RGB frames (newly accumulated since last call).

        Returns:
            Model output text (e.g. "forward forward left stop").
        """
        device = next(self.model.parameters()).device

        prompt = self.prompt_template.format(
            nav_id=NAVIGATION_IDENTIFIER,
            img_token=DEFAULT_IMAGE_TOKEN,
            instruction=instruction,
        )
        # Clean question (strip <image> for update_prompt)
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")

        # Prepend image token in the expected form
        if self.model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt.replace("<image>", "")
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt.replace("<image>", "")

        # Use "vicuna_v1" — same as offline_eval_uninavid.py (conv_mode = "vicuna_v1")
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(device)

        # Inject navigation special tokens around each IMAGE_TOKEN_INDEX (-200).
        # Token injection order matches offline_eval_uninavid.py exactly:
        #   video_start | image_sep | <image> (-200) | video_end |
        #   image_start | image_end | navigation
        indices = torch.where(token_prompt == -200)[0]
        parts = []
        while indices.numel() > 0:
            idx = indices[0].item()
            parts.append(token_prompt[:idx])
            parts.append(self._tok_video_start)
            parts.append(self._tok_img_sep)
            parts.append(token_prompt[idx : idx + 1])
            parts.append(self._tok_video_end)
            parts.append(self._tok_image_start)
            parts.append(self._tok_image_end)
            parts.append(self._tok_nav)
            token_prompt = token_prompt[idx + 1 :]
            indices = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            parts.append(token_prompt)
        input_ids = torch.cat(parts, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        imgs = self.process_images(rgb_list)

        self.model.update_prompt([[question]])
        output_ids = self.model.generate(
            input_ids,
            images=imgs,
            do_sample=True,
            temperature=0.5,
            max_new_tokens=1024,   # same as offline_eval_uninavid.py
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        input_token_len = input_ids.shape[1]
        # Warn if input tokens appear in output (same check as offline_eval_uninavid.py)
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            print(f"[Warning] {n_diff} output_ids differ from input_ids")
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_actions(output: str) -> list:
        """Parse action words from model output text."""
        words = re.findall(r"\b(forward|left|right|stop)\b", output.lower())
        return [ACTION_TEXT_MAP[w] for w in words]

    # ------------------------------------------------------------------
    # Instruction extraction
    # ------------------------------------------------------------------

    @staticmethod
    def get_instruction(episode) -> str:
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

    def eval_episode(self, env_wrapper: SatNavEnvWrapper, episode) -> dict:
        """Evaluate one episode.

        Strategy:
          1. Initialize navigation feature cache (per episode).
          2. Get initial observation and add to rgb_list.
          3. When action queue is empty → call model with rgb_list (new frames).
             Clear rgb_list after call (model has cached features internally).
          4. Execute action from queue → collect new observation into rgb_list.
          5. Repeat until episode ends or max_steps reached.

        Args:
            env_wrapper: SatNavEnvWrapper
            episode: VLNEpisode from SatNav dataset

        Returns:
            Metrics dict with success, spl, oracle_success, distance_to_goal, _step_count.
        """
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

        obs = env_wrapper.reset(episode)
        instruction = self.get_instruction(episode)

        # Seed rgb_list with initial observation
        rgb_list = [env_wrapper.get_rgb(obs)]
        action_queue = []
        step_id = 0
        max_steps = env_wrapper.max_steps

        while not env_wrapper.episode_over and step_id < max_steps:
            if not action_queue:
                output = self.predict_inference(instruction, rgb_list)
                rgb_list = []  # model has processed and cached these frames

                action_queue = self.parse_actions(output)
                if not action_queue:
                    action_queue = [0]  # default STOP

            action = action_queue.pop(0)
            obs, _ = env_wrapper.step(action)
            step_id += 1

            # Collect new observation for next model call (skip if episode ended)
            if not env_wrapper.episode_over:
                rgb_list.append(env_wrapper.get_rgb(obs))

        metrics = env_wrapper.get_metrics()
        metrics["_step_count"] = step_id
        return metrics


# =====================================================================
# Environment factory
# =====================================================================


def create_satnav_env(config_path: str, eval_split: str) -> SatNavEnvWrapper:
    """Load SatNav config and create wrapped environment."""
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = eval_split
    OmegaConf.set_struct(config, True)

    dataset = SatNavDataset(config.DATASET)
    satnav_env = SatNavEnv(config, dataset=dataset, cycle=False)
    return SatNavEnvWrapper(satnav_env, config=config)


# =====================================================================
# Distributed helpers
# =====================================================================


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def init_distributed(args):
    """Initialize torch.distributed if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0


# =====================================================================
# Distributed result aggregation
# =====================================================================


def save_summary(results: list, output_path: str, args) -> None:
    """Compute and print metrics summary, then write to evaluation_summary.json."""
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
        "SR": float(np.mean(sucs)) if sucs else 0.0,
        "SPL": float(np.mean(spls)) if spls else 0.0,
        "OS": float(np.mean(oss)) if oss else 0.0,
        "NE": float(np.mean(nes)) if nes else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "total_episodes": total,
        "model_path": args.model_path,
    }

    # Per-trajectory_type breakdown
    type_stats: dict = {}
    for r in results:
        ttype = r.get("trajectory_type", "unknown")
        type_stats.setdefault(
            ttype, {"sucs": [], "spls": [], "oss": [], "nes": [], "steps": []}
        )
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
                "SR": float(np.mean(ts["sucs"])),
                "SPL": float(np.mean(ts["spls"])),
                "OS": float(np.mean(ts["oss"])),
                "NE": float(np.mean(ts["nes"])) if ts["nes"] else 0.0,
                "avg_steps": float(np.mean(ts["steps"])),
                "count": len(ts["sucs"]),
            }

    print("\n" + "=" * 60)
    print(f"Uni-NaVid SatNav Evaluation Summary ({args.eval_split})")
    print("=" * 60)
    print(f"Success Rate: {summary['SR']:.2%}")
    print(f"SPL:          {summary['SPL']:.4f}")
    print(f"Oracle Succ:  {summary['OS']:.2%}")
    print(f"Nav Error:    {summary['NE']:.2f}m")
    print(f"Avg Steps:    {summary['avg_steps']:.2f}")
    print(f"Total:        {summary['total_episodes']}")

    if "by_trajectory_type" in summary:
        print("\n--- By Trajectory Type ---")
        for ttype, ts in summary["by_trajectory_type"].items():
            print(
                f"  [{ttype}] SR: {ts['SR']:.2%}, SPL: {ts['SPL']:.4f}, "
                f"OS: {ts['OS']:.2%}, NE: {ts['NE']:.2f}m, "
                f"Steps: {ts['avg_steps']:.2f}, N: {ts['count']}"
            )

    print("=" * 60)

    summary_path = os.path.join(output_path, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


def aggregate_distributed(results: list, world_size: int, output_path: str, args) -> None:
    """Gather results from all ranks, then print/save summary on rank 0."""
    device = torch.device(f"cuda:{get_rank()}")
    rank = get_rank()

    sucs = torch.tensor([r["success"] for r in results], device=device)
    spls = torch.tensor([r["spl"] for r in results], device=device)
    oss = torch.tensor([r["oracle_success"] for r in results], device=device)
    nes = torch.tensor([r["distance_to_goal"] for r in results], device=device)
    ep_num = torch.tensor(len(results), device=device)

    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)

    sucs_all = [torch.zeros(ep_num_all[i].item(), dtype=sucs.dtype, device=device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i].item(), dtype=spls.dtype, device=device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i].item(), dtype=oss.dtype, device=device) for i in range(world_size)]
    nes_all = [torch.zeros(ep_num_all[i].item(), dtype=nes.dtype, device=device) for i in range(world_size)]

    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(nes_all, nes)
    dist.barrier()

    if rank == 0:
        sucs_cat = torch.cat(sucs_all).cpu().numpy()
        spls_cat = torch.cat(spls_all).cpu().numpy()
        oss_cat = torch.cat(oss_all).cpu().numpy()
        nes_cat = torch.cat(nes_all).cpu().numpy()
        total = len(sucs_cat)

        summary = {
            "eval_split": args.eval_split,
            "SR": float(sucs_cat.mean()),
            "SPL": float(spls_cat.mean()),
            "OS": float(oss_cat.mean()),
            "NE": float(nes_cat.mean()),
            "total_episodes": total,
            "model_path": args.model_path,
        }

        print("\n" + "=" * 60)
        print(f"Uni-NaVid SatNav Evaluation Summary ({args.eval_split})")
        print("=" * 60)
        print(f"Success Rate: {summary['SR']:.2%}")
        print(f"SPL:          {summary['SPL']:.4f}")
        print(f"Oracle Succ:  {summary['OS']:.2%}")
        print(f"Nav Error:    {summary['NE']:.2f}m")
        print(f"Total:        {summary['total_episodes']}")
        print("=" * 60)

        summary_path = os.path.join(output_path, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")


# =====================================================================
# Main evaluation loop
# =====================================================================


def evaluate(model, tokenizer, image_processor, args) -> None:
    """Distributed-aware evaluation loop."""
    model.eval()
    world_size = get_world_size()
    rank = get_rank()
    is_main = rank == 0

    evaluator = UniNaVidEvaluator(
        args=args, model=model, tokenizer=tokenizer, image_processor=image_processor
    )

    env_wrapper = create_satnav_env(args.satnav_config_path, args.eval_split)

    all_episodes = env_wrapper.env._dataset.episodes
    if args.max_episodes is not None:
        all_episodes = all_episodes[: args.max_episodes]

    # Distribute by scene (same approach as streamvln eval_satnav.py)
    scene_episode_dict: dict = {}
    for episode in all_episodes:
        scene_id = getattr(episode, "scene_id", "default")
        scene_episode_dict.setdefault(scene_id, []).append(episode)

    my_episodes = []
    for scene_id in sorted(scene_episode_dict.keys()):
        my_episodes.extend(scene_episode_dict[scene_id][rank::world_size])

    if is_main:
        print(
            f"[Eval] split={args.eval_split}, total={len(all_episodes)}, "
            f"this_rank={len(my_episodes)}, scenes={len(scene_episode_dict)}, "
            f"world_size={world_size}"
        )

    # Resume support
    result_file = os.path.join(args.output_path, "result.jsonl")
    done_ids: set = set()
    results = []
    if os.path.exists(result_file):
        with open(result_file, encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line)
                    done_ids.add(str(res.get("episode_id", "")))
                    results.append(res)
                except json.JSONDecodeError:
                    pass
        if is_main:
            print(f"[Resume] Loaded {len(done_ids)} done episodes from {result_file}")

    pbar = tqdm.tqdm(
        my_episodes,
        desc=f"Rank {rank}" if world_size > 1 else "Evaluating Uni-NaVid",
        disable=not is_main,
    )

    for episode in pbar:
        ep_id = str(episode.episode_id)
        if ep_id in done_ids:
            continue

        instruction = evaluator.get_instruction(episode)
        trajectory_type = getattr(episode, "trajectory_type", None)

        try:
            metrics = evaluator.eval_episode(env_wrapper, episode)
            result = {
                "episode_id": ep_id,
                "scene_id": getattr(episode, "scene_id", "unknown"),
                "success": float(metrics.get("success", 0)),
                "spl": float(metrics.get("spl", 0)),
                "oracle_success": float(metrics.get("oracle_success", 0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0)),
                "steps": int(metrics.get("_step_count", 0)),
                "instruction": instruction[:200],
            }
        except Exception as e:
            import traceback

            print(f"[Rank {rank}] Error on episode {ep_id}: {e}")
            traceback.print_exc()
            result = {
                "episode_id": ep_id,
                "scene_id": getattr(episode, "scene_id", "unknown"),
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
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")

        # Update progress bar with rolling average
        recent = results[-min(20, len(results)) :]
        avg_sr = np.mean([r["success"] for r in recent])
        avg_ne = np.mean(
            [r["distance_to_goal"] for r in recent if r["distance_to_goal"] < 1e6]
            or [0]
        )
        pbar.set_postfix(SR=f"{avg_sr:.2%}", NE=f"{avg_ne:.1f}m", done=len(results))

    env_wrapper.close()

    if world_size > 1:
        aggregate_distributed(results, world_size, args.output_path, args)
    else:
        save_summary(results, args.output_path, args)


# =====================================================================
# Entry point
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Uni-NaVid SatNav Evaluation")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned Uni-NaVid checkpoint",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Base model path (used by load_pretrained_model for adapter-only ckpts)",
    )
    parser.add_argument(
        "--satnav_config_path",
        type=str,
        required=True,
        help="Path to SatNav task config yaml",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val_unseen",
        choices=["train", "val_seen", "val_unseen", "test"],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/satnav/uninavid",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Cap number of episodes per rank (for quick smoke/debug runs)",
    )
    # Distributed args (populated by init_distributed)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    args = parser.parse_args()

    init_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}")

    # ---- Load model ----
    # load_pretrained_model dispatches on 'vid' in model_name to load
    # LlavaLlamaAttForCausalLM. Our checkpoint directories may not have
    # a "vid"-containing name, so we force a canonical name here.
    #
    # Use device_map={"": local_rank} to pin all layers to this rank's GPU.
    # device_map="auto" (the default) distributes layers across all visible
    # GPUs which breaks distributed eval where each rank owns exactly one GPU.
    model_name = "uninavid"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map={"": args.local_rank},
        device=f"cuda:{args.local_rank}",
    )

    model.eval()

    # ---- Verify flash attention (smoke check) ----
    attn_class = type(model.model.layers[0].self_attn).__name__
    print(f"[FlashAttn Check] attention class: {attn_class}")
    print(f"[FlashAttn Check] using LlamaFlashAttention2: {attn_class == 'LlamaFlashAttention2'}")

    os.makedirs(args.output_path, exist_ok=True)
    evaluate(model, tokenizer, image_processor, args)


if __name__ == "__main__":
    main()
