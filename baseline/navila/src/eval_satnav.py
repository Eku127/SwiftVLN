"""
Evaluate NaVILA fine-tuned model on SatNav task.

Design:
  - Keeps NaVILA's original online navigation prompt and action parsing logic
  - Adapts the evaluator to SatNav's environment / metrics interface
  - Supports distributed evaluation, resume from partial results, and summary export

Run via:
  bash scripts/eval_satnav.sh <exp_name_or_checkpoint_path> [split] [gpus] [max_episodes]

Environment: conda env navila-baseline
"""

import sys
import os

_NAVILA_ROOT = "/mnt/data1/home/jiangjiajun/workspace/NaVILA"
if _NAVILA_ROOT not in sys.path:
    sys.path.insert(0, _NAVILA_ROOT)

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

import copy
import json
import time
import argparse
import traceback
import re

import tqdm
import torch
import numpy as np
import torch.distributed as dist
from PIL import Image, ImageFile
from omegaconf import OmegaConf

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from satnav.core.env import Env as SatNavEnv
from satnav.dataset.satnav_dataset import SatNavDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

ERROR_NE_PENALTY = 500.0
QUEUE_DISTANCE_CHOICES = [25, 50, 75]
QUEUE_DEGREE_CHOICES = [15, 30, 45]

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    'of historical observations {history_tokens}, and current observation <image>\n. Your assigned task is: "{instruction}" '
    "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
    "degree, moving forward a certain distance, or stop if the task is completed."
)


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """Exact sampling logic from upstream NaVILA evaluator."""
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

    latest_frame = frames[-1]
    sampled_indices = np.linspace(
        0,
        len(frames) - 1,
        num=num_frames - 1,
        endpoint=False,
        dtype=int,
    )
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    return sampled_frames


class SatNavEnvWrapper:
    """Wrap satnav.core.env.Env to a baseline-friendly eval interface."""

    def __init__(self, satnav_env: SatNavEnv, config=None):
        self._env = satnav_env
        self._last_info = {}
        self._config = config

    def reset(self, episode):
        return self._env.reset_to_episode(episode)

    def step(self, action: int):
        observations, done, info = self._env.step(action)
        self._last_info = info
        return observations, done

    def get_metrics(self):
        return self._env.get_metrics()

    def get_rgb(self, obs) -> np.ndarray:
        return obs["rgb"]

    @property
    def max_steps(self) -> int:
        return self._env.max_episode_steps

    @property
    def episode_over(self) -> bool:
        return self._env.episode_over

    @property
    def env(self) -> SatNavEnv:
        return self._env

    def close(self):
        pass


class NaVILASatNavEvaluator:
    """NaVILA online evaluator on SatNav with upstream prompt + parsing."""

    def __init__(self, args, model, tokenizer, image_processor):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.patterns = {
            0: re.compile(r"\bstop\b", re.IGNORECASE),
            1: re.compile(r"\bis move forward\b", re.IGNORECASE),
            2: re.compile(r"\bis turn left\b", re.IGNORECASE),
            3: re.compile(r"\bis turn right\b", re.IGNORECASE),
        }

    @staticmethod
    def get_instruction(episode) -> str:
        instruction = getattr(episode, "instruction", "")
        if isinstance(instruction, dict):
            return instruction.get("text", instruction.get("instruction_text", ""))
        if hasattr(instruction, "text"):
            return instruction.text
        if hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        return str(instruction)

    def build_prompt(self, instruction: str, num_frames: int) -> str:
        history_tokens = "<image>\n" * max(num_frames - 1, 0)
        return PROMPT_TEMPLATE.format(
            history_tokens=history_tokens,
            instruction=instruction,
        )

    def predict_action_text(self, frames, instruction: str) -> str:
        question = self.build_prompt(instruction, len(frames))

        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(frames, self.image_processor, self.model.config).to(
            self.model.device,
            dtype=torch.float16,
        )
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor.half().to(self.model.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()

    def map_string_to_action(self, text: str):
        for action, pattern in self.patterns.items():
            if pattern.search(text):
                return action
        return None

    @staticmethod
    def _snap_value(value: int, base: int, choices: list[int]) -> int:
        if value <= 0:
            return base
        if value % base != 0:
            return min(choices, key=lambda x: abs(x - value))
        return value

    def parse_action_and_queue(self, output_text: str):
        action = self.map_string_to_action(output_text)
        if action is None:
            action = 1

        queue_actions = []
        if action == 1:
            match = re.search(r"move forward (\d+) cm", output_text, re.IGNORECASE)
            distance = int(match.group(1)) if match else 25
            distance = self._snap_value(distance, 25, QUEUE_DISTANCE_CHOICES)
            queue_actions.extend([1] * max(int(distance // 25) - 1, 0))
        elif action == 2:
            match = re.search(r"turn left (\d+) degree", output_text, re.IGNORECASE)
            degree = int(match.group(1)) if match else 15
            degree = self._snap_value(degree, 15, QUEUE_DEGREE_CHOICES)
            queue_actions.extend([2] * max(int(degree // 15) - 1, 0))
        elif action == 3:
            match = re.search(r"turn right (\d+) degree", output_text, re.IGNORECASE)
            degree = int(match.group(1)) if match else 15
            degree = self._snap_value(degree, 15, QUEUE_DEGREE_CHOICES)
            queue_actions.extend([3] * max(int(degree // 15) - 1, 0))

        return action, queue_actions

    @torch.inference_mode()
    def eval_episode(self, env_wrapper: SatNavEnvWrapper, episode) -> dict:
        obs = env_wrapper.reset(episode)
        instruction = self.get_instruction(episode)

        past_rgbs = []
        queue_actions = []
        step_id = 0
        max_steps = env_wrapper.max_steps

        while not env_wrapper.episode_over and step_id < max_steps:
            curr_rgb = Image.fromarray(np.uint8(env_wrapper.get_rgb(obs))).convert("RGB")

            if queue_actions:
                action = queue_actions.pop(0)
            else:
                past_and_current = past_rgbs + [curr_rgb]
                num_video_frames = getattr(self.model.config, "num_video_frames", 8)
                sampled_frames = sample_and_pad_images(
                    past_and_current,
                    num_frames=num_video_frames,
                )

                output_text = self.predict_action_text(sampled_frames, instruction)
                action, queue_actions = self.parse_action_and_queue(output_text)

            past_rgbs.append(curr_rgb)
            obs, _ = env_wrapper.step(action)
            step_id += 1

        metrics = env_wrapper.get_metrics()
        metrics["_step_count"] = step_id
        return metrics


def create_satnav_env(config_path: str, eval_split: str) -> SatNavEnvWrapper:
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.DATASET.SPLIT = eval_split
    OmegaConf.set_struct(config, True)

    dataset = SatNavDataset(config.DATASET)
    satnav_env = SatNavEnv(config, dataset=dataset, cycle=False)
    return SatNavEnvWrapper(satnav_env, config=config)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def init_distributed(args):
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


def save_summary(results: list, output_path: str, args) -> None:
    if not results:
        print("[Warning] No results to summarize.")
        return

    sucs = [r["success"] for r in results]
    spls = [r["spl"] for r in results]
    oss = [r["oracle_success"] for r in results]
    nes = [r["distance_to_goal"] for r in results if r["distance_to_goal"] < 1e6]
    steps = [r["steps"] for r in results]

    summary = {
        "eval_split": args.eval_split,
        "SR": float(np.mean(sucs)) if sucs else 0.0,
        "SPL": float(np.mean(spls)) if spls else 0.0,
        "OS": float(np.mean(oss)) if oss else 0.0,
        "NE": float(np.mean(nes)) if nes else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "total_episodes": len(results),
        "model_path": args.model_path,
    }

    type_stats = {}
    for result in results:
        ttype = result.get("trajectory_type", "unknown")
        type_stats.setdefault(
            ttype,
            {"sucs": [], "spls": [], "oss": [], "nes": [], "steps": []},
        )
        type_stats[ttype]["sucs"].append(result["success"])
        type_stats[ttype]["spls"].append(result["spl"])
        type_stats[ttype]["oss"].append(result["oracle_success"])
        if result["distance_to_goal"] < 1e6:
            type_stats[ttype]["nes"].append(result["distance_to_goal"])
        type_stats[ttype]["steps"].append(result["steps"])

    if len(type_stats) > 1 or "unknown" not in type_stats:
        summary["by_trajectory_type"] = {}
        for ttype, stats in sorted(type_stats.items()):
            summary["by_trajectory_type"][ttype] = {
                "SR": float(np.mean(stats["sucs"])),
                "SPL": float(np.mean(stats["spls"])),
                "OS": float(np.mean(stats["oss"])),
                "NE": float(np.mean(stats["nes"])) if stats["nes"] else 0.0,
                "avg_steps": float(np.mean(stats["steps"])),
                "count": len(stats["sucs"]),
            }

    print("\n" + "=" * 60)
    print(f"NaVILA SatNav Evaluation Summary ({args.eval_split})")
    print("=" * 60)
    print(f"Success Rate: {summary['SR']:.2%}")
    print(f"SPL:          {summary['SPL']:.4f}")
    print(f"Oracle Succ:  {summary['OS']:.2%}")
    print(f"Nav Error:    {summary['NE']:.2f}m")
    print(f"Avg Steps:    {summary['avg_steps']:.2f}")
    print(f"Total:        {summary['total_episodes']}")
    if "by_trajectory_type" in summary:
        print("\n--- By Trajectory Type ---")
        for ttype, stats in summary["by_trajectory_type"].items():
            print(
                f"  [{ttype}] SR: {stats['SR']:.2%}, SPL: {stats['SPL']:.4f}, "
                f"OS: {stats['OS']:.2%}, NE: {stats['NE']:.2f}m, "
                f"Steps: {stats['avg_steps']:.2f}, N: {stats['count']}"
            )
    print("=" * 60)

    with open(os.path.join(output_path, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def aggregate_distributed(results: list, world_size: int, output_path: str, args) -> None:
    rank = get_rank()
    device = torch.device(f"cuda:{rank}")

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
        summary = {
            "eval_split": args.eval_split,
            "SR": float(torch.cat(sucs_all).mean().cpu().item()),
            "SPL": float(torch.cat(spls_all).mean().cpu().item()),
            "OS": float(torch.cat(oss_all).mean().cpu().item()),
            "NE": float(torch.cat(nes_all).mean().cpu().item()),
            "total_episodes": int(sum(x.item() for x in ep_num_all)),
            "model_path": args.model_path,
        }

        print("\n" + "=" * 60)
        print(f"NaVILA SatNav Evaluation Summary ({args.eval_split})")
        print("=" * 60)
        print(f"Success Rate: {summary['SR']:.2%}")
        print(f"SPL:          {summary['SPL']:.4f}")
        print(f"Oracle Succ:  {summary['OS']:.2%}")
        print(f"Nav Error:    {summary['NE']:.2f}m")
        print(f"Total:        {summary['total_episodes']}")
        print("=" * 60)

        with open(os.path.join(output_path, "evaluation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def evaluate(model, tokenizer, image_processor, args) -> None:
    model.eval()
    world_size = get_world_size()
    rank = get_rank()
    is_main = rank == 0

    evaluator = NaVILASatNavEvaluator(
        args=args,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    env_wrapper = create_satnav_env(args.satnav_config_path, args.eval_split)

    all_episodes = env_wrapper.env._dataset.episodes
    if args.max_episodes is not None:
        all_episodes = all_episodes[: args.max_episodes]

    scene_episode_dict = {}
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

    result_file = os.path.join(args.output_path, "result.jsonl")
    done_ids = set()
    results = []
    if os.path.exists(result_file):
        with open(result_file, encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    done_ids.add(str(result.get("episode_id", "")))
                    results.append(result)
                except json.JSONDecodeError:
                    pass
        if is_main:
            print(f"[Resume] Loaded {len(done_ids)} done episodes from {result_file}")

    pbar = tqdm.tqdm(
        my_episodes,
        desc=f"Rank {rank}" if world_size > 1 else "Evaluating NaVILA",
        disable=not is_main,
    )

    for episode in pbar:
        ep_id = str(episode.episode_id)
        if ep_id in done_ids:
            continue

        instruction = evaluator.get_instruction(episode)
        trajectory_type = getattr(episode, "trajectory_type", None)

        try:
            start_time = time.time()
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
                "elapsed_sec": round(time.time() - start_time, 3),
            }
        except Exception as exc:
            print(f"[Rank {rank}] Error on episode {ep_id}: {exc}")
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
                "error": str(exc),
            }

        if trajectory_type is not None:
            result["trajectory_type"] = trajectory_type

        results.append(result)
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

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


def main():
    parser = argparse.ArgumentParser(description="NaVILA SatNav Evaluation")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--satnav_config_path", type=str, required=True)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val_unseen",
        choices=["train", "val_seen", "val_unseen", "test"],
    )
    parser.add_argument("--output_path", type=str, default="./results/satnav/navila")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    args = parser.parse_args()

    init_distributed(args)
    device = f"cuda:{args.local_rank}"

    model_name = os.path.basename(os.path.normpath(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        model_name,
        model_base=args.model_base,
        device_map={"": args.local_rank},
        device=device,
    )
    model.to(device)
    model.eval()

    os.makedirs(args.output_path, exist_ok=True)
    evaluate(model, tokenizer, image_processor, args)


if __name__ == "__main__":
    main()
