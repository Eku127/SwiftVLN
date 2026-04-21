"""
Diagnostic: test if the trained NaVILA model can predict correct actions on TRAINING data.
If yes → eval pipeline issue. If no → training issue.

Usage:
    CUDA_VISIBLE_DEVICES=X python baseline/navila/src/diagnose_train_vs_eval.py \
        --model_path /path/to/checkpoint \
        --num_samples 20
"""
import sys, os, json, argparse, re, copy
import torch
import numpy as np
from collections import Counter
from PIL import Image

_NAVILA_ROOT = "/mnt/data1/home/jiangjiajun/workspace/NaVILA"
if _NAVILA_ROOT not in sys.path:
    sys.path.insert(0, _NAVILA_ROOT)

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, vlnce_frame_sampling
from llava.model.builder import load_pretrained_model
from action_formats import ACTION_TO_NAME, build_prompt, get_action_to_text, normalize_action_format, parse_action_text

ACTION_NAMES = {0: "stop", 1: "forward", 2: "left", 3: "right"}


def normalize_instruction(text):
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+", " ", text)
    text = text.capitalize()
    text = re.sub(r"(?<=\.\s)([a-z])", lambda x: x.group().upper(), text)
    return text


def parse_action(text):
    return parse_action_text(text)


def sample_and_pad_images(images, num_frames=8):
    """Same logic as eval."""
    frames = copy.deepcopy(images)
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (448, 448), color=(0, 0, 0)))
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    return sampled_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                        default="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/trajectory_data/annotations.json")
    parser.add_argument("--image_folder", type=str,
                        default="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260418/trajectory_data")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--eval_dtype", type=str, default="auto")
    parser.add_argument(
        "--action_format",
        type=str,
        default=os.getenv("SATNAV_ACTION_FORMAT", "compact"),
    )
    args = parser.parse_args()
    args.action_format = normalize_action_format(args.action_format)
    action_to_text = get_action_to_text(args.action_format)

    device = "cuda:0"
    model_name = os.path.basename(os.path.normpath(args.model_path))
    print(f"Loading model from: {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_name, model_base=None,
        device_map={"": 0}, device=device,
    )

    if args.eval_dtype == "auto":
        eval_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        eval_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.eval_dtype]

    if next(model.parameters()).dtype != eval_dtype:
        model.to(dtype=eval_dtype)
        vt = model.get_vision_tower()
        if vt is not None:
            vt.to(device=device, dtype=eval_dtype)
        mp = model.get_mm_projector()
        if mp is not None:
            mp.to(device=device, dtype=eval_dtype)
    model.eval()
    print(f"Eval dtype: {eval_dtype}")
    print(f"Action format: {args.action_format}")

    with open(args.data_path) as f:
        episodes = json.load(f)

    num_video_frames = getattr(model.config, "num_video_frames", 8)
    print(f"num_video_frames: {num_video_frames}")

    import random
    random.seed(42)

    per_action = max(args.num_samples // 4, 5)
    action_target = {0: per_action, 1: per_action, 2: per_action, 3: per_action}

    candidates = {a: [] for a in action_target}
    for ep in episodes:
        actions = ep["actions"]
        instruction = normalize_instruction(ep["instructions"][0])
        video_dir = ep["video"]
        for step_idx in range(1, len(actions)):
            action = actions[step_idx]
            if action not in ACTION_TO_NAME:
                continue
            candidates[action].append({
                "episode_id": ep.get("id"),
                "step_idx": step_idx,
                "instruction": instruction,
                "video_dir": video_dir,
                "gt_action": action,
                "gt_text": action_to_text[action],
            })

    samples = []
    for a in sorted(action_target):
        pool = candidates[a]
        random.shuffle(pool)
        for c in pool[:action_target[a]]:
            c["frame_paths"] = [
                os.path.join(args.image_folder, c["video_dir"], "rgb", f"{fi:03d}.jpg")
                for fi in range(1, c["step_idx"] + 1)
            ]
            samples.append(c)
    random.shuffle(samples)
    print(f"\nTesting {len(samples)} samples: {dict(Counter(s['gt_action'] for s in samples))}")
    print("=" * 70)

    correct = 0
    results = []

    for i, sample in enumerate(samples):
        # Method 1: Use vlnce_frame_sampling (same as training)
        train_frames = vlnce_frame_sampling(list(sample["frame_paths"]), num_frames=num_video_frames)

        # Method 2: Use sample_and_pad_images (same as eval)
        pil_frames = [Image.open(p).convert("RGB") for p in sample["frame_paths"]]
        eval_frames = sample_and_pad_images(pil_frames, num_frames=num_video_frames)

        for method_name, frames in [("train_pipe", train_frames), ("eval_pipe", eval_frames)]:
            prompt_text = build_prompt(
                instruction=sample["instruction"],
                history_count=max(num_video_frames - 1, 0),
                action_format=args.action_format,
            )

            conv = conv_templates["llama_3"].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = process_images(frames, image_processor, model.config).to(device, dtype=eval_dtype)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, images=images_tensor.to(device),
                    do_sample=False, temperature=1.0, top_p=1.0,
                    max_new_tokens=32, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            if output_ids.shape[1] >= input_ids.shape[1]:
                gen_ids = output_ids[:, input_ids.shape[1]:]
            else:
                gen_ids = output_ids
            output_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            predicted_action = parse_action(output_text)
            is_correct = predicted_action == sample["gt_action"]
            if method_name == "train_pipe":
                correct += int(is_correct)

            mark = "OK" if is_correct else "WRONG"
            if i < 20 or not is_correct:
                print(f"[{i:2d}] [{method_name}] [{mark}] "
                      f"gt={ACTION_NAMES[sample['gt_action']]:>7s}  "
                      f"pred={ACTION_NAMES.get(predicted_action, 'NONE'):>7s}  "
                      f"ep={sample['episode_id']} step={sample['step_idx']}  "
                      f"output={output_text[:80]!r}")

            results.append({
                "method": method_name,
                "gt": sample["gt_action"],
                "pred": predicted_action,
                "correct": is_correct,
                "output": output_text[:100],
            })

    print("\n" + "=" * 70)
    for method in ["train_pipe", "eval_pipe"]:
        method_results = [r for r in results if r["method"] == method]
        n_correct = sum(1 for r in method_results if r["correct"])
        n_total = len(method_results)
        print(f"{method}: {n_correct}/{n_total} correct ({n_correct/n_total*100:.1f}%)")

        by_action = {}
        for r in method_results:
            gt = r["gt"]
            by_action.setdefault(gt, {"correct": 0, "total": 0})
            by_action[gt]["total"] += 1
            if r["correct"]:
                by_action[gt]["correct"] += 1
        for a in sorted(by_action):
            d = by_action[a]
            print(f"  {ACTION_NAMES[a]:>7s}: {d['correct']}/{d['total']}")


if __name__ == "__main__":
    main()
