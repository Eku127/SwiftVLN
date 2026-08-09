"""Training loop for Stage-A sim-to-real alignment."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from swiftvln.utils.distributed import init_distributed  # noqa: E402

if __package__ in {None, ""}:
    from tools.s2r.arguments import TrainConfig, parse_train_args
    from tools.s2r.dataset import SatDronePairDataset, collate_pair_batch
    from tools.s2r.eval import evaluate_model
    from tools.s2r.losses import bidirectional_contrastive_loss, global_cosine_loss
    from tools.s2r.model import (
        ProjectionHead,
        Sim2RealAdapter,
        TeacherVisionTower,
        masked_mean_pool,
        pad_token_sequences,
        resolve_device,
    )
else:
    from .arguments import TrainConfig, parse_train_args
    from .dataset import SatDronePairDataset, collate_pair_batch
    from .eval import evaluate_model
    from .losses import bidirectional_contrastive_loss, global_cosine_loss
    from .model import (
        ProjectionHead,
        Sim2RealAdapter,
        TeacherVisionTower,
        masked_mean_pool,
        pad_token_sequences,
        resolve_device,
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay with linear warmup."""
    total_steps = max(total_steps, 1)
    warmup_steps = max(warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_dataloader(
    manifest_path: str,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    is_train: bool,
    world_size: int,
    rank: int,
    device: torch.device,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = SatDronePairDataset(manifest_path, split=split, max_samples=max_samples)
    sampler: Optional[DistributedSampler] = None
    shuffle = is_train
    if is_train and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pair_batch,
        drop_last=is_train,
    )
    return dataloader, sampler


def _prepare_output_dir(output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _save_checkpoint(
    output_dir: Path,
    step: int,
    *,
    args: TrainConfig,
    adapter: Sim2RealAdapter,
    projection_head: ProjectionHead,
    best_metric: float,
    metrics: Dict[str, object],
    keep_last: int,
    mark_best: bool,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"step_{step:07d}.pt"
    checkpoint = {
        "step": step,
        "best_metric": best_metric,
        "teacher_model_path": args.teacher_model_path,
        "manifest_path": args.manifest_path,
        "adapter_kwargs": {
            "dim": _unwrap(adapter).dim,
            "num_layers": _unwrap(adapter).num_layers,
            "num_heads": _unwrap(adapter).num_heads,
            "mlp_ratio": _unwrap(adapter).mlp_ratio,
            "dropout": _unwrap(adapter).dropout,
        },
        "projection_kwargs": {
            "input_dim": _unwrap(projection_head).net[0].in_features,
            "hidden_dim": _unwrap(projection_head).net[0].out_features,
            "output_dim": _unwrap(projection_head).net[-1].out_features,
        },
        "adapter_state_dict": _unwrap(adapter).state_dict(),
        "projection_head_state_dict": _unwrap(projection_head).state_dict(),
        "metrics": metrics,
        "train_args": args.to_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    latest_path = output_dir / "latest.pt"
    shutil.copy2(checkpoint_path, latest_path)
    if mark_best:
        shutil.copy2(checkpoint_path, output_dir / "best.pt")

    old_ckpts = sorted(checkpoint_dir.glob("step_*.pt"))
    if keep_last > 0 and len(old_ckpts) > keep_last:
        for stale in old_ckpts[:-keep_last]:
            stale.unlink(missing_ok=True)
    return checkpoint_path


def _load_resume_checkpoint(
    resume_path: str,
    *,
    adapter: Sim2RealAdapter,
    projection_head: ProjectionHead,
) -> Tuple[int, float]:
    checkpoint = torch.load(resume_path, map_location="cpu")
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("best_metric", float("-inf")))


def train_main(argv=None):
    """CLI entry for Stage-A training."""
    args: TrainConfig = parse_train_args(argv)
    rank, world_size, local_rank = init_distributed()

    device = resolve_device(args.device)
    if device.type == "cuda" and world_size > 1:
        device = torch.device(f"cuda:{local_rank}")

    seed_everything(args.seed + rank)
    is_main = rank == 0

    if is_main:
        _prepare_output_dir(args.output_dir)
        _save_json(Path(args.output_dir) / "train_args.json", args.to_dict())
        _save_json(
            Path(args.output_dir) / "progress.json",
            {
                "status": "initializing",
                "step": 0,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    train_loader, train_sampler = _build_dataloader(
        args.manifest_path,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
        is_train=True,
        world_size=world_size,
        rank=rank,
        device=device,
    )

    val_loader = None
    if is_main:
        val_loader, _ = _build_dataloader(
            args.manifest_path,
            split=args.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_val_samples,
            is_train=False,
            world_size=1,
            rank=0,
            device=device,
        )

    teacher = TeacherVisionTower.from_pretrained(
        args.teacher_model_path,
        device=device,
        dtype_name=args.teacher_dtype,
    )
    projection_hidden_dim = args.projection_hidden_dim or teacher.hidden_size
    adapter = Sim2RealAdapter(
        dim=teacher.hidden_size,
        num_layers=args.adapter_layers,
        num_heads=args.adapter_heads,
        mlp_ratio=args.adapter_mlp_ratio,
        dropout=args.adapter_dropout,
    ).to(device=device, dtype=torch.float32)
    projection_head = ProjectionHead(
        input_dim=teacher.hidden_size,
        hidden_dim=projection_hidden_dim,
        output_dim=args.projection_dim,
    ).to(device=device, dtype=torch.float32)

    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("Distributed Stage-A training requires CUDA devices.")
        adapter = DistributedDataParallel(adapter, device_ids=[local_rank], output_device=local_rank)
        projection_head = DistributedDataParallel(projection_head, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(
        list(adapter.parameters()) + list(projection_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.max_steps
    if total_steps <= 0:
        total_steps = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    global_step = 0
    best_recall = float("-inf")
    if args.resume_checkpoint:
        global_step, best_recall = _load_resume_checkpoint(
            args.resume_checkpoint,
            adapter=_unwrap(adapter),
            projection_head=_unwrap(projection_head),
        )

    initial_global_step = global_step
    training_started_at = time.monotonic()
    progress_path = Path(args.output_dir) / "progress.json"
    metrics_log_path = Path(args.output_dir) / "metrics.jsonl"
    if is_main and not metrics_log_path.exists():
        metrics_log_path.write_text("", encoding="utf-8")
    if is_main:
        _save_json(
            progress_path,
            {
                "status": "running",
                "step": global_step,
                "total_steps": total_steps,
                "remaining_steps": max(0, total_steps - global_step),
                "world_size": world_size,
                "per_device_batch_size": args.batch_size,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
                "teacher_model_path": args.teacher_model_path,
                "manifest_path": args.manifest_path,
                "output_dir": args.output_dir,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    running = {"loss": 0.0, "contrast": 0.0, "cosine": 0.0, "steps": 0}
    optimizer.zero_grad(set_to_none=True)

    stop_training = False
    last_eval_step = -1
    last_eval_metrics: Optional[Dict[str, object]] = None
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= total_steps:
                stop_training = True
                break

            adapter.train()
            projection_head.train()

            uav_tokens_list = teacher.encode_images(batch["uav_image"])
            sat_tokens_list = teacher.encode_images(batch["sat_image"])
            uav_tokens, uav_mask = pad_token_sequences(uav_tokens_list, device=device, dtype=torch.float32)
            sat_tokens, sat_mask = pad_token_sequences(sat_tokens_list, device=device, dtype=torch.float32)

            aligned_uav = adapter(uav_tokens, attention_mask=uav_mask)
            uav_pooled = masked_mean_pool(aligned_uav, uav_mask)
            sat_pooled = masked_mean_pool(sat_tokens, sat_mask)

            uav_proj = torch.nn.functional.normalize(projection_head(uav_pooled), dim=-1)
            sat_proj = torch.nn.functional.normalize(projection_head(sat_pooled), dim=-1)

            contrast_loss = bidirectional_contrastive_loss(
                uav_proj,
                sat_proj,
                temperature=args.temperature,
                gather_distributed=world_size > 1,
            )
            cosine_loss = global_cosine_loss(uav_pooled, sat_pooled)
            loss = contrast_loss + cosine_loss
            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            running["loss"] += float(loss.item() * max(1, args.grad_accum_steps))
            running["contrast"] += float(contrast_loss.item())
            running["cosine"] += float(cosine_loss.item())
            running["steps"] += 1

            should_step = (running["steps"] % max(1, args.grad_accum_steps)) == 0
            if should_step:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(adapter.parameters()) + list(projection_head.parameters()),
                        max_norm=args.grad_clip_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                should_log = (
                    global_step == initial_global_step + 1
                    or global_step % 10 == 0
                    or global_step == total_steps
                )
                if is_main and should_log:
                    elapsed_seconds = time.monotonic() - training_started_at
                    completed_since_start = max(1, global_step - initial_global_step)
                    seconds_per_step = elapsed_seconds / completed_since_start
                    remaining_steps = max(0, total_steps - global_step)
                    eta_seconds = seconds_per_step * remaining_steps
                    log_payload = {
                        "event": "train_progress",
                        "step": global_step,
                        "total_steps": total_steps,
                        "progress_percent": round(100.0 * global_step / total_steps, 2),
                        "train_loss": running["loss"] / max(1, running["steps"]),
                        "contrast_loss": running["contrast"] / max(1, running["steps"]),
                        "cosine_loss": running["cosine"] / max(1, running["steps"]),
                        "lr": scheduler.get_last_lr()[0],
                        "seconds_per_step": seconds_per_step,
                        "elapsed_seconds": elapsed_seconds,
                        "elapsed": _format_duration(elapsed_seconds),
                        "eta_seconds": eta_seconds,
                        "eta": _format_duration(eta_seconds),
                        "remaining_steps": remaining_steps,
                        "epoch": epoch + 1,
                        "world_size": world_size,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    print(json.dumps(log_payload, ensure_ascii=False), flush=True)
                    _save_json(progress_path, {"status": "running", **log_payload})

                should_eval = is_main and val_loader is not None and (
                    (args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0)
                    or (args.eval_every_steps <= 0 and global_step == total_steps)
                )
                if should_eval:
                    _save_json(
                        progress_path,
                        {
                            "status": "evaluating",
                            "step": global_step,
                            "total_steps": total_steps,
                            "remaining_steps": max(0, total_steps - global_step),
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        },
                    )
                    eval_metrics = evaluate_model(
                        teacher,
                        _unwrap(adapter),
                        _unwrap(projection_head),
                        val_loader,
                        device=device,
                    )
                    eval_metrics["step"] = global_step
                    last_eval_step = global_step
                    last_eval_metrics = eval_metrics
                    with metrics_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(eval_metrics, ensure_ascii=False))
                        handle.write("\n")
                    current_recall = float(eval_metrics.get("u2s_r@1", 0.0))
                    mark_best = current_recall >= best_recall
                    if mark_best:
                        best_recall = current_recall
                    _save_checkpoint(
                        Path(args.output_dir),
                        global_step,
                        args=args,
                        adapter=adapter,
                        projection_head=projection_head,
                        best_metric=best_recall,
                        metrics=eval_metrics,
                        keep_last=args.num_keep_checkpoints,
                        mark_best=mark_best,
                    )
                    print(json.dumps({"event": "eval", **eval_metrics}, ensure_ascii=False), flush=True)

                if (
                    is_main
                    and not should_eval
                    and args.save_every_steps > 0
                    and global_step % args.save_every_steps == 0
                ):
                    _save_checkpoint(
                        Path(args.output_dir),
                        global_step,
                        args=args,
                        adapter=adapter,
                        projection_head=projection_head,
                        best_metric=best_recall,
                        metrics={"step": global_step, "event": "periodic_save"},
                        keep_last=args.num_keep_checkpoints,
                        mark_best=False,
                    )

        if world_size > 1:
            dist.barrier()
        if stop_training:
            break

    if is_main and val_loader is not None and last_eval_step != global_step:
        final_metrics = evaluate_model(
            teacher,
            _unwrap(adapter),
            _unwrap(projection_head),
            val_loader,
            device=device,
        )
        final_metrics["step"] = global_step
        last_eval_step = global_step
        last_eval_metrics = final_metrics
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_metrics, ensure_ascii=False))
            handle.write("\n")
        current_recall = float(final_metrics.get("u2s_r@1", 0.0))
        mark_best = current_recall >= best_recall
        if mark_best:
            best_recall = current_recall
        _save_checkpoint(
            Path(args.output_dir),
            global_step,
            args=args,
            adapter=adapter,
            projection_head=projection_head,
            best_metric=best_recall,
            metrics=final_metrics,
            keep_last=args.num_keep_checkpoints,
            mark_best=mark_best,
        )

    if is_main:
        elapsed_seconds = time.monotonic() - training_started_at
        completion_payload = {
            "status": "completed",
            "event": "train_complete",
            "step": global_step,
            "total_steps": total_steps,
            "progress_percent": 100.0 if global_step >= total_steps else round(
                100.0 * global_step / total_steps,
                2,
            ),
            "elapsed_seconds": elapsed_seconds,
            "elapsed": _format_duration(elapsed_seconds),
            "best_u2s_r@1": best_recall,
            "final_metrics": last_eval_metrics or {},
            "output_dir": args.output_dir,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _save_json(progress_path, completion_payload)
        print(json.dumps(completion_payload, ensure_ascii=False), flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train_main()
