from pathlib import Path
import subprocess
from typing import Sequence


def _repo_root() -> Path:
    # src/swiftvln/runners/queue.py -> repo root
    return Path(__file__).resolve().parents[3]


def run_queue_train(extra_args: Sequence[str]) -> int:
    root = _repo_root()
    script = root / "src" / "swiftvln" / "scripts" / "train" / "train_queue.sh"
    cmd = ["bash", str(script), *list(extra_args)]
    proc = subprocess.run(cmd, cwd=root)
    return proc.returncode


def run_queue_eval(extra_args: Sequence[str]) -> int:
    root = _repo_root()
    script = root / "src" / "swiftvln" / "scripts" / "eval" / "eval_queue.sh"
    cmd = ["bash", str(script), *list(extra_args)]
    proc = subprocess.run(cmd, cwd=root)
    return proc.returncode
