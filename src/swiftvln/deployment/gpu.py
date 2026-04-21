from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import subprocess
from typing import Callable, Iterable, List, Sequence, Set


class DeploymentGPUError(RuntimeError):
    """Raised when deployment GPU selection fails."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    has_compute_process: bool = False


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _parse_csv_lines(text: str) -> List[List[str]]:
    reader = csv.reader(io.StringIO(text.strip()))
    return [row for row in reader if row]


def query_gpus(runner: CommandRunner = _default_runner) -> List[GPUInfo]:
    proc = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        raise DeploymentGPUError(proc.stderr.strip() or "nvidia-smi --query-gpu failed")

    gpus: List[GPUInfo] = []
    for row in _parse_csv_lines(proc.stdout):
        if len(row) < 4:
            continue
        gpus.append(
            GPUInfo(
                index=int(row[0].strip()),
                uuid=row[1].strip(),
                name=row[2].strip(),
                memory_used_mib=int(row[3].strip()),
            )
        )
    return gpus


def query_busy_gpu_uuids(runner: CommandRunner = _default_runner) -> Set[str]:
    proc = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ]
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "No running processes found" in stderr:
            return set()
        if not proc.stdout.strip() and not stderr:
            return set()
        raise DeploymentGPUError(stderr or "nvidia-smi --query-compute-apps failed")

    busy: Set[str] = set()
    for row in _parse_csv_lines(proc.stdout):
        if not row:
            continue
        gpu_uuid = row[0].strip()
        if gpu_uuid:
            busy.add(gpu_uuid)
    return busy


def mark_busy_gpus(gpus: Iterable[GPUInfo], busy_uuids: Set[str]) -> List[GPUInfo]:
    return [
        GPUInfo(
            index=gpu.index,
            uuid=gpu.uuid,
            name=gpu.name,
            memory_used_mib=gpu.memory_used_mib,
            has_compute_process=gpu.uuid in busy_uuids,
        )
        for gpu in gpus
    ]


def select_idle_h100(gpus: Iterable[GPUInfo]) -> GPUInfo:
    candidates = [
        gpu
        for gpu in gpus
        if "H100" in gpu.name and not gpu.has_compute_process and gpu.memory_used_mib <= 1024
    ]
    if not candidates:
        raise DeploymentGPUError("No idle H100 found on this machine.")

    return sorted(candidates, key=lambda gpu: (gpu.memory_used_mib, gpu.index))[0]


def select_idle_h100_from_system(runner: CommandRunner = _default_runner) -> GPUInfo:
    gpus = query_gpus(runner=runner)
    busy_uuids = query_busy_gpu_uuids(runner=runner)
    return select_idle_h100(mark_busy_gpus(gpus, busy_uuids))
