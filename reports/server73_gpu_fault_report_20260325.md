# Server 73 GPU Fault Report

Date: 2026-03-25 16:05 CST
Host: `VM-152-73-ubuntu`
Server IP: `10.246.152.73`

## Summary

Server 73 is not in a healthy 8-GPU state.

Current confirmed state:

- `GPU0` is not accessible from `nvidia-smi`
- `GPU2` is not accessible from `nvidia-smi`
- Remaining visible GPUs are `1, 3, 4, 5, 6, 7`
- No NaVILA training process is still running
- The latest NaVILA training on 73 failed with NCCL/CUDA runtime errors consistent with device loss

Conclusion:

- Server 73 should be treated as degraded
- It is not suitable for 8-GPU NaVILA training until host-level GPU recovery is completed

## Evidence

### 1. Current GPU visibility

Observed on `2026-03-25 16:05:47 CST`:

`nvidia-smi -L` returned:

- `Unable to determine the device handle for gpu 0000:23:00.0: Unknown Error`
- `GPU 1: NVIDIA H100 80GB HBM3`
- `Unable to determine the device handle for gpu 0000:43:00.0: Unknown Error`
- `GPU 3: NVIDIA H100 80GB HBM3`
- `GPU 4: NVIDIA H100 80GB HBM3`
- `GPU 5: NVIDIA H100 80GB HBM3`
- `GPU 6: NVIDIA H100 80GB HBM3`
- `GPU 7: NVIDIA H100 80GB HBM3`

`nvidia-smi --query-gpu=...` returned:

- `Unable to determine the device handle for GPU0: 0000:23:00.0: Unknown Error`
- `Unable to determine the device handle for GPU2: 0000:43:00.0: Unknown Error`

Visible devices at check time:

- GPU 1, PCI `00000000:33:00.0`
- GPU 3, PCI `00000000:63:00.0`
- GPU 4, PCI `00000000:83:00.0`
- GPU 5, PCI `00000000:A3:00.0`
- GPU 6, PCI `00000000:C3:00.0`
- GPU 7, PCI `00000000:E3:00.0`

### 2. Latest failing training

Relevant experiment:

- `navila-pretrain-data0317-defaultsample-73-20260325_152810`

Log:

- `/tmp/navila-pretrain-data0317-defaultsample-73-20260325_152810.log`

Training did start successfully:

- dataset initialized
- dataloader built
- loss decreased normally in early phase

Then it failed with:

- `Process group watchdog thread terminated with exception: CUDA error: unknown error`
- `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`
- root failure on rank 0 with `SIGABRT`

This is consistent with a lower-level GPU/device failure rather than a pure training-logic failure.

### 3. No remaining training processes

At inspection time, there were no residual processes matching:

- `train_satnav.py`
- `torchrun`
- `swanlab_sidecar.py`

So the training is not still active in a partial or zombie state.

## What This Means

This is not explained by:

- NaVILA sample strategy
- SatNav `0317` data format
- SwanLab sidecar logging

Those can affect:

- sample count
- total steps
- startup time
- logging behavior

They do not explain:

- `nvidia-smi` losing device handles
- only specific GPUs becoming unreadable
- NCCL watchdog terminating after CUDA device errors

The fault domain is much more likely to be one of:

- NVIDIA driver / kernel / runtime state
- PCIe or device-level connectivity
- host-level GPU reset or hardware instability

## Limits Of Current Inspection

I could not inspect kernel-side fault logs because:

- `dmesg` access is not permitted for the current user on server 73

So I could not verify:

- `NVRM` messages
- `Xid` codes
- PCIe reset details
- host-level driver reset history

Those are the next places to inspect at the host/ops level.

## Recommended Actions

1. Do not use server 73 for new 8-GPU NaVILA training until GPU health is restored.
2. Have host-level ops inspect:
   - `dmesg -T`
   - NVIDIA `Xid` / `NVRM` logs
   - PCIe / device reset state for:
     - `0000:23:00.0`
     - `0000:43:00.0`
3. After host remediation, verify recovery with:
   - `nvidia-smi -L`
   - `nvidia-smi --query-gpu=index,pci.bus_id,...`
   - a short 8-GPU smoke run
4. Until then, run NaVILA on another healthy server, such as 17.

## Operational Recommendation For SwiftVLN

For immediate experiment continuity:

- mark server 73 as temporarily degraded for 8-GPU jobs
- avoid scheduling NaVILA full training on 73
- prefer server 17 until 73 GPU health is confirmed restored
