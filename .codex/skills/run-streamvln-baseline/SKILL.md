---
name: run-streamvln-baseline
description: "Run StreamVLN baseline training on SatNav data in SwiftVLN with preflight checks, tmux launch, live supervision, and checkpoint verification. Use when user asks to run streamvln baseline / streamvln基线训练 / 用satnav数据训练streamvln / 启动streamvln单模型训练."
---

# Run StreamVLN Baseline

## Scope

- Use this skill when user asks to run StreamVLN baseline training on SatNav data in SwiftVLN.
- Default training environment is conda env `streamvln-baseline`.

## Workflow

1. Preflight
- Confirm dataset/config/model output paths and GPU availability.
- Ensure training is executed inside conda env `streamvln-baseline`.

2. Environment setup
- Activate conda before training commands:
```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln-baseline
```

3. Launch and supervise
- Prefer existing scripts under `src/swiftvln/scripts/train/` for launch.
- Use tmux for long-running jobs and supervise with periodic status checks.
- Verify checkpoint generation after training starts.

## Notes

- StreamVLN project path: `/mnt/data1/home/jiangjiajun/workspace/StreamVLN`
- StreamVLN pinned commit: `60476e81f4c01b29f1a51a7469f1cb4addbc1d62`
- Training can run in conda env: `streamvln-baseline`
