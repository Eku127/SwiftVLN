# Baseline 测试结果汇总（2026-03-31）

## 统一统计表（仅 split 级 SR，不含 overall）

| baseline | val_seen SR | val_unseen SR | delta (unseen-seen) |
| --- | ---: | ---: | ---: |
| `Seq2Seq` | `0.0686%` | `0.3542%` | `+0.2856` |
| `CMA` | `7.8183%` | `N/A` | `N/A` |
| `StreamVLN` | `71.3802%` | `63.1454%` | `-8.2348` |
| `NaVILA` | `1.8654%` | `N/A` | `N/A` |

## StreamVLN 按任务类型 SR（额外表格）

| task type | val_seen SR | val_unseen SR | delta (unseen-seen) |
| --- | ---: | ---: | ---: |
| `Boundary` | `88.2556%` | `86.8205%` | `-1.4351` |
| `LandmarkSet` | `59.5530%` | `55.6755%` | `-3.8775` |
| `Road` | `78.0808%` | `55.3667%` | `-22.7141` |

## OverlapVLN 0327 

| type | seen_SR | seen_SPL | seen_OS | seen_NE | unseen_SR | unseen_SPL | unseen_OS | unseen_NE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `55.55%` | `0.5504` | `65.73%` | `50.65` | `51.44%` | `0.5081` | `62.01%` | `65.51` |
| `baseline + gtc-k512` | `70.90%` | `0.7063` | `75.55%` | `49.66` | `60.57%` | `0.6020` | `68.55%` | `63.30` |
| `baseline + initial` | `42.60%` | `0.4081` | `64.52%` | `61.11` | `34.08%` | `0.3184` | `61.50%` | `80.01` |
| `baseline + log2.0` | `59.60%` | `0.5907` | `67.15%` | `47.14` | `53.70%` | `0.5310` | `62.42%` | `62.46` |
| `baseline + qa10` | `54.50%` | `0.5386` | `64.55%` | `46.96` | `51.62%` | `0.5078` | `61.41%` | `65.02` |
| `baseline + qa20` | `67.71%` | `0.6705` | `75.52%` | `64.02` | `60.31%` | `0.5953` | `70.82%` | `73.54` |

## 结果来源文件

- Seq2Seq
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval/val_seen_260327_parallel8/eval_ckpt_0_val_seen.json`
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval/val_unseen_260327_parallel8/eval_ckpt_0_val_unseen.json`
- CMA
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/results/cma-ddp-g8-bs32-lr1e-4-20260326-on260327/val_seen/eval_ckpt_0_val_seen.json`
- StreamVLN
  - `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/by-path/checkpoint-6091/val_seen/evaluation_summary.json`
  - `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/by-path/checkpoint-6091/val_unseen/evaluation_summary.json`
- NaVILA
  - `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/navila-baseline/by-path/checkpoint-63549/val_seen/result.jsonl`

## 训练模式说明（Seq2Seq / CMA）

- `Seq2Seq` 和 `CMA` 的默认训练流程就是 `scratch`。
- 两者训练脚本均未提供 `continue|scratch` 模式切换参数（均为直接生成新实验名训练）：
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/scripts/seq2seq/train_offline_ddp.sh`
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/scripts/cma/train_ddp.sh`
- 两者 baseline 配置默认均为 `IL.load_from_ckpt: false`：
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/configs/baselines/seq2seq_offline.yaml`
  - `/mnt/data1/home/jiangjiajun/workspace/SatNav/configs/baselines/cma.yaml`
