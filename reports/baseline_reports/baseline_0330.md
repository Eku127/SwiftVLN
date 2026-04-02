# 概述

本文档用于整理当前 baseline 的训练与评测情况。下面先给出状态矩阵（基于本文档现有记录）：

`✅` 已完成，`⏳` 待办，`➖` N/A

| Baseline | Scratch 训练 | Continue 训练 | Scratch 评测 | Continue 评测 |
| --- | --- | --- | --- | --- |
| Seq2Seq | ✅ | ➖ | ✅ | ➖ |
| CMA | ✅ | ➖ | ✅ | ➖ |
| StreamVLN | ✅ | ✅ | ✅ | ✅ |
| NaVILA | ✅ | ⏳ | ⏳ | ⏳ |
| UniNaVid | ✅ | ✅ | ⏳ | ⏳ |

# 传统模型

## Seq2Seq

### Train

当前在 `SatNav` 仓库内可见的 Seq2Seq 训练产物如下，按时间顺序编号：

1. `a. seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/checkpoints/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval/best.pth`
   `训练方式`：离线监督训练（offline trainer）
   `训练数据`：`0317`

### Eval

`0327`：

1. `a. seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval-on260327`（98 服务器，8 卡并行）
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval`
   `val_seen 结果文件`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval/val_seen_260327_parallel8/eval_ckpt_0_val_seen.json`
   `val_unseen 结果文件`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260324-155954-cwfix-5ep-eval/val_unseen_260327_parallel8/eval_ckpt_0_val_unseen.json`
   `val_seen`：SR `0.07%`，SPL `0.07%`，NE `205.17`，PL `224.82`，steps `27.64`，count `5830/5912`
   `val_unseen`：SR `0.35%`，SPL `0.35%`，NE `211.99`，PL `247.67`，steps `30.81`，count `7905/7999`
   `备注`：评测过程中出现部分 `Camera view bounds exceed image bounds`，对应 episode 被跳过，未计入统计（`val_seen` 跳过 `82`，`val_unseen` 跳过 `94`）。

## CMA

### Train

当前在 `SatNav` 仓库内可见的 CMA 训练产物如下，按时间顺序编号：

1. `a. cma-ddp-g8-bs32-lr1e-4-20260326`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/checkpoints/cma-ddp-g8-bs32-lr1e-4-20260326/best.pth`
   `训练方式`：离线监督训练（offline trainer）
   `训练数据`：`0317`

### Eval

`0327`：

1. `a. cma-ddp-g8-bs32-lr1e-4-20260326-on260327`（98 服务器，8 卡并行）
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/results/cma-ddp-g8-bs32-lr1e-4-20260326-on260327/val_seen`
   `结果文件`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/results/cma-ddp-g8-bs32-lr1e-4-20260326-on260327/val_seen/eval_ckpt_0_val_seen.json`
   `val_seen`：SR `7.82%`，SPL `7.79%`，NE `366.47`，PL `605.71`，steps `88.65`，count `5372/5912`
   `备注`：评测过程中出现部分 `Camera view bounds exceed image bounds`，对应 episode 被跳过，未计入统计。

# 端到端模型

## StreamVLN

### Train

当前在仓库内可见的 StreamVLN 训练产物如下，按时间顺序编号：

1. `a. streamvln-baseline-continue-1ep-f32h8s4-data260317-bs32-lr2e-5-20260318-170553`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260317-bs32-lr2e-5-20260318-170553/checkpoint-6091`
   `训练数据`：`0317`

2. `b. streamvln-baseline-scratch-1ep-f32h8s4-data260317-bs32-lr2e-5-20260319-181417`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260317-bs32-lr2e-5-20260319-181417/checkpoint-6091`
   `训练数据`：`0317`

### Eval

`0327`：

1. `a. streamvln-baseline-continue-1ep-f32h8s4-data260317-bs32-lr2e-5-20260318-170553/checkpoint-6091`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/by-path/streamvln_continue_ckpt6091_data260327_20260328_034556`
   `val_seen`：SR `72.41%`，SPL `71.98%`，OS `76.93%`，NE `96.80`，count `5912`
   `val_unseen`：SR `64.63%`，SPL `64.22%`，OS `70.97%`，NE `104.81`，count `7999`

2. `b. streamvln-baseline-scratch-1ep-f32h8s4-data260317-bs32-lr2e-5-20260319-181417/checkpoint-6091`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/by-path/checkpoint-6091`
   `执行信息`：`98` 服务器，`8` 卡并行，`SATNAV_VERSION=ver_260327`
   `val_seen`：SR `71.38%`，SPL `70.43%`，OS `76.71%`，NE `94.88`，count `5912`
   `val_unseen`：SR `63.15%`，SPL `62.31%`，OS `71.28%`，NE `96.55`，count `7999`

## NaVILA

### Train

当前在仓库内可见的 NaVILA 训练产物如下，按时间顺序编号：

1. `a. navila-baseline-scratch-1ep-8f-data260327-bs32-lr3e-5-20260328-221220`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-baseline-scratch-1ep-8f-data260327-bs32-lr3e-5-20260328-221220/checkpoint-63549`
   `训练数据`：`0327`

### Eval

`0327`：`无`

## UniNaVid

### Train

当前在仓库内可见的 UniNaVid 训练产物如下，按时间顺序编号：

1. `a. uninavid-baseline-continue-1ep-data260317-bs192-lr1e-5-h17-20260324-181551`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260317-bs192-lr1e-5-h17-20260324-181551/checkpoint-6000`
   `训练数据`：`0317`

2. `b. uninavid-baseline-scratch-1ep-data260317-bs192-lr1e-5-h17-20260326-113504`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260317-bs192-lr1e-5-h17-20260326-113504/checkpoint-6000`
   `训练数据`：`0317`

### Eval

`0327`：`无`
