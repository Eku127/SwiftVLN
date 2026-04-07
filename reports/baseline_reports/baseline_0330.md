# 概述

本文档用于整理当前 baseline 的训练与评测情况。下面先给出状态矩阵（基于本文档现有记录）：

`✅` 已完成，`⏳` 待办，`➖` N/A

| Baseline | Scratch 训练 | Continue 训练 | Scratch 评测 | Continue 评测 |
| --- | --- | --- | --- | --- |
| Seq2Seq | ✅ | ➖ | ✅ | ➖ |
| CMA | ✅ | ➖ | ✅ | ➖ |
| StreamVLN | ✅ | ✅ | ✅ | ✅ |
| NaVILA | ✅ | ⏳ | ➖ | ⏳ |
| UniNaVid | ✅ | ✅ | ✅ | ⏳ |

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

`0404`：

1. `a. streamvln-baseline-continue-1ep-f32h8s4-data260317-bs32-lr2e-5-20260318-170553/checkpoint-6091`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260317-bs32-lr2e-5-20260318-170553`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_streamvln_continue_152531.log`
   `执行信息`：`98` 服务器，`8` 卡并行，`SATNAV_VERSION=ver_260404`
   `val_seen`：SR `67.70%`，SPL `67.29%`，OS `77.97%`，NE `96.16`，count `6338`
   `val_unseen`：SR `57.82%`，SPL `57.43%`，OS `73.06%`，NE `102.82`，count `8917`
   `整体`：SR `61.93%`

2. `b. streamvln-baseline-scratch-1ep-f32h8s4-data260317-bs32-lr2e-5-20260319-181417/checkpoint-6091`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260317-bs32-lr2e-5-20260319-181417`
   `执行信息`：`98` 服务器，`8` 卡并行，`SATNAV_VERSION=ver_260404`
   `val_seen`：SR `66.49%`，SPL `65.62%`，OS `77.36%`，NE `94.28`，count `6338`
   `val_unseen`：SR `56.70%`，SPL `55.91%`，OS `73.19%`，NE `94.73`，count `8917`
   `整体`：SR `60.77%`

## NaVILA

### Train

当前主线 NaVILA 训练产物如下（sample 策略：`hk7-fs7-stopx4`）：

1. `a. navila-continue-data0327-8gpu-full-r2-sample-hk7-fs7-stopx4`
   `最终模型`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue-data0327-8gpu-full-r2-sample-hk7-fs7-stopx4`
   `保留 ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue-data0327-8gpu-full-r2-sample-hk7-fs7-stopx4/checkpoint-67245`
   `训练数据`：`0327`
   `执行信息`：`17` 服务器，`8` 卡全量训练（continue），已完成
   `训练结果`：`89656/89656`，总时长 `57:54:31`，`train_loss=0.1014`

### Legacy

以下为历史 NaVILA 模型，已迁入 legacy，不作为当前主线：

1. `a. navila-baseline-scratch-1ep-8f-data260327-bs32-lr3e-5-20260328-221220`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-baseline-scratch-1ep-8f-data260327-bs32-lr3e-5-20260328-221220/checkpoint-63549`
   `训练数据`：`0327`

2. `b. navila-baseline-continue-1ep-8f-data260327-bs32-lr3e-5-20260330-224110`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-baseline-continue-1ep-8f-data260327-bs32-lr3e-5-20260330-224110/checkpoint-63549`
   `训练数据`：`0327`

### Eval

`0327`：`无`

## UniNaVid

### Train

当前主线 UniNaVid 训练产物如下（使用最新 `0327` 数据）：

1. `a. uninavid-baseline-scratch-1ep-data260327-bs192-lr1e-5-h98-20260402-212539`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260327-bs192-lr1e-5-h98-20260402-212539/checkpoint-6000`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260327-bs192-lr1e-5-h98-20260402-212539`
   `训练数据`：`0327`
   `执行信息`：`98` 服务器，`8` 卡并行，已完成

2. `b. uninavid-baseline-continue-1ep-data260327-bs192-lr1e-5-h98-20260402-212539`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260327-bs192-lr1e-5-h98-20260402-212539/checkpoint-6000`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260327-bs192-lr1e-5-h98-20260402-212539`
   `训练数据`：`0327`
   `执行信息`：`98` 服务器，`8` 卡并行，已完成

### Legacy

以下 `0317` 训练产物已归为 legacy，不再作为当前主线：

1. `a. uninavid-baseline-continue-1ep-data260317-bs192-lr1e-5-h17-20260324-181551`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260317-bs192-lr1e-5-h17-20260324-181551/checkpoint-6000`
   `训练数据`：`0317`

2. `b. uninavid-baseline-scratch-1ep-data260317-bs192-lr1e-5-h17-20260326-113504`
   `ckpt`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260317-bs192-lr1e-5-h17-20260326-113504/checkpoint-6000`
   `训练数据`：`0317`

### Eval

`0327`：

1. `a. uninavid-baseline-scratch-1ep-data260327-bs192-lr1e-5-h98-20260402-212539/checkpoint-6000`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-scratch-1ep-data260327-bs192-lr1e-5-h98-20260402-212539`
   `执行信息`：`73` 服务器，`SATNAV_VERSION=ver_260327`
   `val_seen`：首轮 `8` 卡完成
   `val_unseen`：首轮在 `val_seen -> val_unseen` 切换后因服务器掉到 `7` 张可见 GPU，`8` 卡启动时报 `invalid device ordinal`；随后改为 `7` 卡补跑完成
   `val_seen`：SR `19.44%`，SPL `19.09%`，OS `43.00%`，NE `367.78`，steps `82.53`，count `5912`
   `val_unseen`：SR `18.50%`，SPL `18.16%`，OS `44.72%`，NE `398.82`，steps `96.35`，count `7999`

2. `b. uninavid-baseline-continue-1ep-data260327-bs192-lr1e-5-h98-20260402-212539/checkpoint-6000`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-continue-1ep-data260327-bs192-lr1e-5-h98-20260402-212539`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid_continue0327_g7_retry_215943.log`
   `执行信息`：`73` 服务器，当前仅 `7` 张可见 GPU，因此以 `7` 卡并行启动，`SATNAV_VERSION=ver_260327`
   `状态`：进行中（记录时刻：`2026-04-04 22:00 CST`）
