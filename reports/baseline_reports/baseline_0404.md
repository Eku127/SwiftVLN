# 概述

本文档用于整理当前使用 `0404` 数据训练的 baseline 模型及其评测情况。下面先给出状态矩阵（基于当前仓库现有记录）：

`说明`：本文档中的 baseline 评测结果已按 `runtime/analysis/satnav_keep_lists/data260404_road345/` 的 keep list 回填。`val_seen / val_unseen` 指标均为各自 keep 子集上的直接平均；`overall` / `ALL` 指标为两个 split 的所有匹配 keep episodes 合并后的直接平均。

`✅` 已完成，`⏳` 进行中，`➖` 暂无记录 / 不适用

| Baseline | Scratch 训练 | Continue 训练 | Scratch 评测 | Continue 评测 |
| --- | --- | --- | --- | --- |
| Seq2Seq | ✅ | ➖ | ✅ | ➖ |
| CMA | ✅ | ➖ | ✅ | ➖ |
| StreamVLN | ✅ | ✅ | ✅ | ✅ |
| NaVILA | ✅ | ✅ | ✅ | ✅ |
| UniNaVid | ✅ | ✅ | ✅ | ✅ |

# 传统模型

## Seq2Seq

### Train

当前 `0404` 的 Seq2Seq baseline 训练产物如下：

1. `seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/checkpoints/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/logs/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab.log`
   `训练数据`：`0404`
   `词表 / embedding`：`0404 vocab + 0404 GloVe embedding`
   `执行信息`：`73` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成
   `训练结果`：日志已完成落盘，`best.pth` 已生成
   `best_loss`：`0.6489`

`备注`：当前 Seq2Seq `0404` 仅有 scratch 训练记录，没有 continue 训练链路。

### Eval

当前 `0404` 的 Seq2Seq baseline 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/checkpoints/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/results/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab`
   `数据版本`：`0404`
   `执行信息`：`73` 服务器，`8` 卡分片 eval（SatSim）
   `状态`：已完成
   `val_seen`：`keep直均 SR 1.44%`，`SPL 1.43%`，`NE 203.97`，`path_length 394.60`，`avg_steps 56.08`，`episodes 4643 / 4844 keep`
   `val_unseen`：`keep直均 SR 1.58%`，`SPL 1.54%`，`NE 219.17`，`path_length 438.85`，`avg_steps 58.49`，`episodes 8371 / 8756 keep`
   `overall`：`SR 1.53%`，`SPL 1.50%`，`NE 213.75`，`path_length 423.06`，`avg_steps 57.63`，`episodes 13014`

`备注`：本轮评测过程中部分 episode 因 `Camera view bounds exceed image bounds` 被跳过；原始参与统计 episode 数为 `val_seen=6086`、`val_unseen=8528`。按 keep list 过滤后，实际命中的 keep episode 数为 `val_seen=4643 / 4844`、`val_unseen=8371 / 8756`。

## CMA

### Train

当前 `0404` 的 CMA baseline 训练产物如下：

1. `cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/checkpoints/cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/logs/cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab.log`
   `训练数据`：`0404`
   `词表 / embedding`：`0404 vocab + 0404 GloVe embedding`
   `执行信息`：`73` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成
   `训练结果`：日志已完成落盘，`best.pth` 已生成
   `best_loss`：`0.3799`

`备注`：当前 CMA `0404` 仅有 scratch 训练记录，没有 continue 训练链路。

### Eval

当前 `0404` 的 CMA baseline 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/checkpoints/cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/cma/results/cma-ddp-g8-bs32-lr1e-4-20260413-125610-260404vocab`
   `数据版本`：`0404`
   `执行信息`：`73` 服务器，`8` 卡分片 eval（SatSim）
   `状态`：已完成
   `val_seen`：`keep直均 SR 9.46%`，`SPL 9.34%`，`NE 323.25`，`path_length 639.76`，`avg_steps 94.85`，`episodes 4430 / 4844 keep`
   `val_unseen`：`keep直均 SR 7.35%`，`SPL 7.16%`，`NE 363.59`，`path_length 679.84`，`avg_steps 99.31`，`episodes 7934 / 8756 keep`
   `overall`：`SR 8.10%`，`SPL 7.94%`，`NE 349.14`，`path_length 665.48`，`avg_steps 97.71`，`episodes 12364`

`备注`：本轮评测过程中部分 episode 因 `Camera view bounds exceed image bounds` 被跳过；原始参与统计 episode 数为 `val_seen=5675`、`val_unseen=8086`。按 keep list 过滤后，实际命中的 keep episode 数为 `val_seen=4430 / 4844`、`val_unseen=7934 / 8756`。当前 CMA 的 keep 子集 SR 仍明显高于 Seq2Seq，但 `NE / path_length / avg_steps` 也更高，说明其大量失败 case 会走得更远后再错误停下。

# 端到端模型

## StreamVLN

### Train

当前主线 StreamVLN `0404` 训练产物如下：

1. `a. streamvln-baseline-scratch-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-081630`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-081630`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/train_streamvln0404_chain_081629.log`
   `训练数据`：`0404`
   `执行信息`：`98` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成

2. `b. streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/train_streamvln0404_chain_081629.log`
   `训练数据`：`0404`
   `执行信息`：`98` 服务器，`8` 卡全量训练（continue）
   `状态`：已完成
   `训练结果`：`6873 / 6873`（`100%`），`epoch 1.0`
   `train_loss`：`0.0510`

`备注`：98 上采用 `scratch -> continue` 串行链路；当前 `scratch` 与 `continue` 均已完成。

### Eval

当前主线 StreamVLN `0404` 评测情况如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-081630`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-081630`
   `数据版本`：`0404`
   `执行信息`：`98` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`SR 57.99%`，`SPL 56.90%`，`OS 67.16%`，`NE 121.53`，`avg_steps 66.96`，`episodes 4844`
   `val_unseen`：`SR 57.14%`，`SPL 56.25%`，`OS 69.61%`，`NE 119.72`，`avg_steps 63.46`，`episodes 8756`
   `overall`：`SR 57.44%`，`SPL 56.48%`，`OS 68.74%`，`NE 120.36`，`avg_steps 64.71`，`episodes 13600`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `数据版本`：`0404`
   `执行信息`：`98` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`SR 62.61%`，`SPL 61.64%`，`OS 72.75%`，`NE 105.20`，`avg_steps 66.94`，`episodes 4844`
   `val_unseen`：`SR 62.36%`，`SPL 61.41%`，`OS 74.59%`，`NE 102.53`，`avg_steps 63.82`，`episodes 8756`
   `overall`：`SR 62.45%`，`SPL 61.49%`，`OS 73.93%`，`NE 103.48`，`avg_steps 64.93`，`episodes 13600`

`备注`：当前 StreamVLN 指标已切到 keep 子集直接平均口径，不再使用 `seen + unseen` 合并分布的重加权 SR。

## NaVILA

### Train

当前主线 NaVILA `0404` 训练产物如下：

1. `a. navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4/train.log`
   `训练数据`：`0404`
   `执行信息`：`17` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成（`2026-04-09 11:28 CST` 落盘完成）
   `训练配置`：`LR 3e-5`，`batch 4 x 1 x 8 = 32`，`Action fmt = compact`
   `训练结果`：`60000 / 60000`（`100%`），`epoch 0.5983`，`train_loss=0.1904`
   `训练速度`：`train_runtime=142290.04s`（约 `39.53h`），`13.494 samples/s`，`0.422 steps/s`
   `产出`：实验根目录下已落盘 `checkpoint-60000`、`llm/`、`vision_tower/`、`mm_projector/`

2. `b. navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4/train.log`
   `训练数据`：`0404`
   `执行信息`：`17` 服务器，`8` 卡全量训练（continue）
   `状态`：已完成（`2026-04-11 04:48 CST` 落盘完成）
   `训练配置`：`LR 3e-5`，`batch 4 x 1 x 8 = 32`，`Action fmt = compact`
   `训练结果`：`60000 / 60000`（`100%`），`epoch 0.5983`，`train_loss=0.1002`
   `训练速度`：`train_runtime=140454.32s`（约 `39.02h`），`13.670 samples/s`，`0.427 steps/s`
   `产出`：实验根目录下已落盘 `checkpoint-60000`、`llm/`、`vision_tower/`、`mm_projector/`

`备注`：17 上这轮 NaVILA `0404` 已完成 `scratch -> continue` 串行训练；对应 `scratch` 与 `continue` 的 `8` 卡评测也已完成，评测动作格式为 `compact`。

### Eval

当前主线 NaVILA `0404` 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/navila-baseline/navila-scratch0404-r1-20260407-195154-sample-hk7-fs7-stopx4`
   `数据版本`：`0404`
   `执行信息`：`17` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`SR 12.39%`，`SPL 12.37%`，`OS 21.55%`，`NE 122.98`，`avg_steps 48.02`，`episodes 4844`
   `val_unseen`：`SR 13.00%`，`SPL 12.96%`，`OS 23.66%`，`NE 128.88`，`avg_steps 53.71`，`episodes 8756`
   `overall`：`SR 12.78%`，`SPL 12.75%`，`OS 22.91%`，`NE 126.78`，`avg_steps 51.69`，`episodes 13600`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4`
   `数据版本`：`0404`
   `执行信息`：`17` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`SR 17.46%`，`SPL 17.38%`，`OS 28.76%`，`NE 118.31`，`avg_steps 56.83`，`episodes 4844`
   `val_unseen`：`SR 18.34%`，`SPL 18.16%`，`OS 31.72%`，`NE 124.96`，`avg_steps 61.88`，`episodes 8756`
   `overall`：`SR 18.03%`，`SPL 17.88%`，`OS 30.66%`，`NE 122.59`，`avg_steps 60.08`，`episodes 13600`

`备注`：当前 NaVILA 指标已切到 keep 子集直接平均口径，不再使用 `seen + unseen` 合并分布的重加权 SR。

## UniNaVid

### Train

当前主线 UniNaVid `0404` 训练产物如下：

1. `a. uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/train_uninavid0404_g7_081605_scratch.log`
   `训练数据`：`0404`
   `执行信息`：`73` 服务器，当前 `7` 张可见 GPU，因此以 `7` 卡全量训练（scratch）
   `状态`：已完成
   `训练结果`：`8262 / 8262`（`100%`），`epoch 1.0`
   `train_loss`：`0.0847`

2. `b. uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/train_uninavid0404_g7_081605_continue.log`
   `训练数据`：`0404`
   `执行信息`：`73` 服务器，当前 `7` 张可见 GPU，因此以 `7` 卡全量训练（continue）
   `状态`：已完成
   `训练结果`：`8262 / 8262`（`100%`），`epoch 1.0`
   `train_loss`：`0.0636`

`备注`：73 上采用 `scratch -> continue` 串行链路；当前 `scratch` 与 `continue` 均已完成。由于 73 当时仅有 `7` 张可见 GPU，本轮 UniNaVid `0404` 训练口径为 `bs168`，不是 `8` 卡 `bs192`。

### Eval

当前主线 UniNaVid `0404` 评测产物如下：

1. `a. uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-scratch-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid0404_g7_chain_091708.log`
   `评测数据`：`0404`
   `执行信息`：`73` 服务器，当前 `7` 张可见 GPU，因此以 `7` 卡评测（scratch）
   `状态`：已完成
   `val_seen`：`SR 22.73%`，`SPL 22.50%`，`OS 52.89%`，`NE 224.80`，`avg_steps 78.24`，`episodes 4844`
   `val_unseen`：`SR 22.52%`，`SPL 22.14%`，`OS 53.27%`，`NE 235.02`，`avg_steps 76.22`，`episodes 8756`
   `overall`：`SR 22.60%`，`SPL 22.27%`，`OS 53.13%`，`NE 231.38`，`avg_steps 76.94`，`episodes 13600`

2. `b. uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid0404_g7_chain_091708.log`
   `评测数据`：`0404`
   `执行信息`：`73` 服务器，当前 `7` 张可见 GPU，因此以 `7` 卡评测（continue）
   `状态`：已完成
   `val_seen`：`SR 41.45%`，`SPL 40.93%`，`OS 59.60%`，`NE 131.99`，`avg_steps 67.90`，`episodes 4844`
   `val_unseen`：`SR 39.71%`，`SPL 39.14%`，`OS 59.84%`，`NE 127.28`，`avg_steps 63.98`，`episodes 8756`
   `overall`：`SR 40.33%`，`SPL 39.77%`，`OS 59.76%`，`NE 128.96`，`avg_steps 65.37`，`episodes 13600`

`备注`：73 上以 `scratch -> continue` 串行链方式完成评测；`SATNAV_VERSION=ver_260404` 已固定，使用的是 `0404` 数据。由于第一次启动未稳定进入 tmux，本轮评测做过一次干净重启，旧 partial 结果已归档到 `runtime/trash/`。当前 UniNaVid 指标已切到 keep 子集直接平均口径，不再使用 `seen + unseen` 合并分布的重加权 SR。
