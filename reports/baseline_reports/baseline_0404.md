# 概述

本文档用于整理当前使用 `0404` 数据训练的 baseline 模型及其评测情况。下面先给出状态矩阵（基于当前仓库现有记录）：

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
   `val_seen`：`原始 SR 1.50%`，`SPL 1.48%`，`NE 207.95`，`path_length 400.55`，`avg_steps 54.92`
   `val_unseen`：`原始 SR 1.69%`，`SPL 1.65%`，`NE 217.80`，`path_length 436.75`，`avg_steps 58.16`

`备注`：本轮评测过程中部分 episode 因 `Camera view bounds exceed image bounds` 被跳过；最终参与统计的 episode 数为 `val_seen=6086`、`val_unseen=8528`。

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
   `val_seen`：`原始 SR 8.90%`，`SPL 8.80%`，`NE 389.53`，`path_length 699.01`，`avg_steps 100.80`
   `val_unseen`：`原始 SR 7.64%`，`SPL 7.46%`，`NE 359.64`，`path_length 673.56`，`avg_steps 98.22`

`备注`：本轮评测过程中部分 episode 因 `Camera view bounds exceed image bounds` 被跳过；最终参与统计的 episode 数为 `val_seen=5675`、`val_unseen=8086`。当前 CMA 的 SR 明显高于 Seq2Seq，但 `NE / path_length / avg_steps` 也更高，说明其大量失败 case 会走得更远后再错误停下。

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
   `val_seen`：`原始 SR 64.50%`，`SPL 63.49%`，`OS 72.31%`，`NE 107.72`，`重加权 SR 65.14%`
   `val_unseen`：`原始 SR 57.77%`，`SPL 56.89%`，`OS 70.09%`，`NE 118.01`，`重加权 SR 57.42%`
   `overall`：`SR 60.56%`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `数据版本`：`0404`
   `执行信息`：`98` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`原始 SR 68.67%`，`SPL 67.72%`，`OS 77.17%`，`NE 93.22`，`重加权 SR 69.42%`
   `val_unseen`：`原始 SR 62.90%`，`SPL 61.96%`，`OS 75.00%`，`NE 101.04`，`重加权 SR 62.57%`
   `overall`：`SR 65.30%`

`备注`：重加权 SR 使用 `seen + unseen` 合并后的任务类型分布作为统一权重：`Boundary 29.18%`、`LandmarkSet 37.01%`、`Road 33.81%`。该口径用于削弱 `seen/unseen` 任务类型配比差异对总 SR 的影响。

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
   `val_seen`：`原始 SR 13.21%`，`SPL 13.18%`，`OS 20.53%`，`NE 120.76`，`重加权 SR 14.21%`
   `val_unseen`：`原始 SR 13.74%`，`SPL 13.70%`，`OS 24.39%`，`NE 127.47`，`重加权 SR 13.06%`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/navila-baseline/navila-continue0404-r1-20260409-134207-sample-hk7-fs7-stopx4`
   `数据版本`：`0404`
   `执行信息`：`17` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`原始 SR 17.73%`，`SPL 17.66%`，`OS 26.70%`，`NE 119.39`，`重加权 SR 19.14%`
   `val_unseen`：`原始 SR 19.10%`，`SPL 18.92%`，`OS 32.42%`，`NE 123.49`，`重加权 SR 18.09%`

`备注`：重加权 SR 使用 `seen + unseen` 合并后的任务类型分布作为统一权重：`Boundary 29.18%`、`LandmarkSet 37.01%`、`Road 33.81%`。该口径用于削弱 `seen/unseen` 任务类型配比差异对总 SR 的影响。

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
   `val_seen`：`原始 SR 29.71%`，`SPL 29.34%`，`OS 57.79%`，`NE 218.15`，`avg_steps 72.92`，`重加权 SR 29.72%`
   `val_unseen`：`原始 SR 23.37%`，`SPL 22.99%`，`OS 53.86%`，`NE 231.33`，`avg_steps 75.45`，`重加权 SR 23.34%`

2. `b. uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid0404_g7_chain_091708.log`
   `评测数据`：`0404`
   `执行信息`：`73` 服务器，当前 `7` 张可见 GPU，因此以 `7` 卡评测（continue）
   `状态`：已完成
   `val_seen`：`原始 SR 50.30%`，`SPL 49.62%`，`OS 65.59%`，`NE 119.14`，`avg_steps 62.63`，`重加权 SR 49.75%`
   `val_unseen`：`原始 SR 40.45%`，`SPL 39.89%`，`OS 60.40%`，`NE 125.33`，`avg_steps 63.40`，`重加权 SR 41.15%`

`备注`：73 上以 `scratch -> continue` 串行链方式完成评测；`SATNAV_VERSION=ver_260404` 已固定，使用的是 `0404` 数据。由于第一次启动未稳定进入 tmux，本轮评测做过一次干净重启，旧 partial 结果已归档到 `runtime/trash/`。重加权 SR 使用 `seen + unseen` 合并后的任务类型分布作为统一权重：`Boundary 29.18%`、`LandmarkSet 37.01%`、`Road 33.81%`。
