# 概述

本文档用于整理当前使用 `0404` 数据训练的 baseline 模型及其评测情况。下面先给出状态矩阵（基于当前仓库现有记录）：

`✅` 已完成，`⏳` 进行中，`➖` 暂无记录 / 不适用

| Baseline | Scratch 训练 | Continue 训练 | Scratch 评测 | Continue 评测 |
| --- | --- | --- | --- | --- |
| Seq2Seq | ➖ | ➖ | ➖ | ➖ |
| CMA | ➖ | ➖ | ➖ | ➖ |
| StreamVLN | ✅ | ✅ | ✅ | ✅ |
| NaVILA | ✅ | ⏳ | ➖ | ➖ |
| UniNaVid | ✅ | ✅ | ✅ | ✅ |

# 传统模型

## Seq2Seq

### Train

`0404`：`无`

### Eval

`0404`：`无`

## CMA

### Train

`0404`：`无`

### Eval

`0404`：`无`

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
   `val_seen`：`SR 64.50%`，`SPL 63.49%`，`OS 72.31%`，`NE 107.72`
   `val_unseen`：`SR 57.77%`，`SPL 56.89%`，`OS 70.09%`，`NE 118.01`
   `overall`：`SR 60.56%`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260404-bs32-lr2e-5-20260405-150737`
   `数据版本`：`0404`
   `执行信息`：`98` 服务器，`8` 卡 eval
   `状态`：已完成
   `val_seen`：`SR 68.67%`，`SPL 67.72%`，`OS 77.17%`，`NE 93.22`
   `val_unseen`：`SR 62.90%`，`SPL 61.96%`，`OS 75.00%`，`NE 101.04`
   `overall`：`SR 65.30%`

## NaVILA

### Train

当前主线 NaVILA `0404` 训练产物如下：

1. `a. navila-scratch-data0404-8gpu-r1-sample-hk7-fs7-stopx4`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-scratch-data0404-8gpu-r1-sample-hk7-fs7-stopx4`
   `训练数据`：`0404`
   `执行信息`：`17` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成（`2026-04-06 14:44 CST`）
   `训练结果`：`60000 / 60000`（`100%`），`epoch 0.6`，`train_loss=0.1369`
   `产出`：最终 checkpoint 及 `llm/vision_tower/mm_projector` 已从 `tmp-checkpoint-60000` 移到实验根目录
   `备注`：训练结束后 watchdog tmux 已退出，实验目录下 `train.log` 记录完整过程

2. `b. navila-continue-data0404-8gpu-r1-sample-hk7-fs7-stopx4`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/navila-baseline/navila-continue-data0404-8gpu-r1-sample-hk7-fs7-stopx4`
   `训练数据`：`0404`
   `执行信息`：`17` 服务器，`8` 卡训练（continue）
   `状态`：未完成 / 已中断
   `当前记录`：至少已跑到 `21655 / 60000`（约 `36.09%`），`epoch 0.22`
   `当前产出`：已有 `checkpoint-20000`
   `异常信息`：日志显示在 `2026-04-07` 凌晨附近收到 `SIGINT`，随后伴随多处 `DataLoader worker exited unexpectedly` 与 `KeyboardInterrupt`

### Eval

`0404`：`无`

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
   `val_seen`：`SR 29.71%`，`SPL 29.34%`，`OS 57.79%`，`NE 218.15`，`avg_steps 72.92`
   `val_unseen`：`SR 23.37%`，`SPL 22.99%`，`OS 53.86%`，`NE 231.33`，`avg_steps 75.45`

2. `b. uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-continue-1ep-data260404-bs168-lr1e-5-h73-20260405-081606`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid0404_g7_chain_091708.log`
   `评测数据`：`0404`
   `执行信息`：`73` 服务器，当前 `7` 张可见 GPU，因此以 `7` 卡评测（continue）
   `状态`：已完成
   `val_seen`：`SR 50.30%`，`SPL 49.62%`，`OS 65.59%`，`NE 119.14`，`avg_steps 62.63`
   `val_unseen`：`SR 40.45%`，`SPL 39.89%`，`OS 60.40%`，`NE 125.33`，`avg_steps 63.40`

`备注`：73 上以 `scratch -> continue` 串行链方式完成评测；`SATNAV_VERSION=ver_260404` 已固定，使用的是 `0404` 数据。由于第一次启动未稳定进入 tmux，本轮评测做过一次干净重启，旧 partial 结果已归档到 `runtime/trash/`。
