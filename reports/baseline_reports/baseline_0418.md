# 概述

本文档用于整理当前使用 `0418` 数据训练的 baseline 模型及其训练与评测情况。下面先给出状态矩阵（基于当前文档回填情况）：

`说明`：若本轮评测使用了 keep list / 子集过滤，请在此补充统计口径说明；若没有，则可删除本行。

`✅` 已完成，`⏳` 进行中 / 待补充，`➖` 暂无记录 / 不适用

| Baseline | Scratch 训练 | Continue 训练 | Scratch 评测 | Continue 评测 |
| --- | --- | --- | --- | --- |
| Seq2Seq | ⏳ | ➖ | ⏳ | ➖ |
| CMA | ⏳ | ➖ | ⏳ | ➖ |
| StreamVLN | ✅ | ✅ | ⏳ | ⏳ |
| NaVILA | ⏳ | ⏳ | ⏳ | ⏳ |
| UniNaVid | ✅ | ⏳ | ⏳ | ⏳ |

# 传统模型

## Seq2Seq

### Train

当前 `0418` 的 Seq2Seq baseline 训练产物如下：

1. `a. scratch train`
   `输出目录`：`待补充`
   `训练日志`：`待补充`
   `训练数据`：`0418`
   `词表 / embedding`：`待补充`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `训练结果`：`待补充`
   `best_loss`：`待补充`

`备注`：当前 Seq2Seq `0418` 默认仅记录 scratch 训练；如存在 continue 链路，可按需补充。

### Eval

当前 `0418` 的 Seq2Seq baseline 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`待补充`
   `结果目录`：`待补充`
   `数据版本`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

`备注`：`待补充`

## CMA

### Train

当前 `0418` 的 CMA baseline 训练产物如下：

1. `a. scratch train`
   `输出目录`：`待补充`
   `训练日志`：`待补充`
   `训练数据`：`0418`
   `词表 / embedding`：`待补充`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `训练结果`：`待补充`
   `best_loss`：`待补充`

`备注`：当前 CMA `0418` 默认仅记录 scratch 训练；如存在 continue 链路，可按需补充。

### Eval

当前 `0418` 的 CMA baseline 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`待补充`
   `结果目录`：`待补充`
   `数据版本`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

`备注`：`待补充`

# 端到端模型

## StreamVLN

### Train

当前主线 StreamVLN `0418` 训练产物如下：

1. `a. scratch train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418-bs32-lr2e-5-20260418-210412`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418-bs32-lr2e-5-20260418-210412/train.log`
   `训练数据`：`0418`
   `执行信息`：`98` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成
   `训练结果`：`6885 / 6885`（`100%`），`epoch 1.0`
   `train_loss`：`0.0579`

2. `b. continue train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260418-bs32-lr2e-5-20260419-035419`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260418-bs32-lr2e-5-20260419-035419/train.log`
   `训练数据`：`0418`
   `执行信息`：`98` 服务器，`8` 卡全量训练（continue）
   `状态`：已完成
   `训练结果`：`6885 / 6885`（`100%`），`epoch 1.0`
   `train_loss`：`0.0509`

`备注`：`98` 上本轮已完成 `scratch -> continue` 串行训练；当前 `continue` 是从官方 `StreamVLN` checkpoint 起训，不是接 `scratch` 产物继续训。

### Eval

当前主线 StreamVLN `0418` 评测情况如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418-bs32-lr2e-5-20260418-210412`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418-bs32-lr2e-5-20260418-210412`
   `数据版本`：`0418`
   `执行信息`：`98` 服务器，`8` 卡 eval，按 `scratch -> continue` 串行启动
   `状态`：已启动，待完成
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260418-bs32-lr2e-5-20260419-035419`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260418-bs32-lr2e-5-20260419-035419`
   `数据版本`：`0418`
   `执行信息`：`98` 服务器，`8` 卡 eval，接在 `scratch eval` 后串行执行
   `状态`：待启动
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

`备注`：当前 `0418` 的 StreamVLN eval 尚未回填最终指标；本轮采用默认 SatNav 口径，直接评 `val_seen + val_unseen`。当前串行 eval session 为 `eval_streamvln0418_chain_105023`，总日志为 `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_streamvln0418_chain_105023.log`。

## NaVILA

### Train

当前主线 NaVILA `0418` 训练产物如下：

1. `a. scratch train`
   `输出目录`：`待补充`
   `训练日志`：`待补充`
   `训练数据`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `训练配置`：`待补充`
   `训练结果`：`待补充`
   `训练速度`：`待补充`
   `产出`：`待补充`

2. `b. continue train`
   `输出目录`：`待补充`
   `训练日志`：`待补充`
   `训练数据`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `训练配置`：`待补充`
   `训练结果`：`待补充`
   `训练速度`：`待补充`
   `产出`：`待补充`

`备注`：`待补充`

### Eval

当前主线 NaVILA `0418` 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`待补充`
   `结果目录`：`待补充`
   `数据版本`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

2. `b. continue eval`
   `模型目录`：`待补充`
   `结果目录`：`待补充`
   `数据版本`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

`备注`：`待补充`

## UniNaVid

### Train

当前主线 UniNaVid `0418` 训练产物如下：

1. `a. scratch train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/uninavid_sc_ct_73_203618.log`
   `训练数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡串行训练链的第一段（scratch）
   `状态`：已完成
   `训练结果`：已完成并切到 `continue`
   `train_loss`：`待最终回填`

2. `b. continue train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/uninavid_sc_ct_73_203618.log`
   `训练数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡串行训练链的第二段（continue）
   `状态`：进行中
   `训练结果`：当前约 `70%+`
   `train_loss`：`待完成后回填`

`备注`：`73` 上当前采用 `scratch -> continue` 串行训练；`continue` 从 `baseline/uninavid/model/Uni-Navid` 起训，不是从 `scratch` 产物继续训。

### Eval

当前主线 UniNaVid `0418` 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`待补充`
   `结果目录`：`待补充`
   `评测日志`：`待补充`
   `评测数据`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

2. `b. continue eval`
   `模型目录`：`待补充`
   `结果目录`：`待补充`
   `评测日志`：`待补充`
   `评测数据`：`0418`
   `执行信息`：`待补充`
   `状态`：`待补充`
   `val_seen`：`待补充`
   `val_unseen`：`待补充`
   `overall`：`待补充`

`备注`：`待补充`
