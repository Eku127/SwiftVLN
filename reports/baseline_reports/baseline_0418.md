# 概述

本文档用于整理当前使用 `0418` 数据训练的 baseline 模型及其训练与评测情况。下面先给出状态矩阵（基于当前文档回填情况）：

`说明`：本页 `StreamVLN` 最终结果使用 `0418` 数据的 `80%` 训练版本（`ver_260418p80`）作为收口结果；评测仍按默认 SatNav `val_seen + val_unseen` 口径。`OverlapVLN` 当前先回填纯 baseline setting（`overlap0 / pf-h8 / b1.0 / pool-s2 / noembed`）的 `20260421-125912` 结果。`Seq2Seq` 当前仅先回填 `0404` 训练 checkpoint 在 `0418` 数据上的 cross-eval，用于看跨版本泛化；`0418` 原生训练产物仍待补充。

`✅` 已完成，`⏳` 进行中 / 待补充，`➖` 暂无记录 / 不适用

| Baseline | Scratch 训练 | Continue 训练 | Scratch 评测 | Continue 评测 |
| --- | --- | --- | --- | --- |
| Seq2Seq | ⏳ | ➖ | ✅ | ➖ |
| CMA | ⏳ | ➖ | ⏳ | ➖ |
| OverlapVLN | ✅ | ➖ | ✅ | ➖ |
| StreamVLN | ✅ | ✅ | ✅ | ✅ |
| OpenFly | ✅ | ✅ | ✅ | ✅ |
| NaVILA | ⏳ | ⏳ | ⏳ | ⏳ |
| UniNaVid | ✅ | ✅ | ✅ | ✅ |

# 传统模型

## Seq2Seq

### Train

当前 `0418` 的 Seq2Seq baseline 训练产物如下：

1. `a. scratch train`
   `输出目录`：`待补充`
   `训练日志`：`待补充`
   `训练数据`：`0418`
   `词表 / embedding`：`待补充`
   `执行信息`：`0418` 原生 scratch 训练尚未回填；本页当前先记录 `0404` checkpoint 在 `0418` 上的 cross-eval
   `状态`：`未回填`
   `训练结果`：`待补充`
   `best_loss`：`待补充`

`备注`：当前 Seq2Seq `0418` 原生训练尚未补齐；本页先回填 `0404` 训练产物在 `0418` 上的 cross-eval，便于先看跨数据版本退化幅度。

### Eval

当前 `0418` 的 Seq2Seq baseline 评测产物先回填如下 cross-eval 结果：

1. `a. scratch eval (0404 -> 0418 cross-eval)`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_offline/checkpoints/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_cross_eval_0418_from_260404/results/by-path/seq2seq-offline-ddp-g8-bs64-lr3e-4-20260412-234752-260404vocab`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SatNav/output/seq2seq_cross_eval_0418_from_260404/logs/seq2seq_0418_xeval_191703.log`
   `数据版本`：`0418`
   `词表 / embedding`：`0404 vocab + 0404 GloVe embedding`
   `执行信息`：`73` 服务器，复用 `0404` scratch checkpoint，`8` 卡分片 eval（SatSim），按 `val_seen -> val_unseen` 串行执行
   `状态`：已完成
   `val_seen`：`SR 2.13% | SPL 0.0212 | OS 31.14% | NE 180.36m | Avg Steps 53.29 | Total 4271 / 4574`
   `val_unseen`：`SR 1.58% | SPL 0.0154 | OS 29.64% | NE 219.17m | Avg Steps 58.49 | Total 8371 / 8756`
   `overall`：`SR 1.76% | SPL 0.0173 | OS 30.15% | NE 206.06m | Avg Steps 56.74 | Total 12642 / 13330`

`备注`：本轮评测复用了 `0404` 训练出来的 checkpoint、词表和 GloVe embedding，因此它反映的是 `0404 -> 0418` 的 cross-version 泛化，不是 `0418` 原生训练 Seq2Seq 的最终口径。评测过程中有部分 episode 因 `Camera view bounds exceed image bounds` 被跳过，最终参与统计的是 `val_seen=4271 / 4574`、`val_unseen=8371 / 8756`。

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

## OverlapVLN

### Train

当前 `0418` 的 OverlapVLN 纯 baseline（`overlap0 / pf-h8 / b1.0 / pool-s2 / noembed`）先回填如下训练产物：

1. `a. scratch train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-data260418-bs64-lr2e-5-20260421-125912`
   `训练日志`：`待补充`
   `训练数据`：`0418`
   `执行信息`：`overlap0 / pf-h8 / b1.0 / pool-s2 / noembed` 纯 baseline setting，`1 epoch` 全量训练
   `状态`：已完成
   `训练结果`：`3518 / 3518`（`100%`），`epoch 1.0`
   `train_loss`：`0.1238`

`备注`：当前 `0418` 的 OverlapVLN baseline 先记录用户指定的 `20260421-125912` 这次 run；同 setting 另有 `20260419-113050` 可作为横向对照。

### Eval

当前 `0418` 的 OverlapVLN 纯 baseline 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-data260418-bs64-lr2e-5-20260421-125912/v0-20260421-125931/checkpoint-3518`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/eval/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-data260418-bs64-lr2e-5-20260421-125912`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/eval_queue_overlapvln-satnav-stage1-3b-1ep-f32s4-overlap0-pf-h8-b1.0-pool-s2-noembed-data260418-bs64-lr2e-5-20260421-125912_20260423_071308.log`
   `数据版本`：`0418`
   `执行信息`：默认 SatNav `val_seen + val_unseen` 口径，纯 baseline setting（`overlap0 / pf-h8 / b1.0 / pool-s2 / noembed`）
   `状态`：已完成
   `val_seen`：`SR 62.00% | SPL 0.6151 | OS 70.75% | NE 43.66m | Avg Steps 47.47 | Total 4574`
   `val_unseen`：`SR 49.26% | SPL 0.4870 | OS 58.37% | NE 78.17m | Avg Steps 54.78 | Total 8756`
   `overall`：当前 `baseline_0418` 中的 OverlapVLN baseline 回填使用该 run 结果；可与同 setting 的 `20260419-113050` 做对照

`备注`：原始 summary 分别位于 `val_seen/20260423_071309/evaluation_summary.json` 与 `val_unseen/20260423_080647/evaluation_summary.json`；最终指标以结果目录中的 summary 为准。

## StreamVLN

### Train

当前主线 StreamVLN `0418` 最终记录采用 `0418 80%` 训练结果，产物如下：

1. `a. scratch train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418p80-bs64-lr2e-5-20260421-051524`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/train_streamvln_scratch80_051524.log`
   `训练数据`：`0418 80%`（`ver_260418p80`）
   `执行信息`：`98` 服务器，`8` 卡全量训练（scratch）
   `状态`：已完成
   `训练结果`：`2769 / 2769`（`100%`），`epoch 1.0`
   `train_loss`：`0.0657`

2. `b. continue train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260418\n80-bs64-lr2e-5-20260420-153328`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/train_streamvln_continue80_153328.log`
   `训练数据`：`0418 80%`（`ver_260418p80`）
   `执行信息`：`98` 服务器，`8` 卡全量训练（continue）
   `状态`：已完成
   `训练结果`：`2769 / 2769`（`100%`），`epoch 1.0`
   `train_loss`：`0.0529`

`备注`：当前 `baseline_0418` 文档中的 StreamVLN 最终记录已切换为 `80%` 数据结果。`continue80` 的实际输出目录名中混入了一个换行，文中用 `\n` 转义显示；checkpoint 保存成功，但后续评测结果按路径落在 `/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/by-path/checkpoint-2769`。

### Eval

当前主线 StreamVLN `0418` 最终评测情况如下（基于 `0418 80%` 训练产物）：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418p80-bs64-lr2e-5-20260421-051524`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/streamvln-baseline-scratch-1ep-f32h8s4-data260418p80-bs64-lr2e-5-20260421-051524`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_streamvln_scratch80_0418_g7.log`
   `数据版本`：`0418`
   `执行信息`：`98` 服务器，`8` 卡 eval，默认 SatNav `val_seen + val_unseen`
   `状态`：已完成
   `val_seen`：`SR 64.30% | SPL 0.6366 | OS 71.71% | NE 53.24m | Avg Steps 51.58 | Total 4574`
   `val_unseen`：`SR 52.25% | SPL 0.5178 | OS 60.99% | NE 84.99m | Avg Steps 57.80 | Total 8756`
   `overall`：`scratch80` 结果已完整回填，可作为 `continue80` 对照组

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/streamvln-baseline/streamvln-baseline-continue-1ep-f32h8s4-data260418\n80-bs64-lr2e-5-20260420-153328/checkpoint-2769`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/streamvln-baseline/by-path/checkpoint-2769`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/streamvln_train_eval_train_chain_153328.log`
   `数据版本`：`0418`
   `执行信息`：`98` 服务器，`8` 卡 eval，训练完成后串行执行 `val_seen + val_unseen`
   `状态`：已完成
   `val_seen`：`SR 70.35% | SPL 0.6966 | OS 77.59% | NE 47.47m | Avg Steps 52.11 | Total 4574`
   `val_unseen`：`SR 58.44% | SPL 0.5780 | OS 68.33% | NE 86.63m | Avg Steps 61.20 | Total 8756`
   `overall`：当前 `baseline_0418` 文档中的 StreamVLN 最终结果采用该 `continue80` 记录

`备注`：`continue80` 训练脚本收尾时有一次 shell 引号报错，但不影响 checkpoint 保存，也不影响后续 `val_seen + val_unseen` 评测完成；最终指标以结果目录中的 `evaluation_summary.json` 为准。

## OpenFly

### Train

当前主线 OpenFly `0418` 先回填上一轮（`20260420`）成功收口结果，产物如下：

1. `a. scratch train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/openfly-baseline/openfly-baseline-1ep-data260418-bkscratch-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-233110`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/openfly_continue_73_233110.log`
   `恢复训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/openfly_continue_resume73_090438.log`
   `训练数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡全量训练（scratch，`compact` action，`hist16`）；首轮在 `56%` 左右触发一次 `NCCL/CUDA` 错误，之后恢复训练完成
   `状态`：已完成
   `训练结果`：`40055 / 40055`（`100%`），`epoch 1.0`
   `train_loss`：`0.0167`

2. `b. continue train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/openfly-baseline/openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/openfly_scratch_73_cachefix2_095357.log`
   `训练数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡全量训练（continue，`compact` action，`hist16`）
   `状态`：已完成
   `训练结果`：`40055 / 40055`（`100%`），`epoch 1.0`
   `train_loss`：`0.0673`

`备注`：当前上一轮 OpenFly 成功口径使用 `compact` action + `head_keep=7` + `sample_stride=3` + `stop_repeat=2` + `stop_window=0` + `tail_keep=5` + `hist16`。`continue` 在 `20260420-093943` 与 `20260420-095136` 有过未最终收口尝试，最终以 `20260420-095357` 这次成功训练为准。

### Eval

当前主线 OpenFly `0418` 上一轮评测产物如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/openfly-baseline/openfly-baseline-1ep-data260418-bkscratch-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-233110/checkpoint-40055`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/openfly-baseline/openfly-baseline-1ep-data260418-bkscratch-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-233110`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/openfly_eval_chain_73_202521.log`
   `数据版本`：`0418`
   `执行信息`：`73` 服务器，`8` 卡 eval，接在 `continue eval` 后串行执行
   `状态`：已完成
   `val_seen`：`SR 13.21% | SPL 0.1303 | OS 33.47% | NE 166.88m | Avg Steps 58.93 | Total 4601`
   `val_unseen`：`SR 11.73% | SPL 0.1162 | OS 32.17% | NE 195.91m | Avg Steps 66.35 | Total 8756`
   `overall`：同轮对比下，`scratch` 明显落后于 `continue`

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/openfly-baseline/openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357/checkpoint-40055`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/openfly-baseline/openfly-baseline-1ep-data260418-bkcontinue-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-bs96-lr2e-5-20260420-095357`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/openfly_eval_chain_73_202521.log`
   `数据版本`：`0418`
   `执行信息`：`73` 服务器，`8` 卡 eval，按 `continue -> scratch` 串行执行 `val_seen + val_unseen`
   `状态`：已完成
   `val_seen`：`SR 21.10% | SPL 0.2099 | OS 38.08% | NE 163.07m | Avg Steps 57.86 | Total 4601`
   `val_unseen`：`SR 17.12% | SPL 0.1691 | OS 34.88% | NE 196.90m | Avg Steps 65.57 | Total 8756`
   `overall`：当前上一轮 `0418` OpenFly 结果中，`continue` 明显高于 `scratch`

`备注`：该轮 `val_seen` 使用的是更新前口径，总 episode 数为 `4601`；后续 `20260422` 那轮评测的 `val_seen` 已变为 `4574`，两轮之间不能直接按 `val_seen` 做严格横比。最终指标以结果目录中的 `evaluation_summary.json` 为准。

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
   `训练结果`：`7246 / 7246`（`100%`），`epoch 1.0`
   `train_loss`：`0.0869`

2. `b. continue train`
   `输出目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `训练日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/uninavid_sc_ct_73_203618.log`
   `训练数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡串行训练链的第二段（continue）
   `状态`：已完成
   `训练结果`：`7246 / 7246`（`100%`），`epoch 1.0`
   `train_loss`：`0.0653`

`备注`：`73` 上本轮采用 `scratch -> continue` 串行训练；`scratch` 与 `continue` 都已完成。`continue` 从 `baseline/uninavid/model/Uni-Navid` 起训，不是从 `scratch` 产物继续训。

### Eval

当前主线 UniNaVid `0418` 评测产物如下：

1. `a. scratch eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-scratch-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-scratch-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid0418_chain_192509.log`
   `评测数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡 eval，最终成功链按 `scratch -> continue` 串行执行
   `状态`：已完成
   `val_seen`：`SR 25.12% | SPL 0.2481 | OS 60.43% | NE 174.68m | Avg Steps 73.21 | Total 4574`
   `val_unseen`：`SR 20.36% | SPL 0.2000 | OS 49.94% | NE 228.46m | Avg Steps 80.55 | Total 8756`
   `overall`：`scratch` 结果已完整回填，可作为 `continue` 对照组

2. `b. continue eval`
   `模型目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `结果目录`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/uninavid-baseline/uninavid-baseline-continue-1ep-data260418-bs192-lr1e-5-20260418-203618`
   `评测日志`：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/logs/train_launch/eval_uninavid0418_chain_192509.log`
   `评测数据`：`0418`
   `执行信息`：`73` 服务器，`8` 卡 eval，接在 `scratch eval` 后串行执行
   `状态`：已完成
   `val_seen`：`SR 49.69% | SPL 0.4915 | OS 68.17% | NE 87.11m | Avg Steps 53.98 | Total 4574`
   `val_unseen`：`SR 36.72% | SPL 0.3629 | OS 55.85% | NE 149.85m | Avg Steps 64.75 | Total 8756`
   `overall`：当前 `baseline_0418` 文档中的 UniNaVid 最终结果采用该 `continue` 记录

`备注`：`0418` 的 UniNaVid 评测前面有过失败尝试：一次是缺少 episodes 文件，另一次是中途出现大量 SatSim camera bounds 报错；最终成功收口的是 `eval_uninavid0418_chain_192509.log`，并已经把 `scratch + continue` 的 `val_seen + val_unseen` 全部跑完。最终指标以结果目录中的 `evaluation_summary.json` 为准。
