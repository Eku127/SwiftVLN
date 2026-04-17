# Landmark Navigation Failure Analysis Report

**Date**: 2026-02-06  
**Model**: OverlapVLN (overlapvln-satnav-stage1-3b, per_frame, log_base=2.0)  
**Best Model**: `overlapvln-satnav-stage1-3b-1ep-f32s4-overlap16-pf-h8-b2.0-pool-s2-bs64-lr2e-5-20260204-230157`

## 1. 问题概述

| 指标 | Boundary (504 episodes) | Landmark (458 episodes) |
|------|------------------------|-------------------------|
| Success Rate | **68.5%** | **4.4%** |
| SPL | 65.6% | 4.2% |
| Oracle Success | 72.0% | 2.2% |
| Mean NE | 13.3m | 230.7m |
| Avg Steps | 74.0 | 42.8 |

Landmark任务的成功率不到5%，且oracle success仅2.2%（意味着模型几乎从未接近过目标），这说明问题不在于"停得太早/太晚"，而是**根本走错了方向**。

## 2. 根因分析

### 2.1 根因 #1（最严重）：评估数据的指令转向方向与参考路径几何不一致

**这是最大的问题。** 对全部458个Landmark评估episode进行分析发现：

| 指令方向一致性 | Episode数 | 成功率 | 平均NE |
|---------------|----------|--------|--------|
| 一致（指令方向=几何方向） | 213 (50.5%) | **8.9%** (19/213) | 170m |
| 不一致（指令方向≠几何方向） | 209 (49.5%) | **0.0%** (0/209) | 314m |

**近一半（49.5%）的Landmark评估指令，转向方向与实际参考路径的几何方向完全相反。**

示例（Episode 528）：
- 指令："Move forward 130 meters toward the red roof, then perform a **60 degree turn left**."
- 参考路径几何：航向从105°变为195° = **向右转90°**
- 模型行为：遵循指令向**左**转60° → 航向变为45°（东北方向）
- 结果：远离目标（250m → 389m）

这些不一致的episode成功率为**0%**——模型忠实地遵循了指令中的转向方向，但因为指令是错的，模型不可避免地走向了错误的方向。

**按指令类型分析**：

| 指令类型 | 总数 | 方向一致 | 方向不一致 | 一致时成功率 |
|---------|-----|---------|-----------|------------|
| 明确角度（如"75 degree turn right"） | 274 | 55.8% | 44.2% | 7.8% |
| 视觉描述（如"turn right until..."） | 150 | 40.0% | 60.0% | 11.7% |

两种指令类型都存在严重的方向不一致问题，视觉描述类指令更差（60%不一致）。

**对比训练数据**：训练数据中仅有**11.9%**的指令存在方向不一致，远低于评估数据的49.5%。这说明评估数据的指令生成存在严重质量问题。

**同一路径的不同指令**：在170条唯一路径中，有4条路径的不同paraphrase存在方向矛盾（有的说left，有的说right），进一步说明指令生成质量不稳定。

### 2.2 根因 #2：模型缺乏视觉Landmark识别能力

即使在方向一致的213个episode中，成功率也仅为8.9%。通过Debug运行分析发现：

1. **模型不能识别卫星图中的Landmark来决定何时停止**
   - 多个episode中模型在转向后继续前进远超所需距离（如F×56=560m, F×41=410m）
   - 模型的停止时机似乎是随机的，不是基于视觉识别

2. **前进距离误差大**
   - 转向前距离误差：-30m 到 +121m
   - 转向后距离误差：-219m 到 +480m

3. **Debug典型模式**（12个episode分析）：

| 行为 | 正确率 |
|------|--------|
| 转向方向 | 6/6 (100%) — 当指令有明确角度时 |
| 转向角度（±15°内） | 6/6 (100%) |
| 前进距离准确 | 约30% — 偏差很大 |
| 停止时机准确 | 约17% — 大部分情况无法正确停止 |

模型学会了"前进→转向→前进→停止"的固定模式，但**不能通过视觉反馈来微调行为**。

### 2.3 根因 #3：Boundary vs Landmark任务的本质差异

| 特性 | Boundary | Landmark |
|------|----------|----------|
| 路径类型 | 闭环（返回起点） | 开放路径（到达新目标） |
| 容错性 | 高 — 沿边界走即可自我修正 | 低 — 方向错误会越走越远 |
| 成功距离阈值 | 10m | 30m |
| 需要Landmark识别 | 低（跟踪边界即可） | 高（需要识别目标地物） |
| 转向次数 | 连续微调 | 1-3次大角度转向 |

Boundary任务的闭环特性使得模型可以"大致跟随边界"就能成功。而Landmark任务要求精确的航向控制和地物识别能力，模型在这方面严重不足。

### 2.4 根因 #4：训练-评估数据分布差异

| 维度 | 训练数据 | 评估数据 |
|------|---------|---------|
| 指令方向不一致率 | 11.9% | **49.5%** |
| Boundary比例 | 57% | 52% |
| Landmark比例 | 43% | 48% |
| 动作分布(Landmark) | 79.4% FWD, 9.7% L, 8.5% R | — |
| 动作分布(Boundary) | 58.7% FWD, 23.5% L, 16.5% R | — |

训练数据中Landmark的动作几乎全是前进（79.4%），转向只有18%，这使模型倾向于"一直前进"。

## 3. 详细Debug证据

### 3.1 成功案例分析

**Episode 802 (SUCCESS)**：
- 指令："Move forward 80m until see the white dock, followed by 45° turn left. Move forward 80m until you see the white dock, then stop."
- 参考路径：90m → 左转75° → 80m
- 模型行为：F×9(90m) → L×4(60°) → F×8(80m) → STOP
- 距离：126m → 21m ✓
- 分析：方向一致，距离几乎完美匹配

**Episode 768 (SUCCESS)**：
- 指令："Move forward 200m until blue swimming pool, followed by 105° turn left. Move forward 230m..."
- 模型行为：F×25(250m) → L×6(90°) → F×19(190m) → STOP
- 距离：279m → 27m ✓
- 分析：前进距离有偏差（+121m, -40m），但最终位置恰好接近目标

### 3.2 失败案例分析

**Episode 665 (FAIL - 方向不一致)**：
- 指令："swing a right until it slides over to your middle-right"
- 参考路径需要：**左**转90°
- 模型行为：F×10 → **R**×8(+120°) → F×56(560m!) → STOP
- 距离：136m → 639m ✗
- 分析：模型遵循指令向右转，但应该向左转；转向后无法识别目标Landmark，一直前进

**Episode 948 (FAIL - 方向一致但距离错误)**：
- 指令："Move forward 130m until see the red roof, followed by 90° turn left"
- 模型行为：F×13(130m) → L×6(90°) → F×26(260m) → STOP
- 距离：90m → 316m ✗
- 分析：转向方向和角度都正确，但初始就朝着远离目标的方向前进

## 4. 建议修复方案

### 4.1 紧急修复：修正评估数据指令（优先级：最高）
- **检查并修复评估数据中49.5%的指令方向不一致问题**
- 预期：仅此修复就能将Landmark成功率从4.4%提升到约8-10%

### 4.2 增强视觉Grounding（优先级：高）
- 在训练数据中加入更多视觉Landmark识别的样本
- 考虑增加QA混合训练，让模型学会识别卫星图中的地物
- 训练时增加距离感知的数据增强

### 4.3 改进Landmark训练数据质量（优先级：高）
- 修复训练数据中11.9%的指令方向不一致
- 增加Landmark训练样本的比例（当前43%）
- 增加多转向（3+ turns）的训练样本

### 4.4 Imitation Learning的根本问题（优先级：中）
- Landmark任务的误差累积效应严重——一个小的方向偏差会导致后续所有步骤偏离
- 考虑引入DAgger等在线学习方法
- 或增加轨迹噪声增强（training augmentation），让模型学会从偏差中恢复

## 5. 相关文件

- Debug episode子集: `/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260202/episodes/eval/debug_landmark_episodes.json`
- Debug运行结果: `results/eval/overlapvln/debug_landmark_analysis/20260206_102248/`
  - 每个episode的详细报告: `debug_landmark/{episode_id}/debug_report.json`
  - 关键帧: `debug_landmark/{episode_id}/frame_*.jpg`
  - 导航视频: `videos/`
- 评估代码中的debug功能: `src/swiftvln/model/evaluator.py` (使用 `--debug_landmark` 开关)
