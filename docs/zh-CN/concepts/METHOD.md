# 方法概览

> 框架页：本页只解释理解和复现 SwiftVLN 所必需的方法设计；命令、参数表和源码导览放在
> 各自页面。

## 1. 任务与模型输入输出

<!-- instruction、current observation、memory、action chunk。 -->

## 2. 训练与在线评测流程

<!-- 一张端到端数据流图；明确 offline trajectory 与 online rollout。 -->

## 3. 轨迹窗口与多轮监督

<!-- NUM_FRAMES、NUM_FUTURE_STEPS、NUM_OVERLAP 与 loss mask。 -->

## 4. 历史记忆

<!-- per-frame、GTC、Segment-GTC 的输入输出和选择建议。 -->

## 5. SatNav 地图记忆

<!-- global/local explored map、适用范围、与 history 的替换关系。 -->

## 6. Embedding enhancement

<!-- none、pose、posefilm、uav；严格互斥和 Stage-A/Stage-B 关系。 -->

## 7. 支持矩阵

<!-- model family × backend × memory × enhancement。 -->

## 8. 当前限制

<!-- 只列代码中真实存在、会影响复现或扩展的限制。 -->
