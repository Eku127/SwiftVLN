# Pose Embedding

## 概述

Pose Embedding 将智能体的位姿信息（位置 + 朝向）注入到 ViT 视觉特征中，使模型在处理每一帧图像时能感知自身在空间中的位置和方向。

该模块作为互斥 embedding mode 之一，在 ViT 编码之后、历史压缩之前对所有图像特征进行增强。

统一入口 `embedding_mode` 只能四选一：

| mode | 行为 |
| --- | --- |
| `none` | 不启用 embedding enhancement |
| `pose` | PoseEmbedding + additive 融合 |
| `posefilm` | PoseEmbedding + FiLM 融合 |
| `uav` | Stage-A UAV adapter |

不存在 pose 与 UAV 叠加模式。

## Pose 表示

每帧图像对应一个 4 维 pose 向量：

```
[delta_forward, delta_right, sin(delta_heading), cos(delta_heading)]
```

- `delta_forward`：相对起点的前进方向位移（米）
- `delta_right`：相对起点的右侧方向位移（米）
- `sin/cos(delta_heading)`：相对起点的朝向变化

所有位移均在 **ego-frame**（以 episode 起点为原点、起始朝向为前方）下计算。

## 数据来源

- **训练时**：从 `annotations.json` 中的 action 序列通过 action 积分重建 pose（`reconstruct_pose_from_actions`），与 SatNav 测试中的 `_expected_pose_from_actions` 逻辑一致。
- **推理时**：同样使用 action 积分，保证与训练完全一致。

Action 到运动的映射：

| 环境     | forward step | turn angle |
|----------|-------------|------------|
| SatNav   | 10 m        | 15°        |
| Habitat  | 0.25 m      | 30°        |

## 融合方式

Pose 融合方式直接由 `embedding_mode` 决定，不再设置第二个 fusion 参数：

**Additive（默认）**

```
embed = embed + β × MLP(pose)
```

结构简单直接。

**FiLM**

```
[γ, bias] = MLP(pose)
embed = embed × (1 + β × γ) + β × bias
```

Feature-wise affine 调制，表达能力更强，但参数量翻倍（MLP 输出维度为 2D）。

## 归一化

位置分量（`delta_forward`, `delta_right`）使用 tanh 压缩：

```
normalized = tanh(value / norm_scale)
```

朝向分量（`sin`, `cos`）天然在 [-1, 1]，不做额外归一化。

`norm_scale` 决定了"半饱和距离"（tanh(1) ≈ 0.76）：

| norm_scale | 半饱和距离 | 饱和距离（>0.96） | 适用场景               |
|-----------|-----------|-----------------|----------------------|
| 100.0     | 100 m     | ~200 m          | SatNav 中短路径（默认） |
| 200.0     | 200 m     | ~400 m          | SatNav 长路径          |
| 50.0      | 50 m      | ~100 m          | Habitat              |

## 初始化策略

MLP 最后一层权重和偏置 **零初始化**，确保训练开始时 pose embedding 输出全零，模型行为与无 pose embedding 的 baseline 完全一致。

## 默认参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `embedding_mode` | `none` | `none|pose|posefilm|uav` 四选一 |
| `pose_norm_scale` | `100.0` | tanh 归一化的缩放因子 |
| `pose_dim` | `4` | pose 向量维度 |
| `pose_hidden_dim` | `256` | MLP 隐藏层维度 |
| `pose_beta` | `1.0` | 融合缩放系数 |

## 涉及文件

| 文件 | 职责 |
|------|------|
| `common/embedding_enhancement/pose_embed.py` | PoseEmbedding 模块（MLP + 融合） |
| `common/embedding_enhancement/pose_utils.py` | action 积分重建 pose |
| `common/embedding_enhancement/__init__.py` | pipeline 工厂函数 |
| `model/arguments.py` | 训练参数定义 |
| `model/model.py` | 模型加载时创建 enhancement container |
| `model/dataset.py` | 训练数据中重建 per-frame pose |
| `model/template.py` | 将 pose 传递到 enhancement module |
| `model/trainer.py` | 确保 enhancement 正确初始化 |
| `model/inference.py` | 推理时计算并传递 pose |

## 启用方式

通过 shell 入口选择其中一个 mode：

```bash
# additive pose
EMBEDDING_MODE=pose POSE_NORM_SCALE=100 \
  bash src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh

# FiLM pose
EMBEDDING_MODE=posefilm POSE_NORM_SCALE=100 \
  bash src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh
```

直接调用 Python 训练/评测 CLI 时使用 `--embedding_mode pose` 或
`--embedding_mode posefilm`。旧布尔参数会被拒绝，避免产生组合状态。

Pose embedding 的权重会自动随 checkpoint 保存和恢复，无需额外配置。
