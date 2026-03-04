# Pixel Embedding

## 概述

Pixel Embedding（MLP_xy）将像素级空间坐标信息注入到 ViT 视觉特征中。ViT 本身使用固定的 2D RoPE 编码位置，但这些信息在 attention 层间逐渐稀释。Pixel Embedding 通过在 ViT 输出特征上直接叠加可学习的坐标编码，为后续 LLM 提供更强的空间定位能力。

该模块是 `EmbeddingEnhancementPipeline` 的一个插件，在 ViT 编码之后、历史压缩之前应用。

## 工作原理

### 坐标编码

对 ViT 输出的 H×W 特征网格，生成归一化到 [-1, 1] 的 (x, y) 坐标，然后进行 Fourier 编码：

```
FourierEncode(x, y) = [x, y, sin(π·x), cos(π·x), sin(2π·x), cos(2π·x), ..., 同理 y]
```

频率为 `π × 2^k`（k = 0, 1, ..., L-1），共 L = `num_bands` 个频带，编码维度为 `2 + 4 × num_bands`。

### 融合方式

采用 additive 融合：

```
V_aug = V + β × MLP(FourierEncode(x, y))
```

MLP 结构为 `Linear → GELU → Linear`，将 Fourier 编码映射到与 ViT 特征相同的维度。

### 缓存机制

坐标张量按 (H, W) 缓存，避免对相同分辨率的图像重复计算 Fourier 编码。

### 时序处理

当输入包含多帧（N = t × H × W）时，对每帧应用相同的坐标编码。

## 初始化策略

MLP 最后一层权重和偏置 **零初始化**，确保训练开始时 `MLP(·) = 0`，模型行为与无 Pixel Embedding 的 baseline 完全一致。

## 默认参数

| 参数               | 默认值  | 说明                                        |
|-------------------|--------|---------------------------------------------|
| `use_pixel_embed` | `False` | 是否启用 pixel embedding                     |
| `pixel_num_bands` | `6`    | Fourier 频带数，编码维度 = 2 + 4×6 = 26       |
| `pixel_hidden_dim`| `256`  | MLP 隐藏层维度                                |
| `pixel_beta`      | `1.0`  | 融合缩放系数                                  |
| `embed_dim`       | 自动   | 取自 `model.config.hidden_size`（如 1536）    |

## Fourier 编码维度说明

| num_bands | 编码维度 | 频率范围                |
|-----------|---------|------------------------|
| 4         | 18      | π, 2π, 4π, 8π          |
| 6（默认）  | 26      | π, 2π, 4π, 8π, 16π, 32π |
| 8         | 34      | π ~ 128π               |

更多频带 → 更精细的空间分辨率，但 MLP 输入维度也相应增大。

## 涉及文件

| 文件 | 职责 |
|------|------|
| `common/embedding_enhancement/pixel_embed.py` | PixelFeatureAugment 模块 |
| `common/embedding_enhancement/base.py` | BaseEmbeddingEnhancement 基类 |
| `common/embedding_enhancement/pipeline.py` | EmbeddingEnhancementPipeline 容器 |
| `common/embedding_enhancement/__init__.py` | pipeline 工厂函数 |
| `overlapvln/arguments.py` | 训练参数定义 |
| `overlapvln/model.py` | 模型加载时创建 pipeline |
| `overlapvln/template.py` | 在 `_post_encode` 中逐图像应用 pipeline |
| `overlapvln/trainer.py` | 确保 pipeline 正确初始化 |

## 启用方式

在训练脚本中添加：

```bash
--use_pixel_embed true
```

权重会自动随 checkpoint 保存和恢复，无需额外配置。

## 与 Pose Embedding 的关系

Pixel Embedding 和 Pose Embedding 可以同时启用，它们在 pipeline 中按注册顺序依次执行（先 pixel 后 pose）。两者作用于不同维度的信息：

- **Pixel Embedding**：图像内部的空间位置（每个 token 的 (x, y) 坐标不同）
- **Pose Embedding**：整张图像的全局位姿（同一张图的所有 token 共享同一个 pose 向量）
