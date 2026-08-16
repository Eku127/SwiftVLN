# SwiftVLN

[English](README.md) | 简体中文

SwiftVLN 是面向视觉语言导航研究的训练与在线评测框架，支持无人机 SatNav 与室内
Habitat 环境。仓库提供从轨迹数据生产、监督微调、Memory 实验到分布式在线评测的完整
研究链路。

## 主要功能

- 支持 Qwen2.5-VL 与 Qwen3-VL 模型族；
- 使用统一入口训练 SatNav 与 Habitat 离线 expert trajectories；
- 支持 per-frame、Map、GTC 与 Segment-GTC 等 Memory 设计；
- 支持历史帧采样、初始观测、相对位姿与视觉 embedding enhancement；
- 支持 SatNav / Habitat trajectory generation 与数据校验；
- 支持 SatNav 单卡、多卡、断点恢复与视频可视化评测；

## 开始使用

### 1. 安装

SwiftVLN 分别使用 `swiftvln-train` 和 `swiftvln-eval` 两套 Conda 环境。首先获取源码：

```bash
git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
```

根据训练或评测目标拉取对应的 third-party 子模块，并按照[安装文档](docs/zh-CN/getting-started/INSTALLATION.md)
创建环境。

### 2. 准备模型与数据

- [模型与 Checkpoint](docs/zh-CN/getting-started/CHECKPOINTS.md)
- [SatNav 训练数据](docs/zh-CN/data/TRAINING_DATA_SATNAV.md)
- [Habitat 训练数据](docs/zh-CN/data/TRAINING_DATA_HABITAT.md)
- [SatNav / Habitat 评测数据](docs/zh-CN/data/EVALUATION_DATA.md)

将本机模型、数据和外部依赖路径写入 `.local/env.sh`：

```bash
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

### 3. 训练与评测

训练 SatNav 或 Habitat：

```bash
bash scripts/train/train_swiftvln_qwen_vl.sh
```

按照完整模型名称评测 checkpoint：

```bash
bash scripts/eval/eval_by_name.sh "${MODEL_NAME}"
```

环境、数据、GPU 与 Memory 参数配置见 [SwiftVLN 训练](docs/zh-CN/training/README.md)和
[SwiftVLN 评测](docs/zh-CN/evaluation/README.md)。

## Model Zoo

推荐使用默认 SatNav 模型开始评测或后续训练：

- [SwiftVLN SatNav 3B reference](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed)

完整的 SatNav Memory ablation 模型收录于
[SwiftVLN SatNav Ablation Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo)。

Backbone 对比模型：

| Backbone | 模型 |
| --- | --- |
| Qwen2.5-VL 7B | [SwiftVLN SatNav Qwen2.5-VL 7B](https://huggingface.co/Eku127/swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) |
| Qwen3-VL 2B | [SwiftVLN SatNav Qwen3-VL 2B](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) |
| Qwen3-VL 8B | [SwiftVLN SatNav Qwen3-VL 8B](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) |

模型下载方式与名称解析规则见[模型与 Checkpoint](docs/zh-CN/getting-started/CHECKPOINTS.md)。

## 文档

| 主题 | 文档 |
| --- | --- |
| 文档导航 | [SwiftVLN 中文文档](docs/zh-CN/README.md) |
| 安装 | [安装](docs/zh-CN/getting-started/INSTALLATION.md) |
| 模型下载 | [模型与 Checkpoint](docs/zh-CN/getting-started/CHECKPOINTS.md) |
| 训练 | [SwiftVLN 训练](docs/zh-CN/training/README.md) |
| Memory | [Memory 训练配置](docs/zh-CN/training/MEMORY.md) |
| Satellite-to-UAV | [Satellite-to-UAV Stage-A](docs/zh-CN/training/S2R_STAGE_A.md) |
| 评测 | [SwiftVLN 评测](docs/zh-CN/evaluation/README.md) |
| 代码架构 | [代码架构](docs/zh-CN/development/ARCHITECTURE.md) |
| 扩展开发 | [扩展 SwiftVLN](docs/zh-CN/development/EXTENDING.md) |

## 仓库结构

```text
SwiftVLN/
├── src/swiftvln/     # 训练、模型、Memory、Backend 与在线评测
├── scripts/          # 训练、评测与任务队列入口
├── tools/s2r/        # SatDronePair 与 Satellite-to-UAV Stage-A 工具
├── environments/     # 训练与评测 Conda 环境
├── third_party/      # 固定版本的外部源码
└── docs/             # zh-CN / en-US 使用与开发文档
```

核心模块与扩展边界见[代码架构](docs/zh-CN/development/ARCHITECTURE.md)。

## 相关项目

- [SatNav](https://github.com/Eku127/SatNav)：卫星图像视觉语言导航环境、数据集与评测工具；
- [ms-swift](https://github.com/modelscope/ms-swift)：SwiftVLN 使用的多模态模型训练框架。
