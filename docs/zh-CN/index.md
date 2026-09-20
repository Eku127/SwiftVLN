# SwiftVLN 文档

SwiftVLN 将轨迹训练、记忆机制实验与在线评测整合到同一框架中，支持 Qwen2.5-VL 和 Qwen3-VL，可用于 SatNav 航空导航与 Habitat 室内导航。

<p align="center">
  <img src="../assets/readme/swiftvln-framework.png" width="75%" alt="SwiftVLN 结合历史记忆、滑动对话窗口与语言指令预测导航动作。">
</p>

## 从这里开始

| 你的目标 | 使用指南 |
| --- | --- |
| 理解实现原理 | [双层记忆与滑窗](concepts/PIPELINE.md) · [历史记忆压缩](concepts/MEMORY.md) · [地图记忆](concepts/MAP_MEMORY.md) · [输入增强](concepts/AUGMENTATION.md) |
| 评测已发布模型 | [安装](getting-started/INSTALLATION.md) · [模型与 Checkpoint](getting-started/CHECKPOINTS.md) · [评测](evaluation/README.md) |
| 训练导航策略 | [SatNav 数据](data/TRAINING_DATA_SATNAV.md) · [Habitat 数据](data/TRAINING_DATA_HABITAT.md) · [训练](training/README.md) |
| 比较记忆机制 | [记忆配置](training/MEMORY.md) |
| 迁移到无人机观测 | [SatDronePair](data/SATDRONEPAIR.md) · [Stage-A 训练](training/S2R_STAGE_A.md) |
| 扩展框架 | [代码架构](development/ARCHITECTURE.md) · [扩展指南](development/EXTENDING.md) |

## 训练与评测流程

先安装环境、下载基础模型或导航 checkpoint，再配置本地数据路径。训练使用离线 RGB 观测与专家动作，评测在 SatNav 或 Habitat Episode 中执行在线导航。通过同一模型名称，将训练时的记忆和窗口设置带入评测。

[使用流程](README.md)串联了安装、数据准备、训练、评测与开发文档。参考配置采用 32 个动作的窗口、每次预测 4 个动作，以及 8 个均匀采样的历史观测。

## 项目资源

- [论文](https://openreview.net/forum?id=hOEniyN6hl)：*SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery*。
- [SwiftVLN Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo)：参考模型与记忆消融实验 checkpoint。
- [SatNav Wiki](https://eku127.github.io/SatNav/wiki/)：模拟器、Episode、卫星场景与基线文档。
- [源代码](https://github.com/Eku127/SwiftVLN)：模型、脚本、配置与文档源文件。

```{toctree}
:hidden:
:maxdepth: 2
:caption: 快速开始

安装 <getting-started/INSTALLATION>
模型与 Checkpoint <getting-started/CHECKPOINTS>
使用流程 <README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 记忆实现原理

双层记忆与滑动窗口 <concepts/PIPELINE>
历史记忆与 token 压缩 <concepts/MEMORY>
地图记忆 <concepts/MAP_MEMORY>
输入增强与 UAV 适配 <concepts/AUGMENTATION>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 数据准备

SatNav 训练轨迹 <data/TRAINING_DATA_SATNAV>
Habitat 训练轨迹 <data/TRAINING_DATA_HABITAT>
评测数据 <data/EVALUATION_DATA>
SatDronePair <data/SATDRONEPAIR>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 训练与记忆

模型训练 <training/README>
记忆配置 <training/MEMORY>
Satellite-to-UAV Stage-A <training/S2R_STAGE_A>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 评测

在线评测 <evaluation/README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 开发

代码架构 <development/ARCHITECTURE>
扩展 SwiftVLN <development/EXTENDING>
```
