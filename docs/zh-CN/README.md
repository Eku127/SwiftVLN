# SwiftVLN 中文文档

本页按照使用目标组织 SwiftVLN 文档。首次使用从“安装与首次运行”开始；已经具备
环境和数据时，可直接进入训练或评测。

> 当前为文档框架草案。各页面先固定职责和章节边界，正文将在命令与输出完成验证后
> 逐篇补齐。

## 1. 安装与首次运行

1. [安装](getting-started/INSTALLATION.md)：分别准备训练环境与评测环境；
2. [模型与 Checkpoint](getting-started/CHECKPOINTS.md)：下载 Qwen 基础模型和 SwiftVLN checkpoint；
3. [SatNav 训练数据](data/TRAINING_DATA_SATNAV.md)：准备 Episode、GeoTIFF 和离线轨迹；
4. [Habitat 训练数据](data/TRAINING_DATA_HABITAT.md)：准备 R2R、RxR 和 EnvDrop 轨迹；
5. [评测数据准备](data/EVALUATION_DATA.md)：准备 SatNav 与 Habitat 在线评测数据。

## 2. 训练

- [模型与 Checkpoint](getting-started/CHECKPOINTS.md)：准备基础模型或已训练的 SatNav 模型；
- [SwiftVLN 训练](training/README.md)：主线模型的训练配置、完整训练和恢复；
- [Memory 训练配置](training/MEMORY.md)：配置历史帧采样、输入增强与长期 Memory 压缩；
- [S2R Stage-A](training/S2R_STAGE_A.md)：SatDronePair manifest、adapter 训练和
  retrieval 评测；
- [SatDronePair 数据生产](data/SATDRONEPAIR.md)：生成 S2R Stage-A 使用的 UAV–Satellite
  配对数据；
- [SatNav 训练数据](data/TRAINING_DATA_SATNAV.md)：生成并校验 SatNav 离线轨迹；
- [Habitat 训练数据](data/TRAINING_DATA_HABITAT.md)：生成并校验 R2R、RxR 和 EnvDrop 轨迹。

## 3. 评测

- [模型与 Checkpoint](getting-started/CHECKPOINTS.md)：下载 SatNav 默认模型与 ablation checkpoint；
- [评测数据准备](data/EVALUATION_DATA.md)：准备 SatNav Episode、GeoTIFF、R2R 和 MP3D；
- [SwiftVLN 评测](evaluation/README.md)：SatNav/Habitat、单卡/多卡、resume、视频和指标。

## 4. 开发与扩展

- [代码架构](development/ARCHITECTURE.md)：训练链路、评测链路、模块边界和依赖方向；
- [扩展 SwiftVLN](development/EXTENDING.md)：新增环境 backend、history processor、
  embedding enhancement 或模型族。

## 5. 按目标查找

| 目标 | 文档 |
| --- | --- |
| 从源码安装 | [安装](getting-started/INSTALLATION.md) |
| 下载基础模型或 checkpoint | [模型与 Checkpoint](getting-started/CHECKPOINTS.md) |
| 准备 SatNav 训练轨迹 | [SatNav 训练数据](data/TRAINING_DATA_SATNAV.md) |
| 准备 Habitat 训练轨迹 | [Habitat 训练数据](data/TRAINING_DATA_HABITAT.md) |
| 准备 SatNav/Habitat 评测数据 | [评测数据准备](data/EVALUATION_DATA.md) |
| 训练 SwiftVLN | [SwiftVLN 训练](training/README.md) |
| 配置 Memory 实验 | [Memory 训练配置](training/MEMORY.md) |
| 准备 S2R SatDronePair 数据 | [SatDronePair 数据生产](data/SATDRONEPAIR.md) |
| 训练 S2R adapter | [S2R Stage-A](training/S2R_STAGE_A.md) |
| 评测 checkpoint | [SwiftVLN 评测](evaluation/README.md) |
| 修改核心代码 | [代码架构](development/ARCHITECTURE.md) |

## 6. 文档约定

- 命令默认从 SwiftVLN 仓库根目录运行；
- `/path/to/...` 表示需要替换的本机路径；
- 机器路径、模型、数据和凭据不写入公共配置；
- smoke 只验证链路，不用于报告模型性能；
- 所有命令同时给出预期输出或可验证的完成条件。
