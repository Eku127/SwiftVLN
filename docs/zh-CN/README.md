# SwiftVLN 中文文档

本页按照使用目标组织 SwiftVLN 文档。首次使用从“安装与首次运行”开始；已经具备
环境和数据时，可直接进入训练或评测。

> 当前为文档框架草案。各页面先固定职责和章节边界，正文将在命令与输出完成验证后
> 逐篇补齐。

## 1. 安装与首次运行

1. [安装](getting-started/INSTALLATION.md)：分别准备训练环境与评测环境；
2. [快速开始](getting-started/QUICKSTART.md)：完成配置检查、最小训练和单 Episode 评测；
3. [SatNav 训练数据](data/TRAINING_DATA_SATNAV.md)：准备 Episode、GeoTIFF 和离线轨迹；
4. [Habitat 训练数据](data/TRAINING_DATA_HABITAT.md)：准备 R2R、RxR 和 EnvDrop 轨迹。

## 2. 理解 SwiftVLN

- [方法概览](concepts/METHOD.md)：模型输入输出、轨迹窗口、历史记忆、地图记忆和
  embedding enhancement；
- [配置参考](reference/CONFIGURATION.md)：SwiftVLN 专有参数、默认值、约束与配置优先级；
- [实验命名](reference/EXPERIMENT_NAMING.md)：训练输出名称的组成、解析与复用规则。

## 3. 训练

- [SwiftVLN 训练](training/README.md)：主线模型的 dry run、smoke、完整训练和恢复；
- [S2R Stage-A](training/S2R_STAGE_A.md)：SatDronePair 数据、adapter 训练、retrieval 评测和
  Stage-B 接入；
- [SatNav 训练数据](data/TRAINING_DATA_SATNAV.md)：生成并校验 SatNav 离线轨迹；
- [Habitat 训练数据](data/TRAINING_DATA_HABITAT.md)：生成并校验 R2R、RxR 和 EnvDrop 轨迹。

## 4. 评测

- [SwiftVLN 评测](evaluation/README.md)：SatNav/Habitat、单卡/多卡、resume、视频和指标；
- [输出格式](reference/OUTPUTS.md)：训练 checkpoint、逐 Episode JSONL、汇总文件与恢复语义；
- [实验命名](reference/EXPERIMENT_NAMING.md)：从模型名恢复训练配置。

## 5. Baseline

[Baseline 导航](baselines/README.md)汇总 StreamVLN、NaVILA、Uni-NaVid 与 OpenFly 的
独立环境、模型来源、训练和 SatNav 评测入口。

## 6. 开发与扩展

- [代码架构](development/ARCHITECTURE.md)：训练链路、评测链路、模块边界和依赖方向；
- [扩展 SwiftVLN](development/EXTENDING.md)：新增环境 backend、history processor、
  embedding enhancement 或模型族。

## 7. 按目标查找

| 目标 | 文档 |
| --- | --- |
| 从源码安装 | [安装](getting-started/INSTALLATION.md) |
| 跑通最小链路 | [快速开始](getting-started/QUICKSTART.md) |
| 准备 SatNav 训练轨迹 | [SatNav 训练数据](data/TRAINING_DATA_SATNAV.md) |
| 准备 Habitat 训练轨迹 | [Habitat 训练数据](data/TRAINING_DATA_HABITAT.md) |
| 训练 SwiftVLN | [SwiftVLN 训练](training/README.md) |
| 训练 S2R adapter | [S2R Stage-A](training/S2R_STAGE_A.md) |
| 评测 checkpoint | [SwiftVLN 评测](evaluation/README.md) |
| 查询参数约束 | [配置参考](reference/CONFIGURATION.md) |
| 查询结果字段 | [输出格式](reference/OUTPUTS.md) |
| 复现其他 VLM | [Baseline 导航](baselines/README.md) |
| 修改核心代码 | [代码架构](development/ARCHITECTURE.md) |

## 8. 文档约定

- 命令默认从 SwiftVLN 仓库根目录运行；
- `/path/to/...` 表示需要替换的本机路径；
- 机器路径、模型、数据和凭据不写入公共配置；
- smoke 只验证链路，不用于报告模型性能；
- 操作页只保留完成任务所需步骤，原理集中在“方法概览”，字段集中在“参考”；
- 所有命令同时给出预期输出或可验证的完成条件。
