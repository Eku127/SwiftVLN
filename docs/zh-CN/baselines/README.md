# Baseline 导航

> 框架页：本页只做 baseline 选择与入口导航。每个 baseline 的依赖、模型和命令由其
> 自有 README 维护，避免在两处复制。

## 1. Baseline 范围

| Baseline | 运行环境 | 代码位置 | 完整文档 |
| --- | --- | --- | --- |
| StreamVLN | SatNav | `baseline/streamvln/` | [StreamVLN](../../../baseline/streamvln/README.md) |
| NaVILA | SatNav | `baseline/navila/` | [NaVILA](../../../baseline/navila/README.md) |
| Uni-NaVid | SatNav | `baseline/uninavid/` | [Uni-NaVid](../../../baseline/uninavid/README.md) |
| OpenFly | SatNav | `baseline/openfly/` | [OpenFly](../../../baseline/openfly/README.md) |

## 2. 选择 baseline

<!-- 参数规模、训练起点、外部源码、显存与依赖矩阵。 -->

## 3. 统一数据与评测边界

<!-- 哪些输入和指标可比较，哪些 trainer/checkpoint 由模型自有实现负责。 -->

## 4. 复现顺序

<!-- 独立环境 -> model/data preflight -> smoke train -> checkpoint reload -> eval。 -->

## 5. 代码来源与许可

<!-- 汇总上游仓库与固定 revision，详细声明留在各 baseline README。 -->
