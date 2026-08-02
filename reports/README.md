# SwiftVLN 当前报告

最后核验：2026-08-02

本目录只维护能够描述仓库当前状态的报告，不再作为历史实验归档。旧的按日期报告、单次故障记录、迁移过程记录、原始 CSV/JSON 导出和重复工作簿已移除；需要追溯时使用 Git 历史，实验指标则以 model zoo 中的原始评测 JSON 为准。

## 报告入口

| 文档 | 内容 |
| --- | --- |
| [current_state.md](current_state.md) | 当前代码入口、环境、默认数据与持久化产物 |
| [model_zoo.md](model_zoo.md) | 22 个保留模型、评测完整性和当前核心指标 |
| [s2r_rebuttal.md](s2r_rebuttal.md) | S2R 正式权重、检索指标、可复现性限制与 rebuttal episode 数据 |

## 权威来源

报告是便于阅读的汇总，不替代以下原始来源：

- 仓库路径、脚本和运行约定：[../.codex/CODEX_CONTEXT.md](../.codex/CODEX_CONTEXT.md)
- 模型与评测结果：../output/model_zoo/
- S2R 正式产物：../output/s2r/ 与 ../runtime/s2r/
- Rebuttal episodes：../output/rebuttal/
- 安装流程：[../docs/installation.md](../docs/installation.md)

## 维护规则

1. 内容变化时原位更新这三份报告，不新增带日期后缀的快照。
2. reports/ 不保存训练日志、checkpoint、原始评测结果副本或临时分析表。
3. model zoo 有增删时，同步更新 model_zoo.md；精确数值必须从 evaluation_summary.json 重新读取。
4. S2R 权重、manifest 或 episode 数据变化时，同步更新 s2r_rebuttal.md。
5. 目录、默认数据、环境或主流程变化时，同步更新 current_state.md 和 .codex/CODEX_CONTEXT.md。
