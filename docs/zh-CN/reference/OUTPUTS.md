# 输出格式

> 框架页：本页记录可被脚本或分析程序依赖的文件布局与字段，不解释训练或评测方法。

## 1. 训练输出目录

<!-- experiment/version/checkpoint、train_metadata、日志。 -->

## 2. SwiftVLN checkpoint

<!-- ms-swift/HF 目录、embedding enhancement 权重与恢复要求。 -->

## 3. 评测输出目录

<!-- result.jsonl、all_results.jsonl、evaluation_summary.json、timing、video、.dist_sync。 -->

## 4. `result.jsonl`

<!-- 逐 Episode 字段与 stable key。 -->

## 5. `evaluation_summary.json`

<!-- 聚合指标和配置字段。 -->

## 6. 分布式完成标记

<!-- rank_<n>.done.json；生命周期和超时。 -->

## 7. Resume 与兼容性

<!-- append-only 日志、非法/截断行行为、不同配置不得混用。 -->

## 8. S2R Stage-A 输出

<!-- checkpoint、adapter_kwargs/state_dict、retrieval metrics。 -->
