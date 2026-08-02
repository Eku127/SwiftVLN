# S2R 与 Rebuttal 当前状态

最后核验：2026-08-02

## S2R Stage-A 正式产物

当前唯一保留的正式训练目录：

../output/s2r/s2r-swiftvln-baseline-gta-dedup-10ep-bs64-lr1e-4-20260727-160227/

核心状态：

| 项目 | 当前值 |
| --- | --- |
| 权重 | best.pt，423,878,141 bytes（约 404MiB） |
| 训练状态 | completed |
| 总步数 | 2610 / 2610 |
| 训练耗时 | 42m 7s |
| Epochs | 10 |
| Learning rate | 1e-4 |
| Temperature | 0.07 |
| Projection dim | 512 |
| Teacher | model_zoo 中的 SwiftVLN 3B overlap0 per-frame baseline |
| Stage-B loader check | PASS |

正式目录只保留 best.pt、train_args.json、progress.json、metrics.jsonl、独立评测日志和 Stage-B loader 检查日志。smoke 输出、latest.pt 与重复 step checkpoint 已删除。

## 独立检索评测

评测集共 2634 条。百分数由 progress.json 的最终指标换算；paired cosine 保持原始小数。

| 数据源 | 样本 | U2S R@1 | S2U R@1 | U2S R@5 | S2U R@5 | U2S R@10 | S2U R@10 | Paired cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 2634 | 37.51 | 38.27 | 72.10 | 75.02 | 79.16 | 82.69 | 0.6774 |
| DenseUAV | 546 | 55.13 | 52.93 | 96.70 | 97.44 | 98.35 | 98.53 | 0.7756 |
| GTA | 544 | 60.29 | 56.62 | 93.75 | 90.81 | 98.16 | 96.88 | 0.7718 |
| SUES | 150 | 30.00 | 32.00 | 92.67 | 90.00 | 99.33 | 100.00 | 0.6847 |
| UAV-VisLoc | 1394 | 22.60 | 26.47 | 51.87 | 58.82 | 62.34 | 69.51 | 0.6013 |

当前主要短板是 UAV-VisLoc；它占验证集一半以上，同时 R@1 明显低于 DenseUAV/GTA。后续优化应先区分数据域差异与数据量权重，而不是只看 overall。

## 可复现性边界

正式 best.pt 可加载，独立评测与 Stage-B loader 检查均已通过，但该次训练不是“原样可重跑”状态：

- train_args.json 指向 runtime/s2r/manifests/manifest_v2_gta_dedup.jsonl。
- 该 v2 manifest 当前已经不存在。
- 现存 canonical manifest 是 runtime/s2r/manifests/manifest_v1.jsonl，共 19,365 行；它覆盖 DenseUAV、GTA、SUES 与 UAV-VisLoc，但不能假定与正式训练使用的 v2 文件字节或采样完全一致。
- 如需精确复训，应从 SatDronePair 原始数据重新生成 v2，验证去重与 train/val 划分，再启动训练。
- 日志中的旧 ms-swift-lateset 路径是当时运行 provenance，不是当前依赖路径；当前规范路径是 /mnt/data1/home/jiangjiajun/workspace/ms-swift。

因此，当前保留等级是“权重可评测、可接 Stage-B；训练过程不可直接从现有 manifest 精确复刻”。

## S2R 当前代码入口

| 用途 | 路径 |
| --- | --- |
| Manifest 构建 | ../src/swiftvln/s2r/scripts/build_manifest.py |
| Dataset/split | ../src/swiftvln/s2r/dataset.py、../src/swiftvln/s2r/split.py |
| Stage-A 模型 | ../src/swiftvln/s2r/model.py |
| 训练 | ../src/swiftvln/s2r/trainer.py |
| 评测 | ../src/swiftvln/s2r/eval.py |
| 数据生产 | ../src/swiftvln/s2r/data_generation/ |
| 数据生产说明 | ../src/swiftvln/s2r/data_generation/README.md |

## Rebuttal episodes

目录：../output/rebuttal/，当前大小 292M。

共有 12 份有效 JSON，全部通过 JSON 解析检查；各文件中的 episode_id 记录合计 425 条。该合计包含不同汇总/标注版本中的重复语义记录，不表示 425 个全局唯一 episode。

| 文件 | episode 记录数 |
| --- | ---: |
| Birmingham-1/VLN_episodes.json | 30 |
| Cambridge-1/VLN_episodes.json | 42 |
| HUGECity-1/VLN_episodes.json | 4 |
| HUGECity-1/VLN_episodes_origin.json | 2 |
| HUGEOffice-1/VLN_episodes.json | 14 |
| HUGEOffice-1/VLN_episodes_origin.json | 7 |
| HUGERoad-1/VLN_episodes.json | 4 |
| HUGERoad-1/VLN_episodes_origin.json | 2 |
| VLN_Episodes_3DGS.json | 22 |
| VLN_Episodes_PCD.json | 72 |
| VLN_episodes_human.json | 113 |
| VLN_episodes_llm.json | 113 |

与 episode 直接相关的辅助资产也保留：

- 5 份 GeoTIFF 场景图；
- Birmingham/Cambridge 共 36 张轨迹预览图；
- 4 份点云 XY/经纬度转换说明；
- GeoTIFF_Episode_Maker.zip 数据生成工具。

重复的 Birmingham/Cambridge 压缩包、Birmingham-old、评测报告、敏感性原始运行结果和无关图片已删除，不再作为当前资料引用。

## 相关场景数据

| 路径 | 大小 | 内容 |
| --- | ---: | --- |
| ../output/PCD-data/ | 42G | Birmingham/Cambridge PCD 与渲染缓存 |
| ../output/3DGS-data/ | 94G | 3DGS data_3d、CityGaussianV2 与缓存 |

这两处是 episode 运行/核验所需的场景数据，不属于历史训练输出。删除前必须重新确认是否仍需 PCD/3DGS rebuttal。

## 更新规则

- S2R 重新训练、替换 best.pt 或恢复 v2 manifest 后，更新正式产物、指标和可复现性结论。
- Rebuttal episode 增删后，重新校验 12 类文件统计、JSON 可解析性以及场景资产数量。
- 不在 reports/ 复制权重、manifest、episode JSON 或评测日志；本报告只维护当前摘要。
