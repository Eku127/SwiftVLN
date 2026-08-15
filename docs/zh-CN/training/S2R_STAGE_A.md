# S2R Stage-A

> 框架页：本页覆盖从 SatDronePair 数据到可供 SwiftVLN `uav` 模式加载的 adapter
> checkpoint。

## 1. 流程与产物

<!-- raw datasets -> normalized pairs -> manifest.jsonl -> train -> retrieval eval -> checkpoint。 -->

## 2. 安装依赖

<!-- s2r-data extra 与训练环境复用边界。 -->

## 3. 生成 SatDronePair

<!-- 链接或收敛 tools/s2r/data_generation/README.md 的四数据源入口。 -->

## 4. 构建 manifest

<!-- build_manifest.py 参数、split 和输出 schema。 -->

## 5. 训练 adapter

<!-- train_s2r_stagea.sh / tools.s2r.trainer；smoke 与完整训练。 -->

## 6. Retrieval 评测

<!-- split、指标与 checkpoint 加载。 -->

## 7. 接入 SwiftVLN Stage-B

<!-- EMBEDDING_MODE=uav、UAV_ADAPTER_PATH、checkpoint 兼容性。 -->

## 8. 输出与恢复

<!-- checkpoints/step_*.pt、元数据、resume。 -->

## 9. 常见问题

<!-- 缺失图像、维度不匹配、数据泄漏、DDP 与显存。 -->
