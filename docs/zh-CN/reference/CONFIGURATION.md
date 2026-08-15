# 配置参考

> 框架页：本页是事实型参考，不写操作教程。字段以 `swiftvln.experiment`、训练参数和评测
> 参数的当前实现为准。

## 1. 配置入口与优先级

<!-- 显式环境变量 / CLI、.local/env.sh、脚本默认值、YAML。 -->

## 2. 模型与环境

<!-- MODEL_FAMILY、MODEL_PATH、VLN_ENV_TYPE / ENV_TYPE。 -->

## 3. 轨迹窗口

<!-- NUM_FRAMES、NUM_FUTURE_STEPS、NUM_OVERLAP。 -->

## 4. 历史记忆

<!-- per-frame / GTC / Segment-GTC 参数表。 -->

## 5. 地图记忆

<!-- 尺寸、render、mask、cache。 -->

## 6. Embedding enhancement

<!-- none / pose / posefilm / uav 参数表。 -->

## 7. 训练执行

<!-- GPU、batch、optimizer、DeepSpeed、resume、output。 -->

## 8. 评测执行

<!-- split、GPU、episode limit、video、resume、output。 -->

## 9. 跨字段约束

<!-- 直接整理 SwiftVLNExperimentSpec 的可判定规则。 -->
