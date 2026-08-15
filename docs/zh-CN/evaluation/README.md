# SwiftVLN 评测

> 框架页：本页从“已有 checkpoint、Episode 和场景”开始，说明在线评测任务。

## 1. 评测流程

<!-- model-name parse -> runner -> inference session -> backend -> recorder。 -->

## 2. 准备 checkpoint 与数据

<!-- MODEL_PATH、SatNav/Habitat 的 episode 与 scene。 -->

## 3. 配置检查

<!-- CHECK_ONLY；训练配置与评测配置一致性。 -->

## 4. 单 Episode smoke

<!-- SatNav 与 Habitat 各一个最小命令；预期 JSONL。 -->

## 5. 完整评测

<!-- split、GPU、输出目录和正式运行条件。 -->

## 6. 多 GPU 分片

<!-- scene 排序、global round-robin、rank markers。 -->

## 7. Resume 与去重

<!-- result.jsonl 的恢复键、AUTO_RESUME_EVAL、改变实验条件时的边界。 -->

## 8. 指标与视频

<!-- SR、SPL、OS、NE、steps；SAVE_VIDEO / VIDEO_COMPRESSION。 -->

## 9. 常见问题

<!-- 模型名解析、checkpoint 定位、EGL、rank 同步、动作窗口、结果数。 -->
