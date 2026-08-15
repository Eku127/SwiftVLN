# SwiftVLN

SwiftVLN 是面向视觉语言导航的训练与在线评测代码库，支持 SatNav 与 Habitat
环境、Qwen2.5-VL 与 Qwen3-VL 模型族，以及多种历史记忆和视觉 embedding 增强方式。

> 中文开源文档正在重写。当前目录已经建立新的信息架构，正文将按页面逐步补齐。

## 文档

- [中文文档导航](docs/zh-CN/README.md)
- [安装](docs/zh-CN/getting-started/INSTALLATION.md)
- [训练](docs/zh-CN/training/README.md)
- [评测](docs/zh-CN/evaluation/README.md)

旧文档保存在 [`docs/legacy/`](docs/legacy/)，仅作为重写时的事实线索，不再作为
公开使用入口。

## 本地配置

```bash
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

训练和评测脚本会读取 `.local/env.sh`。模型、数据、外部仓库、缓存与输出路径均应
保存在该本地配置或调用命令的环境变量中。
