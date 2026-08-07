# SwiftVLN

SwiftVLN 是当前 `swiftvln` 主线模型的视觉语言导航训练与评测代码仓库。

## 文档入口

- [环境安装](docs/installation.md)
- [S2R SatDronePair 数据生产](src/swiftvln/s2r/data_generation/README.md)

`reports/`、`tests/`、`.codex/`、`AGENTS.md` 以及 `.local/` 是维护者本地内容，
不会进入公开仓库。

## 本地路径配置

公开脚本保留了可直接修改的相对路径/模型 ID 默认值，同时支持被 Git 忽略的本地
覆盖层。推荐复制模板后填写本机的模型、SatNav 数据、Conda 和缓存路径：

```bash
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

训练和评测入口会自动加载 `.local/env.sh`。也可以不创建该文件，改为在 shell 中
导出同名环境变量，或直接修改脚本/配置文件里的公开默认值。四个 baseline 还各自
提供 `baseline/<name>/local.env.example`；复制到对应的
`baseline/<name>/.local/env.sh` 后，会在共享配置之后自动加载。

## 环境安装

训练与评测环境的完整安装说明统一维护在
[docs/installation.md](docs/installation.md)。
