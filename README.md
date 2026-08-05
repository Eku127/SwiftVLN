# SwiftVLN

SwiftVLN 是当前 `swiftvln` 主线模型的视觉语言导航训练与评测代码仓库。

## 文档入口

- [环境安装](docs/installation.md)
- [当前状态与结果报告](reports/README.md)
- [S2R SatDronePair 数据生产](src/swiftvln/s2r/data_generation/README.md)

## 环境安装

训练与评测环境的完整安装说明统一维护在
[docs/installation.md](docs/installation.md)。

## 推理 wheel

使用独立构建入口生成用于推理/评测部署的 wheel：

```bash
python packaging/build_inference_wheel.py --output-dir dist/inference
```

`swiftvln-inference` 保留 SatNav、Habitat、视频和 UAV adapter 运行时，但明确排除
`swiftvln/s2r/data_generation/`；完整源码仓库和常规开发安装仍保留数据生产工具。
