# Third-party 源码

本目录包含 SwiftVLN 依赖的 Git Submodule：

```text
third_party/
├── ms-swift/
├── SatNav/
└── habitat-lab-0.2.4/
```

SwiftVLN 记录各子模块的仓库地址与 commit。按任务初始化所需源码：

```bash
# 训练
git submodule update --init third_party/ms-swift

# SatNav 评测
git submodule update --init third_party/ms-swift third_party/SatNav

# Habitat 评测
git submodule update --init \
  third_party/ms-swift \
  third_party/habitat-lab-0.2.4
```

完整安装顺序见[中文安装指南](../docs/zh-CN/getting-started/INSTALLATION.md)。
