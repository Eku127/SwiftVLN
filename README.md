# SwiftVLN

SwiftVLN 是从 `ms-swift/examples/vln` 迁移出来的独立仓库，当前代码已重构为包结构并移除 `examples/vln` 路径。

## 目录结构

- `src/swiftvln/model/`: 主线 OverlapVLN 模型实现
- `src/swiftvln/common/`: 通用训练与评估基础组件
- `src/swiftvln/configs/`: Habitat/SatNav 配置
- `src/swiftvln/scripts/`: 训练/评测队列与数据处理脚本
- `src/swiftvln/cli.py`: 统一 CLI 入口

说明：当前主线仓库只保留 `overlapvln`；其他历史模型实现已移除。基线实现仍保留在 `baseline/` 下。

## 安装

```bash
cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN
pip install -e .
```

## CLI 用法

```bash
# 训练
swiftvln train --model overlapvln -- --model_type overlapvln_qwen2_5_vl ...

# 评测
swiftvln eval --model overlapvln -- --model_path /path/to/checkpoint --env-type habitat ...

# 队列
swiftvln queue train
swiftvln queue eval
```

说明：`--` 后参数会原样透传给对应模型的训练/评测入口。

## 兼容脚本

仍可直接使用脚本入口（已切换到新路径）：

- `bash src/swiftvln/scripts/train/train_queue.sh`
- `bash src/swiftvln/scripts/eval/eval_by_name.sh <model_name>`
- `bash src/swiftvln/scripts/eval/eval_queue.sh`

脚本会自动设置 `PYTHONPATH=${SWIFTVLN_ROOT}/src`。
