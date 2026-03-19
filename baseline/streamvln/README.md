# StreamVLN Baseline（SatNav）

本说明面向在 SwiftVLN 仓库内运行 StreamVLN baseline 的训练与评测流程。

## 1. 环境如何准备

- 工作目录使用 SwiftVLN 仓库根目录：`/mnt/data1/home/jiangjiajun/workspace/SwiftVLN`。
- 训练与评测统一使用 conda 环境：`streamvln-baseline`。

环境安装按以下顺序完成：

```bash
# Step 1: 创建 conda 环境
conda create -n streamvln-baseline python=3.9
conda activate streamvln-baseline

# Step 2: 安装 PyTorch（CUDA 12.1）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Step 3: 安装 flash_attn（从预编译 whl，须匹配 torch+CUDA+Python 版本）
pip install /mnt/data1/home/jiangjiajun/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# Step 4: 安装其余依赖
pip install -r baseline/streamvln/requirements.txt

# Step 5: 安装 SatNav（评测必需，editable install）
pip install -e /mnt/data1/home/jiangjiajun/workspace/SatNav
```

补充说明：

- 若通过 ModelScope 下载模型，需额外安装 `modelscope`。
- 确保本机可访问 `/mnt/data1/home/jiangjiajun/workspace/StreamVLN` 源码目录（训练脚本通过 `PYTHONPATH` 依赖该目录下的 `llava/streamvln/trl`）。
- 检查数据根目录 `/mnt/data3/jiangjiajun/dataset/satnav_datasets`，并确认目标 `ver_xxxxxx` 版本完整。

环境可用判定标准：

- `streamvln-baseline` 能正常导入训练依赖（`torch`、`transformers`、`deepspeed`、`flash_attn`）；
- `satnav` 可导入；
- `StreamVLN` 源码目录存在且可被 `PYTHONPATH` 解析；
- 目标数据版本存在训练轨迹和评测 episodes 文件。

## 2. 三个模型如何下载

训练与评测依赖以下三个模型目录：

- 官方 StreamVLN checkpoint（continue 模式）：
  `baseline/streamvln/model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`
- LLaVA-Video-7B-Qwen2（scratch 模式与 tokenizer fallback）：
  `baseline/streamvln/model/LLaVA-Video-7B-Qwen2`
- SigLIP vision tower：
  `baseline/streamvln/model/siglip-so400m-patch14-384`

推荐使用统一下载入口：`baseline/streamvln/scripts/download_model.sh`。

下载策略建议：

- 官方 StreamVLN checkpoint：
  默认从 HuggingFace（通过 `hf-mirror`）下载；网络受限时可切换到 ModelScope。
- LLaVA-Video-7B-Qwen2：
  可从 HuggingFace 或 ModelScope 下载；在国内网络通常优先 ModelScope。
- SigLIP vision tower：
  优先 HuggingFace；若目标环境访问 HF 不稳定，建议使用同名可用镜像源并保持目录名不变。
- 推荐顺序：先官方 checkpoint，再 LLaVA-Video-7B-Qwen2，最后 SigLIP。
- 下载完成后，检查上述三个目录均存在且内容完整；跨机器运行时需保证各机器路径一致。

## 3. 训练 skill 如何启动

本仓库已定义 baseline 训练/评测专用 skill：

- `.codex/skills/run-streamvln-baseline/SKILL.md`

启动方式说明：

- 在 Codex 会话中明确提出“运行 StreamVLN baseline 训练（SatNav）”。
- 同时给出关键参数：训练模式（continue 或 scratch）、目标服务器（98/73/17）、数据版本（可默认最新）。
- 若希望训练后自动评测，说明使用 train+eval 串行流程。
- 若只做训练，需在请求中明确“只训练不评测”。

该 skill 的执行约定：

- 使用 `streamvln-baseline` 环境。
- 训练输出写入 `output/streamvln-baseline/<EXP_NAME>/`。
- 训练日志保存在对应输出目录内。

## 4. Eval 如何启动

可通过以下两种方式启动评测：

- baseline 直接评测入口：`baseline/streamvln/scripts/eval_satnav.sh`
- 通过训练 skill 的 train+eval 串行流程自动触发评测。

评测输入支持两类：

- 按实验名评测（推荐）：使用训练产出的 `EXP_NAME`。
- 按 checkpoint 路径评测：用于兼容历史目录或手工路径。

SatNav 评测 split 约定：

- `bash baseline/streamvln/scripts/eval_satnav.sh <exp_or_ckpt>`：
  默认顺序运行 `val_seen` 和 `val_unseen`
- `bash baseline/streamvln/scripts/eval_satnav.sh <exp_or_ckpt> val_seen`：
  只跑 `val_seen`
- `bash baseline/streamvln/scripts/eval_satnav.sh <exp_or_ckpt> val_unseen`：
  只跑 `val_unseen`

评测输出位置：

- `results/streamvln-baseline/<EXP_NAME_or_subpath>/<split>/`
- 评测日志与 summary 写在同一结果目录下。

评测前建议检查：

- 目标 checkpoint 是否存在。
- 对应 SatNav 数据版本是否可用。
- 三个模型目录是否完整（尤其是 tokenizer fallback 所需基座模型）。
