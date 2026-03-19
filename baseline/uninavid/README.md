# Uni-NaVid Baseline

原始项目：[Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid)（RSS 2025）

## Step 1: 下载模型权重

下载脚本统一放在 `baseline/uninavid/scripts/`，模型保存到 `baseline/uninavid/model/`。

### 默认下载（EVA-CLIP + Uni-NaVid 权重）

```bash
bash baseline/uninavid/scripts/download_uninavid_models.sh
```

### 同时下载 Vicuna-7B 底座模型（可选，~13 GB）

```bash
bash baseline/uninavid/scripts/download_uninavid_models.sh --vicuna
```

### 下载全部模型

```bash
bash baseline/uninavid/scripts/download_uninavid_models.sh --all
```

### 其他选项

```bash
# 自定义保存目录
bash baseline/uninavid/scripts/download_uninavid_models.sh --model-dir /path/to/model

# 调整进度播报间隔（秒）
bash baseline/uninavid/scripts/download_uninavid_models.sh --monitor-interval 10

# 只下载 Uni-NaVid 权重（跳过 EVA-CLIP）
bash baseline/uninavid/scripts/download_uninavid_models.sh --skip-eva

# 查看帮助
bash baseline/uninavid/scripts/download_uninavid_models.sh --help
```

---

## 模型说明

| 模型 | 来源 | 大小 | 默认下载 |
|------|------|------|----------|
| EVA-CLIP (`eva_vit_g.pth`) | Google Storage 直连 | ~3.9 GB | ✅ |
| Uni-NaVid weights | HuggingFace `Jzzhang/Uni-NaVid` | — | ✅ |
| Vicuna-7B (`lmsys/vicuna-7b-v1.5`) | HuggingFace via hf-mirror.com | ~13 GB | ❌（需 `--vicuna`）|

HuggingFace 模型通过 `hf-mirror.com` 直连下载，无需代理，支持断点续传。

---

## 目录结构

```

## 评测

SatNav 评测入口：

```bash
bash baseline/uninavid/scripts/eval_satnav.sh <exp_name_or_checkpoint_path>
```

SatNav 评测 split 约定：

- 不传 `split`：默认顺序运行 `val_seen` 和 `val_unseen`
- 传 `val_seen` / `val_unseen` / `test`：只跑指定单个 split

示例：

```bash
# 默认双 split
bash baseline/uninavid/scripts/eval_satnav.sh \
  /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/<exp_name>

# 单独跑 val_unseen
bash baseline/uninavid/scripts/eval_satnav.sh \
  /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/output/uninavid-baseline/<exp_name> \
  val_unseen 8
```
baseline/uninavid/
├── model/
│   ├── eva_vit_g.pth
│   ├── uninavid-7b-full-224-video-fps-1-grid-2/
│   └── vicuna-7b-v1.5/          # 仅 --vicuna 时存在
├── scripts/
│   └── download_uninavid_models.sh
├── doc/
└── README.md
```
