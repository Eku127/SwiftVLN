# S2R SatDronePair 数据生产

本目录把 DenseUAV、GTA-UAV、SUES-200 和 UAV-VisLoc 转换为 SwiftVLN
Stage-A adapter 使用的 UAV–Satellite 配对数据：

```text
上游原始数据 -> source converter -> SatDronePair/{4 sources}
             -> manifest.jsonl -> Stage-A train/eval
```

仓库不包含、不会自动下载或重新分发上游图像。请从官方入口下载并遵守各自条款；
在没有明确再分发授权时，不要公开镜像生成后的图像包。

| 数据源 | 官方入口 | 转换器需要的内容 |
|---|---|---|
| DenseUAV | [Dmmm1997/DenseUAV](https://github.com/Dmmm1997/DenseUAV) | GPS 文本及完整 train/test 图像 |
| GTA-UAV | [Yux1angJi/GTA-UAV](https://github.com/Yux1angJi/GTA-UAV) | `GTA-UAV-LR`、四个 metadata JSON、drone/satellite 图像 |
| SUES-200 | [Reza-Zhu/SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark) | `SUES-200-512x512`；上游注明仅供学术研究 |
| UAV-VisLoc | [IntelliSensing/UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) | 完整序列和 `satellite_coordinates_range.csv` |

## 1. 安装与目录

在仓库根目录执行：

```bash
python -m pip install -e ".[s2r-data]"
swiftvln s2r-data --help
```

下面两种入口等价；所有子命令都支持 `--help`：

```bash
swiftvln s2r-data <dataset> <command> [args...]
python -m swiftvln.s2r.data_generation <dataset> <command> [args...]
```

最终目录固定为：

```text
SatDronePair/
├── denseuav/{drone,satellite,pairs.csv,dataset_info.json}
├── gta/{drone,satellite,pairs.csv,dataset_info.json}
├── sues/{drone,satellite,pairs.csv,dataset_info.json}
└── uavvisloc/{drone,satellite,pairs.csv,dataset_info.json}
```

建议把中间 variant 与最终数据分开：

```bash
export PAIR_ROOT=/path/to/SatDronePair
export PAIR_WORK=/path/to/SatDronePair-work
mkdir -p "$PAIR_ROOT" "$PAIR_WORK"
```

如需复用参数，可复制并编辑
[`config.example.yaml`](config.example.yaml)，然后给统一入口增加
`--config runtime/s2r/data_generation.yaml`。显式 CLI 参数优先于 YAML。

## 2. 生成四个数据源

### DenseUAV

`--dataset-root` 需要包含 `Dense_GPS_{train,test}.txt`、`train/` 和 `test/`：

```bash
swiftvln s2r-data denseuav build_pairs \
  --dataset-root /path/to/DenseUAV \
  --output-dir "$PAIR_ROOT/denseuav" \
  --workers 16
```

### GTA-UAV

`--data-root` 指向 `GTA-UAV-LR`。转换器会在航向检查前合并
same-area/cross-area 的协议重复项，每个物理配对只导出一次，同时在 `pairs.csv`
保留两个协议的 ID 和原始 split：

```bash
swiftvln s2r-data gta_uav build_pairs \
  --data-root /path/to/GTA-UAV-LR \
  --output-dir "$PAIR_ROOT/gta" \
  --workers 8
```

### SUES-200

原始目录需要包含 `satellite-view/` 和 `drone_view_512/`：

```bash
swiftvln s2r-data sues pipeline \
  --data-root /path/to/SUES-200-512x512 \
  --output-dir "$PAIR_WORK/sues_export" \
  --heights 150,200,250,300 \
  --nadir-min-conf 0.30 \
  --skip-preview

swiftvln s2r-data sues merge_variants \
  --dataset-dir "$PAIR_WORK/sues_export" \
  --variant-map "orig:satellite:drone,crop384:satellite_crop384:drone_crop384,crop256:satellite_crop256:drone_crop256" \
  --output-dir "$PAIR_ROOT/sues"
```

默认保留 `orig/crop384/crop256` 三个训练 variant；它们是有效增强，不是重复文件。

### UAV-VisLoc

`--data-root` 需要包含各序列的 CSV、drone 图像和 satellite TIF：

```bash
swiftvln s2r-data uavvisloc export_selected \
  --data-root /path/to/UAV-VisLoc/data \
  --sat-bounds-csv /path/to/UAV-VisLoc/satellite_coordinates_range.csv \
  --output-dir "$PAIR_WORK/uavvisloc_orig" \
  --allow-missing-pose

swiftvln s2r-data uavvisloc center_recrop_pairs \
  --dataset-dir "$PAIR_WORK/uavvisloc_orig" \
  --output-dir "$PAIR_WORK/uavvisloc_crop384" \
  --crop-size 384 \
  --output-size 512

swiftvln s2r-data uavvisloc merge_variants \
  --variant \
    "orig=$PAIR_WORK/uavvisloc_orig" \
    "crop384=$PAIR_WORK/uavvisloc_crop384" \
  --output-dir "$PAIR_ROOT/uavvisloc"
```

最终保留 `orig/crop384` 两个训练 variant。卫星 crop 尺度由高度近似推导，正式使用前
应抽样检查图像质量。

SUES 与 UAV-VisLoc 的 `center_recrop_pairs` 命令共享根目录实现，并根据
`pairs.csv` 的图像字段自动选择 schema；命令名与输出结构保持不变。

## 3. 构建 manifest 并训练

严格模式会检查四个 `pairs.csv` 及其全部图像引用：

```bash
python src/swiftvln/s2r/scripts/build_manifest.py \
  --data_root "$PAIR_ROOT" \
  --output_path runtime/s2r/manifests/manifest_v1.jsonl \
  --val_ratio 0.1 \
  --seed 42 \
  --skip_missing false
```

split 按位置分组，避免同地点泄漏：DenseUAV 按基础位置、GTA 按卫星 tile、SUES
按 `scene_id`、UAV-VisLoc 按 `seq_id`。当前正式数据应得到 19,365 条记录：
DenseUAV 5,464、GTA 5,102、SUES 1,497、UAV-VisLoc 7,302。

启动 Stage-A：

```bash
MANIFEST_PATH=runtime/s2r/manifests/manifest_v1.jsonl \
TEACHER_MODEL_PATH=/path/to/qwen-vl-or-swiftvln-checkpoint \
bash src/swiftvln/s2r/scripts/train_s2r.sh
```

## 4. 质量检查

预览图只用于人工 QA，不参与训练，也不需要保留在正式数据目录：

四个命令共享 `data_generation/sample_preview.py` renderer；launcher 根据 dataset
选择 pairs.csv schema adapter，因此 CLI 与各数据源原有参数保持不变。

```bash
swiftvln s2r-data denseuav sample_preview --dataset-dir "$PAIR_ROOT/denseuav"
swiftvln s2r-data gta_uav sample_preview --dataset-dir "$PAIR_ROOT/gta"
swiftvln s2r-data sues sample_preview --dataset-dir "$PAIR_ROOT/sues"
swiftvln s2r-data uavvisloc sample_preview --dataset-dir "$PAIR_ROOT/uavvisloc"
```

生产时使用新的空输出目录；`dataset_info.json` 与 `pairs.csv` 应保留，preview 和
中间 variant 目录可在严格 manifest 校验通过后清理。
