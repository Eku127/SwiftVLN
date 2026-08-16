# Satellite-to-UAV SatDronePair 数据生产

简体中文 | [English](../../en-US/data/SATDRONEPAIR.md)

本章将 DenseUAV、GTA-UAV、SUES-200 和 UAV-VisLoc 转换为 Satellite-to-UAV Stage-A 使用的
UAV–Satellite 配对数据：

```text
上游原始数据 -> source converter -> SatDronePair/{4 sources}
             -> manifest.jsonl -> Stage-A train/eval
```

上游图像需要从各数据集的官方入口下载，并按照对应数据集的许可协议使用：

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
python -m tools.s2r.data_generation --help
```

统一的数据生产入口为：

```bash
python -m tools.s2r.data_generation <dataset> <command> [args...]
```

最终目录固定为：

```text
SatDronePair/
├── denseuav/{drone,satellite,pairs.csv,dataset_info.json}
├── gta/{drone,satellite,pairs.csv,dataset_info.json}
├── sues/{drone,satellite,pairs.csv,dataset_info.json}
└── uavvisloc/{drone,satellite,pairs.csv,dataset_info.json}
```

将中间 variant 与最终数据分别保存：

```bash
export PAIR_ROOT=/path/to/SatDronePair
export PAIR_WORK=/path/to/SatDronePair-work
mkdir -p "$PAIR_ROOT" "$PAIR_WORK"
```

需要复用参数时，可以复制并编辑
[`config.example.yaml`](../../../tools/s2r/data_generation/config.example.yaml)，然后给统一入口增加
`--config runtime/s2r/data_generation.yaml`。显式 CLI 参数优先于 YAML。

## 2. 生成四个数据源

### 2.1 DenseUAV

`--dataset-root` 需要包含 `Dense_GPS_{train,test}.txt`、`train/` 和 `test/`：

```bash
python -m tools.s2r.data_generation denseuav build_pairs \
  --dataset-root /path/to/DenseUAV \
  --output-dir "$PAIR_ROOT/denseuav" \
  --workers 16
```

### 2.2 GTA-UAV

`--data-root` 指向 `GTA-UAV-LR`。转换器会在航向检查前合并 same-area/cross-area
协议中的重复项，每个物理配对只导出一次，同时在 `pairs.csv` 保留协议 ID 和原始 split：

```bash
python -m tools.s2r.data_generation gta_uav build_pairs \
  --data-root /path/to/GTA-UAV-LR \
  --output-dir "$PAIR_ROOT/gta" \
  --workers 8
```

### 2.3 SUES-200

原始目录需要包含 `satellite-view/` 和 `drone_view_512/`：

```bash
python -m tools.s2r.data_generation sues pipeline \
  --data-root /path/to/SUES-200-512x512 \
  --output-dir "$PAIR_WORK/sues_export" \
  --heights 150,200,250,300 \
  --nadir-min-conf 0.30 \
  --skip-preview

python -m tools.s2r.data_generation sues merge_variants \
  --dataset-dir "$PAIR_WORK/sues_export" \
  --variant-map "orig:satellite:drone,crop384:satellite_crop384:drone_crop384,crop256:satellite_crop256:drone_crop256" \
  --output-dir "$PAIR_ROOT/sues"
```

最终保留 `orig/crop384/crop256` 三个训练 variant。

### 2.4 UAV-VisLoc

`--data-root` 需要包含各序列的 CSV、drone 图像和 satellite TIF：

```bash
python -m tools.s2r.data_generation uavvisloc export_selected \
  --data-root /path/to/UAV-VisLoc/data \
  --sat-bounds-csv /path/to/UAV-VisLoc/satellite_coordinates_range.csv \
  --output-dir "$PAIR_WORK/uavvisloc_orig" \
  --allow-missing-pose

python -m tools.s2r.data_generation uavvisloc center_recrop_pairs \
  --dataset-dir "$PAIR_WORK/uavvisloc_orig" \
  --output-dir "$PAIR_WORK/uavvisloc_crop384" \
  --crop-size 384 \
  --output-size 512

python -m tools.s2r.data_generation uavvisloc merge_variants \
  --variant \
    "orig=$PAIR_WORK/uavvisloc_orig" \
    "crop384=$PAIR_WORK/uavvisloc_crop384" \
  --output-dir "$PAIR_ROOT/uavvisloc"
```

最终保留 `orig/crop384` 两个训练 variant。卫星 crop 尺度由高度近似推导，可以通过
预览图检查生成结果。

SUES 与 UAV-VisLoc 的 `center_recrop_pairs` 命令会根据 `pairs.csv` 的图像字段选择
对应 schema。

## 3. 构建 Manifest

严格模式会检查四个 `pairs.csv` 及其全部图像引用：

```bash
python -m tools.s2r.scripts.build_manifest \
  --data_root "$PAIR_ROOT" \
  --output_path runtime/s2r/manifests/manifest_v1.jsonl \
  --val_ratio 0.1 \
  --seed 42 \
  --skip_missing false
```

Split 按位置分组：DenseUAV 按基础位置、GTA-UAV 按 Satellite tile、SUES-200 按
`scene_id`、UAV-VisLoc 按 `seq_id`。当前完整数据生成 19,365 条记录：DenseUAV
5,464、GTA-UAV 5,102、SUES-200 1,497、UAV-VisLoc 7,302。

Stage-A 的训练与 retrieval 评测见 [Satellite-to-UAV Stage-A 训练](../training/S2R_STAGE_A.md)。

## 4. 质量检查

分别生成四个数据源的预览图：

```bash
python -m tools.s2r.data_generation denseuav sample_preview --dataset-dir "$PAIR_ROOT/denseuav"
python -m tools.s2r.data_generation gta_uav sample_preview --dataset-dir "$PAIR_ROOT/gta"
python -m tools.s2r.data_generation sues sample_preview --dataset-dir "$PAIR_ROOT/sues"
python -m tools.s2r.data_generation uavvisloc sample_preview --dataset-dir "$PAIR_ROOT/uavvisloc"
```

预览图用于人工检查，不参与训练。正式数据目录保留 `dataset_info.json`、`pairs.csv`、
`drone/` 和 `satellite/`；中间 variant 可以在 Manifest 校验完成后清理。
