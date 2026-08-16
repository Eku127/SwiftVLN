# Satellite-to-UAV SatDronePair Data Generation

[简体中文](../../zh-CN/data/SATDRONEPAIR.md) | English

This page converts DenseUAV, GTA-UAV, SUES-200, and UAV-VisLoc into the paired UAV–satellite format used by Satellite-to-UAV Stage-A:

```text
upstream raw data -> source converter -> SatDronePair/{4 sources}
             -> manifest.jsonl -> Stage-A train/eval
```

Download each upstream dataset from its official source and follow its license terms:

| Data source | Official entry point | Contents required by the converter |
|---|---|---|
| DenseUAV | [Dmmm1997/DenseUAV](https://github.com/Dmmm1997/DenseUAV)| GPS text and full train/test image |
| GTA-UAV | [Yux1angJi/GTA-UAV](https://github.com/Yux1angJi/GTA-UAV)|`GTA-UAV-LR`, four metadata JSON, drone/satellite image |
| SUES-200 | [Reza-Zhu/SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)|`SUES-200-512x512`; The upstream indicates that it is for academic research only |
| UAV-VisLoc | [IntelliSensing/UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)| Full sequence and `satellite_coordinates_range.csv`|

## 1. Install the tools and create directories

Run from the repository root:

```bash
python -m pip install -e ".[s2r-data]"
python -m tools.s2r.data_generation --help
```

The unified data-generation entry point is:

```bash
python -m tools.s2r.data_generation <dataset> <command> [args...]
```

The final directory is fixed as:

```text
SatDronePair/
├── denseuav/{drone,satellite,pairs.csv,dataset_info.json}
├── gta/{drone,satellite,pairs.csv,dataset_info.json}
├── sues/{drone,satellite,pairs.csv,dataset_info.json}
└── uavvisloc/{drone,satellite,pairs.csv,dataset_info.json}
```

Store intermediate variants separately from the final data:

```bash
export PAIR_ROOT=/path/to/SatDronePair
export PAIR_WORK=/path/to/SatDronePair-work
mkdir -p "$PAIR_ROOT" "$PAIR_WORK"
```

To reuse a parameter set, copy and edit [`config.example.yaml`](../../../tools/s2r/data_generation/config.example.yaml), then pass `--config runtime/s2r/data_generation.yaml` to the unified entry point. Explicit CLI arguments override YAML values.

## 2. Convert the four data sources

### 2.1 DenseUAV

`--dataset-root` needs to contain `Dense_GPS_{train,test}.txt`, `train/` and `test/`:

```bash
python -m tools.s2r.data_generation denseuav build_pairs \
  --dataset-root /path/to/DenseUAV \
  --output-dir "$PAIR_ROOT/denseuav" \
  --workers 16
```

### 2.2 GTA-UAV

`--data-root` points to `GTA-UAV-LR`. Converter will merge same-area/cross-area before heading check Duplicates in protocols are exported only once per physical pair while preserving the protocol ID and original split at `pairs.csv`:

```bash
python -m tools.s2r.data_generation gta_uav build_pairs \
  --data-root /path/to/GTA-UAV-LR \
  --output-dir "$PAIR_ROOT/gta" \
  --workers 8
```

### 2.3 SUES-200

The original directory needs to contain `satellite-view/` and `drone_view_512/`:

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

The result retains three training variants: `orig`, `crop384`, and `crop256`.

### 2.4 UAV-VisLoc

`--data-root` requires a CSV, drone image, and satellite TIF for each sequence:

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

The result retains the `orig` and `crop384` variants. The satellite crop scale is estimated from altitude; inspect the preview images to verify the generated pairs.

For SUES and UAV-VisLoc, `center_recrop_pairs` selects the matching schema from the image fields in `pairs.csv`.

## 3. Build the manifest

Strict mode checks four `pairs.csv` and all their image references:

```bash
python -m tools.s2r.scripts.build_manifest \
  --data_root "$PAIR_ROOT" \
  --output_path runtime/s2r/manifests/manifest_v1.jsonl \
  --val_ratio 0.1 \
  --seed 42 \
  --skip_missing false
```

The split is grouped by location: DenseUAV by base location, GTA-UAV by satellite tile, SUES-200 by `scene_id`, and UAV-VisLoc by `seq_id`. The complete dataset contains 19,365 records: DenseUAV 5,464, GTA-UAV 5,102, SUES-200 1,497, and UAV-VisLoc 7,302.

For Stage-A training and retrieval evaluation, see [Satellite-to-UAV Stage-A Training](../training/S2R_STAGE_A.md).

## 4. Inspect data quality

Generate previews of the four data sources respectively:

```bash
python -m tools.s2r.data_generation denseuav sample_preview --dataset-dir "$PAIR_ROOT/denseuav"
python -m tools.s2r.data_generation gta_uav sample_preview --dataset-dir "$PAIR_ROOT/gta"
python -m tools.s2r.data_generation sues sample_preview --dataset-dir "$PAIR_ROOT/sues"
python -m tools.s2r.data_generation uavvisloc sample_preview --dataset-dir "$PAIR_ROOT/uavvisloc"
```

Preview images are for manual inspection and are not used during training. Keep `dataset_info.json`, `pairs.csv`, `drone/`, and `satellite/` in each final data directory. Intermediate variants can be removed after validating the manifest.
