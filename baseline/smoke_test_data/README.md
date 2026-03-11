# Smoke Test Data — SatNav ver_260306 子集

供 `baseline/` 下所有 baseline（streamvln、uninavid、navila 等）复用的 **小规模多样化测试数据集**，用于快速验证训练流程是否可以跑通。

## 数据来源

- 原始数据版本：`ver_260306`
- 原始图片路径：`/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data`
- **图片不拷贝**，`annotations.json` 中的 `video` 字段（如 `images/Geneva-1_satnav_000000`）仍指向原始目录，使用时需将 `image_folder` / `video_folder` 设置为上述原始路径。

## 文件说明

```
smoke_test_data/
├── annotations.json        # 99 条 episode 的子集标注
└── generate_smoke_data.py  # 生成脚本（可复现，seed=42）
```

## 数据特征

| 项目 | 值 |
|---|---|
| 总 episodes | 99 |
| 覆盖城市 | 9 个（每城市各 11 条） |
| 每城市分布 | 短（steps<30）×3 + 中（30≤steps<80）×5 + 长（steps≥80）×3 |
| steps 范围 | min=17, max=240, median=54 |
| 随机种子 | 42（可复现） |

覆盖城市：Boston-1 · Geneva-1 · Minneapolis-1 · Minneapolis-2 · NewYork-1 · Paris-1 · Rome-1 · TheBayArea-1 · TheBayArea-2

## 各 baseline 使用方式

### StreamVLN baseline

```bash
# train_satnav.sh 中修改以下两行：
DATA_PATH="<SwiftVLN_ROOT>/baseline/smoke_test_data/annotations.json"
VIDEO_FOLDER="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data"
```

### Uni-NaVid baseline

```bash
# train_satnav.sh 中修改以下两行：
DATA_PATH="<SwiftVLN_ROOT>/baseline/smoke_test_data/annotations.json"
VIDEO_FOLDER="/mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260306/trajectory_data"
```

### 其他 baseline

只需将 `data_path` / `annotations_path` 指向 `smoke_test_data/annotations.json`，  
`image_folder` / `video_folder` 指向原始轨迹数据目录即可。

## annotations.json 字段说明

与原始 `ver_260306/trajectory_data/annotations.json` 字段完全兼容，额外增加：

| 字段 | 说明 |
|---|---|
| `smoke_source_id` | 原始数据集中的 `id`，便于追溯 |

其余字段（`id`、`trajectory_id`、`steps`、`video`、`instructions`、`actions`）均与原始格式一致。
