# Trajectory Frames 分析脚本使用说明

## 功能介绍

`analyze_trajectory_frames.py` 是一个用于分析VLN数据集trajectory frames统计的工具。它可以：

- 统计每个episode的frames数量（基于annotations中actions数组的长度）
- 生成多维度的可视化图表（分布直方图、对比柱状图、累积分布图）
- 生成详细的Markdown格式分析报告
- **支持任意数量的数据集进行对比分析**

## 基本用法

### 1. 使用默认数据集（R2R和RxR）

```bash
python analyze_trajectory_frames.py
```

这会分析默认的R2R和RxR两个数据集，输出到 `trajectory_analysis/` 目录。

### 2. 添加额外的数据集

如果您有其他数据源，可以使用 `--extra` 参数添加：

```bash
# 添加一个额外的数据集
python analyze_trajectory_frames.py \
    --extra "CustomDataset:/path/to/custom/annotations.json"

# 添加多个额外的数据集
python analyze_trajectory_frames.py \
    --extra "Dataset1:/path/to/dataset1.json" \
    --extra "Dataset2:/path/to/dataset2.json" \
    --extra "Dataset3:/path/to/dataset3.json"
```

**注意**：额外数据集的格式为 `DatasetName:FilePath`，用冒号分隔。

### 3. 只分析自定义数据集

如果不想使用默认的R2R和RxR，可以使用 `--no-defaults` 参数：

```bash
python analyze_trajectory_frames.py --no-defaults \
    --extra "MyDataset1:/path/to/dataset1.json" \
    --extra "MyDataset2:/path/to/dataset2.json"
```

### 4. 指定输出目录

```bash
python analyze_trajectory_frames.py --output /path/to/output_dir
```

## 完整参数说明

```
--extra NAME:PATH    添加额外的数据集，格式: DatasetName:/path/to/annotations.json
                     （可多次使用）

--no-defaults        不使用默认的R2R和RxR数据集，仅分析通过--extra指定的数据集

--output OUTPUT      输出目录路径（默认：trajectory_analysis文件夹）

-h, --help          显示帮助信息
```

## 输出文件

运行脚本后，会在输出目录生成以下文件：

1. **frames_distribution.png** - 包含多个子图的可视化图表：
   - 各数据集的frames分布直方图
   - 多数据集对比柱状图
   - 累积分布曲线图

2. **frames_analysis_report.md** - 详细的分析报告，包含：
   - 数据源信息
   - 数据概览（episode数量、总frames数）
   - Frames数量统计对比（最小值、最大值、平均值、中位数、标准差、分位数）
   - 分区间统计
   - 关键发现
   - Top 10最长的episodes

## 数据格式要求

输入的JSON文件应该是一个数组，每个元素代表一个episode，需要包含以下字段：

```json
[
    {
        "id": 1803,
        "video": "images/17DRP5sb8fy_r2r_001803",
        "instructions": ["..."],
        "actions": [
            -1, 1, 1, 1, 2, 1, 3, ...
        ]
    },
    ...
]
```

脚本通过 `actions` 数组的长度来统计frames数量。

## 使用示例

### 示例1：对比R2R、RxR和自定义数据集

```bash
python analyze_trajectory_frames.py \
    --extra "Processed_R2R:/data/processed/r2r/annotations.json" \
    --output comparison_results
```

这会分析R2R、RxR和Processed_R2R三个数据集，并生成对比报告。

### 示例2：分析多个自定义split

```bash
python analyze_trajectory_frames.py --no-defaults \
    --extra "Train_Split:/data/train/annotations.json" \
    --extra "Val_Split:/data/val/annotations.json" \
    --extra "Test_Split:/data/test/annotations.json" \
    --output split_analysis
```

这会只分析三个自定义的数据split，不包含默认的R2R和RxR。

### 示例3：对比不同预处理版本

```bash
python analyze_trajectory_frames.py --no-defaults \
    --extra "Original:/data/original/annotations.json" \
    --extra "Cleaned:/data/cleaned/annotations.json" \
    --extra "Augmented:/data/augmented/annotations.json" \
    --output preprocessing_comparison
```

## 注意事项

1. 确保所有数据集的JSON文件格式一致
2. 数据集名称不要包含冒号（:），因为它被用作分隔符
3. 如果数据集很大，加载和分析可能需要一些时间
4. 输出的图表支持最多8个数据集的对比，超过这个数量可能会影响可读性

## 故障排除

### 错误：文件不存在

```
错误: 文件不存在: /path/to/file.json
```

**解决方案**：检查文件路径是否正确，确保文件存在且有读取权限。

### 错误：格式不正确

```
错误: 额外数据集格式不正确: CustomDataset
```

**解决方案**：确保使用正确的格式 `DatasetName:/path/to/file.json`，注意冒号不能省略。

### 错误：至少需要指定一个数据集

```
错误: 至少需要指定一个数据集
```

**解决方案**：如果使用了 `--no-defaults`，必须至少通过 `--extra` 添加一个数据集。

## 依赖环境

- Python 3.6+
- numpy
- matplotlib

在satnav环境中运行：

```bash
conda activate satnav
python analyze_trajectory_frames.py
```
