#!/usr/bin/env python3
"""
R2R和RxR Trajectory数据集Frames统计分析脚本

分析R2R和RxR数据集中每个episode的frames数量（基于annotation中actions数组的长度），
生成柱状图和详细统计信息。支持添加额外的数据集进行对比分析。

使用方法:
    # 使用默认的R2R和RxR数据集
    python analyze_trajectory_frames.py
    
    # 添加额外的数据集
    python analyze_trajectory_frames.py --extra CustomDataset:/path/to/custom/annotations.json
    
    # 添加多个额外的数据集
    python analyze_trajectory_frames.py \
        --extra Dataset1:/path/to/dataset1.json \
        --extra Dataset2:/path/to/dataset2.json
    
    # 只分析自定义数据集（不使用默认的R2R和RxR）
    python analyze_trajectory_frames.py --no-defaults \
        --extra Dataset1:/path/to/dataset1.json \
        --extra Dataset2:/path/to/dataset2.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple
from datetime import datetime

# 默认数据路径
DEFAULT_DATASETS = {
    "R2R": "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R/annotations.json",
    "RxR": "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new/annotations.json"
}


def load_annotations(file_path: str) -> List[Dict]:
    """加载annotation文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_frames(annotations: List[Dict]) -> List[int]:
    """统计每个episode的frames数量（actions数组长度）"""
    return [len(episode['actions']) for episode in annotations]


def calculate_statistics(frame_counts: List[int]) -> Dict:
    """计算统计指标"""
    if not frame_counts:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'q25': 0.0,
            'q75': 0.0,
        }
    
    return {
        'count': len(frame_counts),
        'min': int(np.min(frame_counts)),
        'max': int(np.max(frame_counts)),
        'mean': float(np.mean(frame_counts)),
        'median': float(np.median(frame_counts)),
        'std': float(np.std(frame_counts)),
        'q25': float(np.percentile(frame_counts, 25)),
        'q75': float(np.percentile(frame_counts, 75)),
    }


def plot_frame_distribution(
    datasets: Dict[str, List[int]],
    output_path: Path
):
    """生成frames数量分布的柱状图，支持多个数据集"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 颜色方案
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    dataset_names = list(datasets.keys())
    num_datasets = len(dataset_names)
    
    # 动态确定子图布局
    if num_datasets <= 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    else:
        # 前n个数据集各自的分布图 + 1个对比图 + 1个累积分布图
        nrows = (num_datasets + 1) // 2 + 1
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows))
        axes = axes.flatten()
    
    title_text = ' and '.join(dataset_names) if num_datasets <= 3 else f'{num_datasets} Datasets'
    fig.suptitle(f'{title_text} Trajectory Frames Distribution', fontsize=16, fontweight='bold')
    
    ax_idx = 0
    
    # 1. 为每个数据集生成单独的分布直方图
    for i, (name, frames) in enumerate(datasets.items()):
        if ax_idx >= len(axes) - 2:  # 保留最后两个位置给对比图和累积分布图
            break
            
        ax = axes[ax_idx]
        color = colors[i % len(colors)]
        
        ax.hist(frames, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Frames', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name} Frames Distribution (n={len(frames)})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats = calculate_statistics(frames)
        stats_text = f"Mean: {stats['mean']:.1f}\nMedian: {stats['median']:.1f}\nStd: {stats['std']:.1f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax_idx += 1
    
    # 2. 对比柱状图（分区间统计）
    ax_compare = axes[-2]
    
    # 定义区间
    max_frames = max(max(frames) for frames in datasets.values())
    bins = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, max_frames + 1]
    bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-150', '150-200', '200-300', '300+']
    
    x = np.arange(len(bin_labels))
    width = 0.8 / num_datasets
    
    for i, (name, frames) in enumerate(datasets.items()):
        hist, _ = np.histogram(frames, bins=bins)
        offset = (i - num_datasets / 2 + 0.5) * width
        ax_compare.bar(x + offset, hist, width, label=name, color=colors[i % len(colors)], alpha=0.7)
    
    ax_compare.set_xlabel('Frame Count Range', fontsize=12)
    ax_compare.set_ylabel('Number of Episodes', fontsize=12)
    ax_compare.set_title('Frames Distribution Comparison', fontsize=14, fontweight='bold')
    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax_compare.legend()
    ax_compare.grid(True, alpha=0.3, axis='y')
    
    # 3. 累积分布图
    ax_cumulative = axes[-1]
    
    for i, (name, frames) in enumerate(datasets.items()):
        sorted_frames = np.sort(frames)
        cumulative = np.arange(1, len(sorted_frames) + 1) / len(sorted_frames) * 100
        ax_cumulative.plot(sorted_frames, cumulative, label=name, color=colors[i % len(colors)], linewidth=2)
    
    ax_cumulative.set_xlabel('Number of Frames', fontsize=12)
    ax_cumulative.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax_cumulative.set_title('Cumulative Distribution of Frames', fontsize=14, fontweight='bold')
    ax_cumulative.legend()
    ax_cumulative.grid(True, alpha=0.3)
    
    # 添加百分位线
    for percentile in [25, 50, 75]:
        ax_cumulative.axhline(y=percentile, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_cumulative.text(ax_cumulative.get_xlim()[1] * 0.02, percentile + 1, f'{percentile}%', 
                          fontsize=9, color='gray')
    
    # 隐藏多余的子图
    for i in range(ax_idx, len(axes) - 2):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    plt.close()


def generate_report(
    datasets: Dict[str, List[int]],
    dataset_paths: Dict[str, str],
    output_path: Path
):
    """生成Markdown格式的分析报告，支持多个数据集"""
    
    # 计算所有数据集的统计信息
    all_stats = {name: calculate_statistics(frames) for name, frames in datasets.items()}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 标题
        dataset_names = list(datasets.keys())
        if len(dataset_names) <= 3:
            title = ' 和 '.join(dataset_names)
        else:
            title = f"{len(dataset_names)}个数据集"
        f.write(f"# {title} Trajectory数据集Frames统计分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 数据源信息
        f.write("## 1. 数据源\n\n")
        f.write("| 数据集 | 文件路径 |\n")
        f.write("|--------|----------|\n")
        for name in dataset_names:
            f.write(f"| {name} | {dataset_paths.get(name, 'N/A')} |\n")
        f.write("\n")
        
        # 数据概览
        f.write("## 2. 数据概览\n\n")
        f.write("| 数据集 | Episode数量 | 总Frames数 |\n")
        f.write("|--------|-------------|------------|\n")
        
        total_episodes = 0
        total_frames = 0
        for name, frames in datasets.items():
            stats = all_stats[name]
            total_episodes += stats['count']
            total_frames += sum(frames)
            f.write(f"| {name} | {stats['count']} | {sum(frames)} |\n")
        
        f.write(f"| **合计** | **{total_episodes}** | **{total_frames}** |\n")
        f.write("\n")
        
        # 统计信息对比
        f.write("## 3. Frames数量统计对比\n\n")
        
        # 表头
        header = "| 指标 |"
        separator = "|------|"
        for name in dataset_names:
            header += f" {name} |"
            separator += "-----:|"
        f.write(header + "\n")
        f.write(separator + "\n")
        
        # 各项指标
        metrics = [
            ('最小值', 'min', ''),
            ('最大值', 'max', ''),
            ('平均值', 'mean', '.2f'),
            ('中位数', 'median', '.2f'),
            ('标准差', 'std', '.2f'),
            ('25%分位数', 'q25', '.2f'),
            ('75%分位数', 'q75', '.2f'),
        ]
        
        for metric_name, metric_key, fmt in metrics:
            row = f"| {metric_name} |"
            for name in dataset_names:
                value = all_stats[name][metric_key]
                if fmt:
                    row += f" {value:{fmt}} |"
                else:
                    row += f" {value} |"
            f.write(row + "\n")
        
        f.write("\n")
        
        # 分区间统计
        f.write("## 4. Frames数量分区间统计\n\n")
        
        max_frames = max(max(frames) for frames in datasets.values())
        bins = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, max_frames + 1]
        bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-150', '150-200', '200-300', '300+']
        
        # 计算每个数据集的直方图
        histograms = {}
        for name, frames in datasets.items():
            hist, _ = np.histogram(frames, bins=bins)
            histograms[name] = hist
        
        # 表头
        header = "| Frames区间 |"
        separator = "|------------|"
        for name in dataset_names:
            header += f" {name}数量 | {name}占比 |"
            separator += "---------|----------|"
        f.write(header + "\n")
        f.write(separator + "\n")
        
        # 各区间数据
        for i, label in enumerate(bin_labels):
            row = f"| {label} |"
            for name in dataset_names:
                count = histograms[name][i]
                pct = count / all_stats[name]['count'] * 100 if all_stats[name]['count'] > 0 else 0
                row += f" {count} | {pct:.1f}% |"
            f.write(row + "\n")
        
        f.write("\n")
        
        # 关键发现
        f.write("## 5. 关键发现\n\n")
        
        # 找出各项指标的极值
        max_mean_dataset = max(dataset_names, key=lambda x: all_stats[x]['mean'])
        min_mean_dataset = min(dataset_names, key=lambda x: all_stats[x]['mean'])
        max_frames_dataset = max(dataset_names, key=lambda x: all_stats[x]['max'])
        max_std_dataset = max(dataset_names, key=lambda x: all_stats[x]['std'])
        largest_dataset = max(dataset_names, key=lambda x: all_stats[x]['count'])
        
        f.write(f"1. **平均Frames数量**: {max_mean_dataset}数据集的平均frames数量最高({all_stats[max_mean_dataset]['mean']:.1f})，{min_mean_dataset}最低({all_stats[min_mean_dataset]['mean']:.1f})\n\n")
        f.write(f"2. **最大Frames数量**: {max_frames_dataset}数据集有最长的trajectory，最大值为{all_stats[max_frames_dataset]['max']} frames\n\n")
        f.write(f"3. **数据波动性**: {max_std_dataset}数据集的标准差最大({all_stats[max_std_dataset]['std']:.1f})，说明trajectory长度最不均匀\n\n")
        f.write(f"4. **数据集规模**: {largest_dataset}数据集episode数量最多({all_stats[largest_dataset]['count']}个)\n\n")
        
        # Top 10 最长的episodes
        f.write("## 6. Top 10 最长的Episodes\n\n")
        
        for name, frames in datasets.items():
            f.write(f"### {name}数据集\n\n")
            f.write("| 排名 | Episode ID | Frames数量 |\n")
            f.write("|------|------------|------------|\n")
            
            # 获取最长的10个episodes
            annotations = load_annotations(dataset_paths[name])
            with_frames = [(ep['id'], len(ep['actions'])) for ep in annotations]
            sorted_eps = sorted(with_frames, key=lambda x: x[1], reverse=True)[:10]
            
            for rank, (ep_id, frame_count) in enumerate(sorted_eps, 1):
                f.write(f"| {rank} | {ep_id} | {frame_count} |\n")
            
            f.write("\n")
    
    print(f"报告已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='分析trajectory数据集的frames统计，支持多个数据集对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认的R2R和RxR数据集
  python %(prog)s
  
  # 添加额外的数据集
  python %(prog)s --extra CustomDataset:/path/to/custom/annotations.json
  
  # 添加多个额外的数据集
  python %(prog)s --extra Dataset1:/path/to/dataset1.json --extra Dataset2:/path/to/dataset2.json
  
  # 只分析自定义数据集（不使用默认的R2R和RxR）
  python %(prog)s --no-defaults --extra Dataset1:/path/to/dataset1.json --extra Dataset2:/path/to/dataset2.json
        """
    )
    parser.add_argument(
        '--extra',
        action='append',
        default=[],
        metavar='NAME:PATH',
        help='添加额外的数据集，格式: DatasetName:/path/to/annotations.json (可多次使用)'
    )
    parser.add_argument(
        '--no-defaults',
        action='store_true',
        help='不使用默认的R2R和RxR数据集，仅分析通过--extra指定的数据集'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录路径（默认：当前脚本所在目录下的trajectory_analysis文件夹）'
    )
    
    args = parser.parse_args()
    
    # 构建数据集字典
    dataset_paths = OrderedDict()
    
    # 添加默认数据集（如果未禁用）
    if not args.no_defaults:
        dataset_paths.update(DEFAULT_DATASETS)
    
    # 添加额外的数据集
    for extra in args.extra:
        if ':' not in extra:
            print(f"错误: 额外数据集格式不正确: {extra}")
            print("正确格式: DatasetName:/path/to/annotations.json")
            return
        
        name, path = extra.split(':', 1)
        if not Path(path).exists():
            print(f"错误: 文件不存在: {path}")
            return
        
        dataset_paths[name] = path
    
    # 检查是否至少有一个数据集
    if not dataset_paths:
        print("错误: 至少需要指定一个数据集")
        print("使用 --help 查看使用方法")
        return
    
    # 确定输出目录
    if args.output is None:
        output_dir = Path(__file__).parent / 'trajectory_analysis'
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有数据集
    datasets = OrderedDict()
    print("正在加载数据集...")
    print("="*60)
    
    for name, path in dataset_paths.items():
        print(f"\n加载 {name} 数据: {path}")
        try:
            annotations = load_annotations(path)
            frames = count_frames(annotations)
            datasets[name] = frames
            print(f"  ✓ 加载完成，共 {len(annotations)} 个episodes")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            return
    
    # 生成图表
    print("\n" + "="*60)
    print("正在生成统计图表...")
    plot_path = output_dir / 'frames_distribution.png'
    plot_frame_distribution(datasets, plot_path)
    
    # 生成报告
    print("正在生成分析报告...")
    report_path = output_dir / 'frames_analysis_report.md'
    generate_report(datasets, dataset_paths, report_path)
    
    # 打印简要统计
    print("\n" + "="*60)
    print("统计摘要:")
    print("="*60)
    
    for name, frames in datasets.items():
        stats = calculate_statistics(frames)
        print(f"\n{name}数据集:")
        print(f"  Episodes: {stats['count']}")
        print(f"  平均Frames: {stats['mean']:.1f}")
        print(f"  中位数Frames: {stats['median']:.1f}")
        print(f"  范围: {stats['min']} - {stats['max']}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"  - 统计图表: {plot_path.name}")
    print(f"  - 分析报告: {report_path.name}")


if __name__ == '__main__':
    main()
