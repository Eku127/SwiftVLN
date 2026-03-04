#!/usr/bin/env python3
"""
SatNav 评估结果分析脚本

分析 all_results.json 文件，生成详细的统计报告和CSV数据文件。
用于了解模型在不同场景、指令复杂度下的表现，以及分析失败原因分布。

使用方法:
    python analyze_satnav_results.py --input path/to/all_results.json
    python analyze_satnav_results.py --input path/to/all_results.json --output custom_output_dir/
"""

import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np

# ==================== 阈值参数 ====================
STEPS_NORMAL_MAX = 120
STEPS_LARGE_MIN = 150
DISTANCE_CLOSE_MAX = 25
DISTANCE_FAR_MIN = 50
SPL_HIGH_THRESHOLD = 0.9
INSTRUCTION_SIMPLE_MAX = 29
INSTRUCTION_COMPLEX_MIN = 50


def classify_instruction_complexity(instruction: str) -> str:
    """根据单词数分类指令复杂度"""
    word_count = len(instruction.split())
    if word_count <= INSTRUCTION_SIMPLE_MAX:
        return "simple"
    elif word_count < INSTRUCTION_COMPLEX_MIN:
        return "standard"
    else:
        return "complex"


def classify_episode_status(episode: Dict) -> str:
    """分类episode的运行状态"""
    success = episode['success']
    oracle_success = episode['oracle_success']
    steps = episode['steps']
    distance = episode['distance_to_goal']
    spl = episode['spl']
    
    if success == 1:
        if spl > SPL_HIGH_THRESHOLD:
            return "高效成功"
        else:
            return "低效成功"
    else:
        # 失败情况
        if oracle_success == 1:
            if steps <= STEPS_NORMAL_MAX and distance <= DISTANCE_CLOSE_MAX:
                return "停止位置不对"
            elif steps > STEPS_LARGE_MIN:
                return "绕圈"
            else:
                return "其他失败"
        else:
            if steps <= STEPS_NORMAL_MAX:
                return "早停"
            elif distance > DISTANCE_FAR_MIN:
                return "走偏"
            else:
                return "其他失败"


def calculate_statistics(episodes: List[Dict]) -> Dict:
    """计算统计指标"""
    if not episodes:
        return {
            'count': 0,
            'success_rate': 0.0,
            'oracle_success_rate': 0.0,
            'avg_spl': 0.0,
            'median_spl': 0.0,
            'avg_distance': 0.0,
            'median_distance': 0.0,
            'avg_steps': 0.0,
            'median_steps': 0.0,
        }
    
    success_count = sum(1 for e in episodes if e['success'] == 1)
    oracle_success_count = sum(1 for e in episodes if e['oracle_success'] == 1)
    spls = [e['spl'] for e in episodes]
    distances = [e['distance_to_goal'] for e in episodes]
    steps_list = [e['steps'] for e in episodes]
    
    return {
        'count': len(episodes),
        'success_rate': success_count / len(episodes) if episodes else 0.0,
        'oracle_success_rate': oracle_success_count / len(episodes) if episodes else 0.0,
        'avg_spl': np.mean(spls) if spls else 0.0,
        'median_spl': np.median(spls) if spls else 0.0,
        'avg_distance': np.mean(distances) if distances else 0.0,
        'median_distance': np.median(distances) if distances else 0.0,
        'avg_steps': np.mean(steps_list) if steps_list else 0.0,
        'median_steps': np.median(steps_list) if steps_list else 0.0,
    }


def analyze_by_scene(episodes: List[Dict]) -> Dict[str, Dict]:
    """按scene_id分组分析"""
    scene_groups = defaultdict(list)
    for episode in episodes:
        scene_groups[episode['scene_id']].append(episode)
    
    scene_stats = {}
    for scene_id, scene_episodes in scene_groups.items():
        scene_stats[scene_id] = calculate_statistics(scene_episodes)
    
    return scene_stats


def analyze_by_complexity(episodes: List[Dict]) -> Dict[str, Dict]:
    """按指令复杂度分组分析"""
    complexity_groups = defaultdict(list)
    for episode in episodes:
        complexity = classify_instruction_complexity(episode['instruction'])
        complexity_groups[complexity].append(episode)
    
    complexity_stats = {}
    for complexity, comp_episodes in complexity_groups.items():
        complexity_stats[complexity] = calculate_statistics(comp_episodes)
    
    return complexity_stats


def analyze_by_status(episodes: List[Dict]) -> Dict[str, Dict]:
    """按episode状态分类统计，返回每个状态的episode_id列表"""
    status_episodes = defaultdict(list)
    for episode in episodes:
        status = classify_episode_status(episode)
        status_episodes[status].append(episode['episode_id'])
    
    # 转换为包含count和episode_ids的字典
    result = {}
    for status, episode_ids in status_episodes.items():
        result[status] = {
            'count': len(episode_ids),
            'episode_ids': sorted(episode_ids, key=lambda x: int(x) if x.isdigit() else 0)
        }
    
    return result


def generate_markdown_report(
    overall_stats: Dict,
    scene_stats: Dict[str, Dict],
    complexity_stats: Dict[str, Dict],
    status_data: Dict[str, Dict],
    output_path: Path
):
    """生成Markdown报告"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# SatNav 评估结果分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 整体统计
        f.write("## 1. 整体统计\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 总Episode数 | {overall_stats['count']} |\n")
        f.write(f"| 成功率 | {overall_stats['success_rate']:.2%} |\n")
        f.write(f"| Oracle成功率 | {overall_stats['oracle_success_rate']:.2%} |\n")
        f.write(f"| 平均SPL | {overall_stats['avg_spl']:.4f} |\n")
        f.write(f"| 中位数SPL | {overall_stats['median_spl']:.4f} |\n")
        f.write(f"| 平均距离目标 | {overall_stats['avg_distance']:.2f}米 |\n")
        f.write(f"| 中位数距离目标 | {overall_stats['median_distance']:.2f}米 |\n")
        f.write(f"| 平均步数 | {overall_stats['avg_steps']:.1f} |\n")
        f.write(f"| 中位数步数 | {overall_stats['median_steps']:.0f} |\n")
        f.write("\n")
        
        # 按Scene统计
        f.write("## 2. 按Scene统计\n\n")
        f.write("| Scene ID | 数量 | 成功率 | Oracle成功率 | 平均SPL | 中位数SPL | 平均距离 |\n")
        f.write("|----------|------|--------|--------------|---------|----------|----------|\n")
        for scene_id in sorted(scene_stats.keys()):
            stats = scene_stats[scene_id]
            f.write(f"| {scene_id} | {stats['count']} | {stats['success_rate']:.2%} | "
                   f"{stats['oracle_success_rate']:.2%} | {stats['avg_spl']:.4f} | "
                   f"{stats['median_spl']:.4f} | {stats['avg_distance']:.2f} |\n")
        f.write("\n")
        
        # 按指令复杂度统计
        f.write("## 3. 按Instruction复杂度统计\n\n")
        f.write("| 复杂度 | 数量 | 成功率 | Oracle成功率 | 平均SPL | 中位数SPL | 平均距离 |\n")
        f.write("|--------|------|--------|--------------|---------|----------|----------|\n")
        for complexity in ['simple', 'standard', 'complex']:
            if complexity in complexity_stats:
                stats = complexity_stats[complexity]
                complexity_name = {'simple': '简单', 'standard': '标准', 'complex': '复杂'}[complexity]
                f.write(f"| {complexity_name} | {stats['count']} | {stats['success_rate']:.2%} | "
                       f"{stats['oracle_success_rate']:.2%} | {stats['avg_spl']:.4f} | "
                       f"{stats['median_spl']:.4f} | {stats['avg_distance']:.2f} |\n")
        f.write("\n")
        
        # Episode状态分类
        f.write("## 4. Episode运行状态分类\n\n")
        f.write("| 状态分类 | 数量 | 占比 |\n")
        f.write("|----------|------|------|\n")
        total = sum(status_data[status]['count'] for status in status_data)
        status_order = ['高效成功', '低效成功', '停止位置不对', '绕圈', '早停', '走偏', '其他失败']
        for status in status_order:
            if status in status_data:
                count = status_data[status]['count']
                percentage = count / total if total > 0 else 0.0
                f.write(f"| {status} | {count} | {percentage:.2%} |\n")
        f.write("\n")
        
        # 状态分类详细说明
        f.write("### 4.1 状态分类说明\n\n")
        f.write("#### 成功类\n\n")
        f.write(f"- **高效成功**: success=1 且 SPL > {SPL_HIGH_THRESHOLD}\n")
        f.write("  - 说明: 成功到达目标位置，且路径效率高（SPL接近1.0）\n")
        f.write(f"  - 判断依据: `success == 1 AND spl > {SPL_HIGH_THRESHOLD}`\n\n")
        f.write(f"- **低效成功**: success=1 且 SPL ≤ {SPL_HIGH_THRESHOLD}\n")
        f.write("  - 说明: 成功到达目标位置，但路径效率较低（可能走了弯路）\n")
        f.write(f"  - 判断依据: `success == 1 AND spl <= {SPL_HIGH_THRESHOLD}`\n\n")
        
        f.write("#### 失败类（Oracle成功）\n\n")
        f.write(f"- **停止位置不对**: success=0, oracle_success=1, steps≤{STEPS_NORMAL_MAX}, distance≤{DISTANCE_CLOSE_MAX}米\n")
        f.write("  - 说明: Oracle可以成功，但实际导航在接近目标时停止位置略有偏差（可能超过去了一点点）\n")
        f.write(f"  - 判断依据: `success == 0 AND oracle_success == 1 AND steps <= {STEPS_NORMAL_MAX} AND distance <= {DISTANCE_CLOSE_MAX}`\n\n")
        f.write(f"- **绕圈**: success=0, oracle_success=1, steps>{STEPS_LARGE_MIN}\n")
        f.write("  - 说明: Oracle可以成功，但实际导航步数过多，说明一直在绕圈，无法找到正确路径\n")
        f.write(f"  - 判断依据: `success == 0 AND oracle_success == 1 AND steps > {STEPS_LARGE_MIN}`\n\n")
        
        f.write("#### 失败类（Oracle失败）\n\n")
        f.write(f"- **早停**: success=0, oracle_success=0, steps≤{STEPS_NORMAL_MAX}\n")
        f.write("  - 说明: 步数较少就停止了，说明模型过早发出了停止信号\n")
        f.write(f"  - 判断依据: `success == 0 AND oracle_success == 0 AND steps <= {STEPS_NORMAL_MAX}`\n\n")
        f.write(f"- **走偏**: success=0, oracle_success=0, distance>{DISTANCE_FAR_MIN}米\n")
        f.write("  - 说明: 距离目标很远，说明导航过程中走偏了方向，偏离了正确路径\n")
        f.write(f"  - 判断依据: `success == 0 AND oracle_success == 0 AND distance > {DISTANCE_FAR_MIN}`\n\n")
        f.write("- **其他失败**: 不符合上述任何分类条件的失败case\n")
        f.write("  - 说明: 其他类型的失败情况，需要进一步分析\n")
        f.write("  - 判断依据: 不属于上述任何分类的失败episode\n\n")
        
        # 阈值说明
        f.write("## 5. 分类阈值说明\n\n")
        f.write("| 参数 | 阈值 |\n")
        f.write("|------|------|\n")
        f.write(f"| Steps正常范围上限 | {STEPS_NORMAL_MAX} |\n")
        f.write(f"| Steps很大下限 | {STEPS_LARGE_MIN} |\n")
        f.write(f"| Distance接近上限 | {DISTANCE_CLOSE_MAX}米 |\n")
        f.write(f"| Distance很远下限 | {DISTANCE_FAR_MIN}米 |\n")
        f.write(f"| SPL高效阈值 | {SPL_HIGH_THRESHOLD} |\n")
        f.write(f"| Instruction简单上限 | {INSTRUCTION_SIMPLE_MAX}词 |\n")
        f.write(f"| Instruction复杂下限 | {INSTRUCTION_COMPLEX_MIN}词 |\n")
        f.write("\n")


def generate_csv_stats(
    overall_stats: Dict,
    scene_stats: Dict[str, Dict],
    complexity_stats: Dict[str, Dict],
    status_data: Dict[str, Dict],
    output_path: Path
):
    """生成CSV统计数据文件"""
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 整体统计
        writer.writerow(['Section', 'Metric', 'Value'])
        writer.writerow(['Overall', 'Total Episodes', overall_stats['count']])
        writer.writerow(['Overall', 'Success Rate', f"{overall_stats['success_rate']:.4f}"])
        writer.writerow(['Overall', 'Oracle Success Rate', f"{overall_stats['oracle_success_rate']:.4f}"])
        writer.writerow(['Overall', 'Avg SPL', f"{overall_stats['avg_spl']:.4f}"])
        writer.writerow(['Overall', 'Median SPL', f"{overall_stats['median_spl']:.4f}"])
        writer.writerow(['Overall', 'Avg Distance', f"{overall_stats['avg_distance']:.2f}"])
        writer.writerow(['Overall', 'Median Distance', f"{overall_stats['median_distance']:.2f}"])
        writer.writerow(['Overall', 'Avg Steps', f"{overall_stats['avg_steps']:.1f}"])
        writer.writerow(['Overall', 'Median Steps', f"{overall_stats['median_steps']:.0f}"])
        
        # 按Scene统计
        writer.writerow([])
        writer.writerow(['Scene Statistics'])
        writer.writerow(['Scene ID', 'Count', 'Success Rate', 'Oracle Success Rate', 
                        'Avg SPL', 'Median SPL', 'Avg Distance'])
        for scene_id in sorted(scene_stats.keys()):
            stats = scene_stats[scene_id]
            writer.writerow([
                scene_id,
                stats['count'],
                f"{stats['success_rate']:.4f}",
                f"{stats['oracle_success_rate']:.4f}",
                f"{stats['avg_spl']:.4f}",
                f"{stats['median_spl']:.4f}",
                f"{stats['avg_distance']:.2f}"
            ])
        
        # 按复杂度统计
        writer.writerow([])
        writer.writerow(['Complexity Statistics'])
        writer.writerow(['Complexity', 'Count', 'Success Rate', 'Oracle Success Rate',
                        'Avg SPL', 'Median SPL', 'Avg Distance'])
        for complexity in ['simple', 'standard', 'complex']:
            if complexity in complexity_stats:
                stats = complexity_stats[complexity]
                writer.writerow([
                    complexity,
                    stats['count'],
                    f"{stats['success_rate']:.4f}",
                    f"{stats['oracle_success_rate']:.4f}",
                    f"{stats['avg_spl']:.4f}",
                    f"{stats['median_spl']:.4f}",
                    f"{stats['avg_distance']:.2f}"
                ])
        
        # 状态分类统计
        writer.writerow([])
        writer.writerow(['Status Classification'])
        writer.writerow(['Status', 'Count', 'Percentage', 'Episode IDs'])
        total = sum(status_data[status]['count'] for status in status_data)
        status_order = ['高效成功', '低效成功', '停止位置不对', '绕圈', '早停', '走偏', '其他失败']
        for status in status_order:
            if status in status_data:
                count = status_data[status]['count']
                percentage = count / total if total > 0 else 0.0
                episode_ids = ','.join(status_data[status]['episode_ids'])
                writer.writerow([status, count, f"{percentage:.4f}", episode_ids])


def main():
    parser = argparse.ArgumentParser(
        description='分析SatNav评估结果，生成统计报告'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的all_results.json文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录路径（默认：输入文件所在目录下的analysis文件夹）'
    )
    
    args = parser.parse_args()
    
    # 读取数据
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    # 确定输出目录
    if args.output is None:
        # 默认：输入文件所在目录下的analysis文件夹
        output_dir = input_path.parent / 'analysis'
    else:
        output_dir = Path(args.output)
    
    print(f"正在读取数据文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        episodes = json.load(f)
    
    print(f"共读取 {len(episodes)} 个episode")
    
    # 执行分析
    print("正在分析数据...")
    overall_stats = calculate_statistics(episodes)
    scene_stats = analyze_by_scene(episodes)
    complexity_stats = analyze_by_complexity(episodes)
    status_data = analyze_by_status(episodes)
    
    # 生成输出文件
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = output_dir / 'analysis_report.md'
    csv_path = output_dir / 'analysis_stats.csv'
    
    print(f"正在生成Markdown报告: {md_path}")
    generate_markdown_report(
        overall_stats, scene_stats, complexity_stats, status_data, md_path
    )
    
    print(f"正在生成CSV统计文件: {csv_path}")
    generate_csv_stats(
        overall_stats, scene_stats, complexity_stats, status_data, csv_path
    )
    
    print("分析完成！")
    print(f"  - Markdown报告: {md_path}")
    print(f"  - CSV统计文件: {csv_path}")


if __name__ == '__main__':
    main()
