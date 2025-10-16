import json
import matplotlib
matplotlib.use('Agg')  # 不显示图形窗口，直接保存
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 读取JSON文件
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载数据
vllm_data = load_json_data('demo/vllm.json')
light_data = load_json_data('demo/light.json')

# 确保两个文件的键是相同的
assert set(vllm_data.keys()) == set(light_data.keys()), "两个JSON文件的文件列表不一致"

# 1. 计算平均时间
vllm_avg = np.mean(list(vllm_data.values()))
light_avg = np.mean(list(light_data.values()))

print("=" * 60)
print("处理时间统计分析")
print("=" * 60)
print(f"\n总文件数: {len(vllm_data)}")
print(f"\nVLLM 平均处理时间: {vllm_avg:.4f} 秒")
print(f"Light 平均处理时间: {light_avg:.4f} 秒")
print(f"平均时间差: {abs(vllm_avg - light_avg):.4f} 秒")
print(f"速度提升比: {vllm_avg / light_avg:.2f}x" if light_avg < vllm_avg else f"{light_avg / vllm_avg:.2f}x")

# 统计信息
print(f"\nVLLM 统计:")
print(f"  最小时间: {min(vllm_data.values()):.4f} 秒")
print(f"  最大时间: {max(vllm_data.values()):.4f} 秒")
print(f"  中位数: {np.median(list(vllm_data.values())):.4f} 秒")
print(f"  标准差: {np.std(list(vllm_data.values())):.4f} 秒")

print(f"\nLight 统计:")
print(f"  最小时间: {min(light_data.values()):.4f} 秒")
print(f"  最大时间: {max(light_data.values()):.4f} 秒")
print(f"  中位数: {np.median(list(light_data.values())):.4f} 秒")
print(f"  标准差: {np.std(list(light_data.values())):.4f} 秒")

# 2. 创建对比数据框
comparison_df = pd.DataFrame({
    'file_name': list(vllm_data.keys()),
    'vllm_time': list(vllm_data.values()),
    'light_time': [light_data[k] for k in vllm_data.keys()]
})
comparison_df['time_diff'] = comparison_df['vllm_time'] - comparison_df['light_time']
comparison_df['speedup'] = comparison_df['vllm_time'] / comparison_df['light_time']

# 保存对比数据到CSV
comparison_df.to_csv('demo/time_comparison.csv', index=False, encoding='utf-8')
print(f"\n对比数据已保存到: demo/time_comparison.csv")

# 创建img目录
import os
os.makedirs('demo/img', exist_ok=True)

# 3. 绘图函数
def plot_average_time_comparison(vllm_avg, light_avg, save_path):
    """绘制平均时间对比柱状图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['VLLM', 'Light'], [vllm_avg, light_avg], color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Average Time (seconds)', fontsize=12)
    ax.set_title('Average Processing Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")

def plot_time_difference_distribution(comparison_df, save_path):
    """绘制时间差分布直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(comparison_df['time_diff'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero line')
    ax.set_xlabel('Time Difference (VLLM - Light) (seconds)', fontsize=12)
    ax.set_ylabel('File Count', fontsize=12)
    ax.set_title('Time Difference Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")

def plot_cdf_comparison(vllm_data, light_data, save_path):
    """绘制累积分布函数(CDF)对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_vllm = np.sort(list(vllm_data.values()))
    sorted_light = np.sort(list(light_data.values()))
    
    ax.plot(sorted_vllm, np.arange(1, len(sorted_vllm) + 1) / len(sorted_vllm), 
            label='VLLM', linewidth=2, color='#ff7f0e')
    ax.plot(sorted_light, np.arange(1, len(sorted_light) + 1) / len(sorted_light), 
            label='Light', linewidth=2, color='#2ca02c')
    
    ax.set_xlabel('Processing Time (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")

# 生成各个图表
print("\n开始生成图表...")
plot_average_time_comparison(vllm_avg, light_avg, 'demo/img/average_time_comparison.png')
plot_time_difference_distribution(comparison_df, 'demo/img/time_difference_distribution.png')
plot_cdf_comparison(vllm_data, light_data, 'demo/img/cdf_comparison.png')
print("所有图表生成完成！")

# 4. 找出处理时间差异最大的文件
print("\n" + "=" * 60)
print("TOP 10 VLLM更慢的文件:")
print("=" * 60)
top_slower = comparison_df.nlargest(10, 'time_diff')[['file_name', 'vllm_time', 'light_time', 'time_diff']]
print(top_slower.to_string(index=False))

print("\n" + "=" * 60)
print("TOP 10 VLLM更快的文件:")
print("=" * 60)
top_faster = comparison_df.nsmallest(10, 'time_diff')[['file_name', 'vllm_time', 'light_time', 'time_diff']]
print(top_faster.to_string(index=False))

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)

