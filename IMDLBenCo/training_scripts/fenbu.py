import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_cdf_comparison(scores_base, scores_diff):
    """
    绘制 CDF (累积分布) 对比图
    """
    # 构造 DataFrame
    df_base = pd.DataFrame({'Score': scores_base, 'Model': 'SegFormer (Baseline)'})
    df_diff = pd.DataFrame({'Score': scores_diff, 'Model': 'DiffIML (Ours)'})
    df = pd.concat([df_base, df_diff])

    # 设置画布
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid") # 使用网格背景，更清晰

    # === 核心：绘制 ECDF (Empirical Cumulative Distribution Function) ===
    # complementary=False (默认): 也就是 CDF，显示 "小于 x 的比例"
    ax = sns.ecdfplot(
        data=df, 
        x='Score', 
        hue='Model', 
        palette={'SegFormer (Baseline)': 'red', 'DiffIML (Ours)': 'green'}, 
        linewidth=3
    )

    # === 添加辅助线和标注 (让图会说话) ===
    
    # 1. 安全底噪线 (Noise Threshold) - 比如 0.05
    # 含义：我们容忍 0.05 的底噪，看看谁能达标
    noise_thresh = 0.05
    plt.axvline(noise_thresh, color='gray', linestyle='--', alpha=0.7)
    plt.text(noise_thresh + 0.01, 0.05, 'Safe Noise Limit (0.05)', color='gray', rotation=90, verticalalignment='bottom')

    # 2. 决策报警线 (Decision Threshold) - 通常是 0.5
    # 含义：过了这就误报了
    decision_thresh = 0.5
    plt.axvline(decision_thresh, color='black', linestyle='-', linewidth=2, alpha=0.5)
    plt.text(decision_thresh + 0.01, 0.05, 'Decision Threshold (0.5)', color='black', rotation=90, verticalalignment='bottom', fontweight='bold')

    # 3. 标注"差距" (The Gap)
    # 在 x=0.05 处，DiffIML 应该接近 100%，SegFormer 可能只有 80%
    # 这里我们手动加个箭头示意 (具体位置根据数据自动调整)
    # 这一步是锦上添花，如果不想手动调坐标可以注释掉
    '''
    plt.arrow(x=0.1, y=0.8, dx=0, dy=0.15, width=0.005, color='black')
    plt.text(0.12, 0.85, "Performance Gap\n(Robustness)", fontsize=12)
    '''

    # === 坐标轴美化 ===
    
    # 横轴：使用线性坐标，范围 0 到 0.6 (重点关注左半部分，因为右边通常没数据)
    plt.xlim(-0.01, 0.6) 
    plt.xlabel("Prediction Score on Authentic Images (Loss/Error)", fontsize=12, fontweight='bold')
    
    # 纵轴：显示百分比
    plt.ylabel("Proportion of Images within Error Limit", fontsize=12, fontweight='bold')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0)) # 格式化为 0% - 100%

    # 标题
    plt.title("Cumulative Distribution of Errors (CDF)\n(Steeper Curve = Better Robustness)", fontsize=14)
    
    # 图例
    plt.legend(title='Model', fontsize=11, loc='lower right')

    plt.tight_layout()
    save_path = 'cdf_robustness_proof.png'
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {save_path}")

# ================= 模拟数据与主程序 =================

def main():
    print("生成模拟数据中 (请替换为真实推理代码)...")
    
    # --- 模拟数据 (这里模拟了真实情况) ---
    # 1. DiffIML (绿线): 
    # 绝大多数分数集中在 0.001 ~ 0.02 之间，极少数到 0.04
    # 用 Beta 分布模拟这种"贴墙"的效果
    scores_diff = np.random.beta(a=1, b=50, size=500) 
    
    # 2. SegFormer (红线): 
    # 虽然大部分也在低分，但有"长尾"，很多分数组在 0.1 ~ 0.4
    # 用混合分布模拟：80% 正常，20% 误判(离群点)
    scores_base_normal = np.random.beta(a=2, b=15, size=400) # 正常部分
    scores_base_outlier = np.random.uniform(low=0.1, high=0.45, size=100) # 误判部分(棱锥等)
    scores_base = np.concatenate([scores_base_normal, scores_base_outlier])

    # --- 绘图 ---
    plot_cdf_comparison(scores_base, scores_diff)

if __name__ == '__main__':
    main()