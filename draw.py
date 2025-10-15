import matplotlib.pyplot as plt
import numpy as np

# 解决中文显示问题，请根据您的操作系统选择合适的字体
# Windows: plt.rcParams['font.sans-serif'] = ['SimHei']
# Mac: plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# Linux: 需要您安装中文字体，例如'WenQuanYi Zen Hei'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 1. 准备数据
categories = ['异常检测准确率', '根因定位速度', '故障修复成功率', '资源开销']
optimized_perf = [98.4, 3.0, 94.6, 15]
general_perf = [85, 30, 80, 40]

# 2. 设置绘图参数
x = np.arange(len(categories))  # x轴刻度的位置
width = 0.35  # 柱子的宽度

# "深海远航" 配色方案
color_optimized = '#005f73'
color_general = '#94d2bd'

# 3. 创建画布和子图
fig, ax = plt.subplots(figsize=(12, 7))

# 4. 绘制柱子
rects1 = ax.bar(x - width/2, optimized_perf, width, label='优化性能', color=color_optimized)
rects2 = ax.bar(x + width/2, general_perf, width, label='一般性能', color=color_general)

# 5. 添加图表元素和美化
ax.set_ylabel('性能数值 (%) / 时间 (秒)', fontsize=14)
ax.set_xticks(x)

# --- 在这里修改了字体大小 ---
ax.set_xticklabels(categories, fontsize=14) # 将字体大小从 12 调整为 14
# -----------------------------

ax.set_ylim(0, 105)
ax.legend(fontsize=12)

# 移除顶部和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 移除 x 轴和 y 轴的刻度线
ax.tick_params(axis='both', which='both', length=0)

# 6. 在柱子顶部添加数值标签
def add_labels(rects, data):
    """在每个柱子顶部添加带有单位的标签"""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        # 根据数据类别添加不同的单位
        unit = '秒' if i == 1 else '%'
        label = f'{data[i]}{unit}'
        
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 垂直方向向上偏移5个点
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11,
                    fontweight='bold')

add_labels(rects1, optimized_perf)
add_labels(rects2, general_perf)

# 7. 显示图表
plt.tight_layout()
plt.show()