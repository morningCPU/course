import matplotlib.pyplot as plt
import numpy as np

# 中文兼容
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 基础参数
p = np.array([1, 2, 4])
T1 = 0.01184
# 实测耗时
t_p = np.array([0.01184, 0.00630, 0.00346])
actual_S = T1 / t_p

# Amdahl参数 f=0.95（串行5%，和示例图一致）
f = 0.95
amdahl_S = 1 / ((1 - f) + f / p)
# 理想线性加速 S=p
ideal_S = p

# 2. 绘图
plt.figure(figsize=(8, 5.5), dpi=300)

# 理想线性 绿色点线
plt.plot(p, ideal_S, marker='o', linestyle=':', color='#2ca02c', linewidth=2.5, label='理想线性加速')
# Amdahl理论 红色虚线
plt.plot(p, amdahl_S, marker='s', linestyle='--', color='#d62728', linewidth=2, label=f'Amdahl理论 (f={f})')
# 实测 蓝色实线
plt.plot(p, actual_S, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, label='实测加速比')

# 删掉了所有点数值标注代码

# 图表配置
plt.xlabel('进程数 $p$', fontsize=11, weight='bold')
plt.ylabel('加速比 $S$', fontsize=11, weight='bold')
plt.title('实测加速比 vs Amdahl理论 vs 理想线性加速', fontsize=13, pad=15, weight='bold')
plt.xticks(p)
plt.xlim(0.5, 4.5)
plt.ylim(0, 4.2)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('amdahl_three_curve.png', dpi=300)
plt.show()