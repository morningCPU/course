import matplotlib.pyplot as plt
import numpy as np

# 解决matplotlib中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# ===================== 正确数据 =====================
p = np.array([1, 2, 4])
t_p = np.array([0.01184, 0.00622, 0.00335])
actual_S = t_p[0] / t_p  # 实测加速比

# Amdahl 理论值（f≈0.93）
f = 0.93
amdahl_S = 1 / ((1 - f) + (f / p))

# ===================== 绘图 =====================
plt.figure(figsize=(8, 5.5), dpi=300)

# 实测加速比折线
plt.plot(p, actual_S, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, 
         label=f'实测加速比\nS(2)={actual_S[1]:.2f}, S(4)={actual_S[2]:.2f}')

# Amdahl 理论加速比折线
plt.plot(p, amdahl_S, marker='s', linestyle='--', color='#d62728', linewidth=2, 
         label=f'Amdahl 理论值 (f≈{f})')

# 图表细节
plt.xlabel('MPI 进程数 ($p$)', fontsize=11, fontweight='bold')
plt.ylabel('加速比 ($S$)', fontsize=11, fontweight='bold')
plt.title('性能对比：实测加速比 vs. Amdahl 理论加速比', fontsize=13, pad=15, fontweight='bold')

plt.xticks(p)
plt.xlim(0.5, 4.5)
plt.ylim(0, 4.0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('amdahl_speedup_comparison.png', dpi=300)
plt.show()