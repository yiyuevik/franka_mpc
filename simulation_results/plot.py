import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = '/mnt/c/Users/ASUS/Desktop/franka_mpc/simulation_results/all_simX_20250624_213554.npy'

# 加载数据
data = np.load(file_path)

# 指定要绘制的 sim 索引（0-based）和最后三个 state 索引
sim_indices = [3, 6, 24, 44, 52, 55, 56, 63, 67, 87, 97] 
state_indices = [-3, -2, -1]  # 最后三个 state

# 创建 4x2 子图
fig, axes = plt.subplots(3, 4, figsize=(12, 16), sharex=True)
axes = axes.flatten()

for ax, sim_idx in zip(axes, sim_indices):
    x = np.arange(data.shape[0])  # 样本索引 0-400
    for s in state_indices:
        ax.plot(x, data[:, s, sim_idx], label=f'state{17 + s + 1}')  # 计算实际 state 编号
    ax.set_title(f'sim{sim_idx + 1}')
    ax.set_xlabel('Sample Index (0-400)')
    ax.set_ylabel('State Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
