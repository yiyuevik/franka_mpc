import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# 文件路径
file_path = '/mnt/c/Users/ASUS/Desktop/franka_mpc/mpc_data_batch1_u4_-20to-7.5_20250629_133209/batch_0007_successful_trajectories_X.npy'

# 加载数据
data = np.load(file_path)
print(f'Loaded X data shape: {data.shape}')

# 确保数据维度为 (N, T, S)：N=轨迹数, T=时间步, S=状态数
if data.ndim == 3 and data.shape[0] not in (401, 17):
    sims = data  # (N, T, S)
elif data.ndim == 3 and data.shape[0] in (401, 17):
    sims = data.transpose(2, 0, 1)  # (T, S, N) -> (N, T, S)
else:
    raise ValueError('无法识别的数据维度结构')

N, T, S = sims.shape
print(f'Number of trajectories: {N}, Time steps: {T}, States: {S}')

# 只绘制最后三个状态
state_indices = [-3, -2, -1]

# 布局：根据 N 自动计算行列
cols = 2
rows = math.ceil(N / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=True)
axes = axes.flatten()

for idx in range(N):
    ax = axes[idx]
    x = np.arange(T)
    for si in state_indices:
        ax.plot(x, sims[idx, :, si], label=f'state{si + 1}')
    ax.set_title(f'Trajectory {idx + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State Value')
    ax.legend()
    ax.grid(True)

# 删除多余子图
for j in range(N, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



cols = 2
rows = math.ceil(N / cols)
fig = plt.figure(figsize=(cols * 6, rows * 4))
axes = [fig.add_subplot(rows, cols, i + 1, projection='3d') for i in range(rows * cols)]

for idx in range(N):
    ax = axes[idx]
    x = np.arange(T)
    ax.plot(sims[idx, :, -3], sims[idx, :, -2], sims[idx, :, -1], 'b-', label='End Effector Path')
    ax.scatter(sims[idx, 0, -3], sims[idx, 0, -2], sims[idx, 0, -1], 
                c='g', s=100, label='Start')
    ax.scatter(0.3, 0.3, 0.5, 
                c='r', s=100, label='Target')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End Effector Trajectory')
    ax.legend()
    ax.grid(True)

# 删除多余子图
for j in range(N, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()