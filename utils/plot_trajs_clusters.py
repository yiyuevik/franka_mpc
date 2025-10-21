import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math

# 读取路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
folder_path = os.path.join(project_root, "data", "multimodality", "20250907_1633", "cluster")


files = sorted(glob.glob(os.path.join(folder_path, "cluster_*_trajX.npy")))

num_files = len(files)

if num_files < 6*6:
    rows = int(math.floor(math.sqrt(num_files)))
    cols = int(math.ceil(num_files / rows))
    plt.subplots(rows, cols)
else:
    # only plot up to 36 files
    rows, cols = 6, 6
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()

for i, file in enumerate(files):
    if i == rows * cols:
        break
    data = np.load(file, allow_pickle=True)
    end_effector_pos = data[:, -3:]
    timesteps = np.arange(data.shape[0])
    
    ax = axes[i]
    ax.plot(timesteps, end_effector_pos[:, 0], label='X')
    ax.plot(timesteps, end_effector_pos[:, 1], label='Y')
    ax.plot(timesteps, end_effector_pos[:, 2], label='Z')
    ax.set_title(os.path.basename(file))
    ax.grid(True)
    
    if i % cols == 0:
        ax.set_ylabel("Position")
    if i // cols == rows - 1:
        ax.set_xlabel("Timestep")

    # only add legend to the first subplot
    if i == 0:
        ax.legend()

# hide unused subplots
for j in range(len(files), rows * cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
