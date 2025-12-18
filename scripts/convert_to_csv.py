import numpy as np
import pandas as pd
from pathlib import Path

# ===== 路径配置 =====
ROOT_DIR = Path("data/multimodality/parallel_rollout_20251105_1414")  # 解压后的根目录
OUT_DIR  = Path("data/copy/sim_q_trajs_mpc")                 # 输出给 Pinocchio 的目录
OUT_DIR.mkdir(exist_ok=True)

# 每条轨迹的时间间隔，随便定一个（可视化脚本不真正用时间）
DT = 0.2

for g in range(100):
    npy_path = ROOT_DIR / f"group_{g:02d}" / "main_simX.npy"
    if not npy_path.exists():
        print(f"[WARN] {npy_path} 不存在，跳过")
        continue

    X = np.load(npy_path)  # 形状 (100, 21, 7)
    if X.ndim != 3 or X.shape[2] != 7:
        raise ValueError(f"{npy_path} 形状异常: {X.shape}")

    # === 核心变化在这里 ===
    # 21 是 horizon，只取 horizon 第一个：X[:, 0, :] -> 形状 (100, 7)
    q_traj = X[:, 0, :]          # (time=100, 7 joints)

    T = q_traj.shape[0]          # 100
    t = np.arange(T) * DT        # 时间戳 [0, DT, 2*DT, ...]

    # 组织成 DataFrame，列名符合你的可视化脚本要求
    df = pd.DataFrame(
        q_traj,
        columns=[f"q{i}" for i in range(1, 8)],
    )
    df.insert(0, "stamp_sec", t)

    # 每个 group 变成一条轨迹：q_now_0.csv ~ q_now_99.csv
    csv_name = f"q_now_{g}.csv"
    csv_path = OUT_DIR / csv_name
    df.to_csv(csv_path, index=False)

    print(f"✅ group_{g:02d} -> {csv_name}  (形状 {q_traj.shape})")

print("全部转换完成。")
