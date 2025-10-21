"""
Batch-run MPC with different initial control guesses read from clusters.json.
For each representative_initial_guess, run closed-loop simulation and
print top-3 most influential joints based on L2 energy of simU columns.
"""

import os
import time
import json
import numpy as np
import sys
from pathlib import Path

import configs
from controllers.mpc_ocp import create_ocp_solver, simulate_closed_loop
from simulators.mujoco_simulator import MuJoCoSimulator, simulate_closed_loop_mujoco
from utils.helpers import generate_random_initial_guess
from utils.plotting import plot_trajectories, animate_trajectory

# ---------- User options ----------
CLUSTERS_JSON_PATH = "/app/data/mpc_multimodal_20250907_001743/cluster_info.json"   # 将你粘贴的 JSON 存到该文件
N_SIM = 80                             # 仿真步数
PLOT_EACH_RUN = False                  # 是否每次画图
SAVE_EACH_RUN = False                  # 是否保存每次的 simX/simU/pos
SAVE_DIR = "batch_runs"                # 保存目录（当 SAVE_EACH_RUN=True 时生效）
# ----------------------------------

def compute_influence_l2(simU: np.ndarray) -> np.ndarray:
    """
    Compute L2 energy per joint over time: I_j = ||simU[:, j]||_2
    simU shape: [T, 7]
    Returns: array of shape [7], larger means more influential.
    """
    if simU.ndim != 2 or simU.shape[1] != 7:
        raise ValueError(f"simU expected shape [T, 7], got {simU.shape}")
    return np.sqrt(np.sum(simU * simU, axis=0))

def print_top3_influential_joints(cluster_id: int, influence: np.ndarray):
    """
    Print top-3 joints (1-based index) with their influence scores.
    """
    idx_sorted = np.argsort(-influence)  # descending
    top3 = idx_sorted[:3]
    triples = [(int(j+1), float(influence[j])) for j in top3]  # (joint_index_1based, score)
    print(f"[Cluster {cluster_id:03d}] Top-3 influential joints (by L2 energy of simU): {triples}")

def load_clusters(path: str):
    p = Path(path)
    if not p.exists():
        # 兼容备用文件名
        alt = Path("clusters_data.json")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(
                f"未找到 {path}（或 clusters_data.json）。请把你提供的 JSON 存为 {path}。"
            )
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 基础字段检查
    if "clusters" not in data or not isinstance(data["clusters"], list):
        raise ValueError("JSON 中缺少 'clusters' 列表。")
    return data

def main():
    # 1) 初始关节角（你的原代码）
    x0 = np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi], dtype=float)

    # 2) 读取聚类 JSON
    data = load_clusters(CLUSTERS_JSON_PATH)
    clusters = data["clusters"]
    print(f"Loaded {len(clusters)} clusters from '{CLUSTERS_JSON_PATH}'.")

    # 3) 初始化模拟器（物理环境可以复用）
    mujoco_sim = MuJoCoSimulator()

    if SAVE_EACH_RUN:
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # 4) 遍历每个簇的代表初值，逐个仿真
    for c in clusters:
        cluster_id = c.get("cluster_id")
        u_guess_list = c.get("representative_initial_guess", None)
        if u_guess_list is None or len(u_guess_list) != 7:
            print(f"[Cluster {cluster_id}] 跳过：代表初值缺失或维度非 7。")
            continue

        u_guess = np.array(u_guess_list, dtype=float)

        # 每个簇都重新创建 OCP/solver，避免状态污染
        ocp, ocp_solver, integrator = create_ocp_solver(x0)

        start_time = time.time()
        try:
            t, simX, simU, simCost, success, pos = simulate_closed_loop_mujoco(
                ocp, ocp_solver, mujoco_sim, x0, u_guess, N_sim=N_SIM
            )
        except Exception as e:
            print(f"[Cluster {cluster_id:03d}] 仿真异常：{e}")
            continue
        elapsed = time.time() - start_time

        # 输出基本信息
        print(f"\n=== Cluster {cluster_id:03d} run ===")
        print(f"Success: {bool(success)} | Steps: {simX.shape[0]} | Time: {elapsed:.4f}s ({elapsed/max(1, simX.shape[0]):.4f}s/step)")

        # 5) 计算并打印影响力 Top-3（基于 simU 的 L2 能量）
        try:
            influence = compute_influence_l2(simU)
            print_top3_influential_joints(cluster_id, influence)
        except Exception as e:
            print(f"[Cluster {cluster_id:03d}] 影响力计算失败：{e}")

        # 6) 可选：保存数据/画图
        if SAVE_EACH_RUN:
            np.savez(
                Path(SAVE_DIR) / f"cluster_{cluster_id:03d}_run.npz",
                t=t, simX=simX, simU=simU, simCost=simCost, success=success, pos=pos,
                u_guess=u_guess, x0=x0
            )
        if PLOT_EACH_RUN:
            try:
                plot_trajectories(simX[:, :7], simU, pos, target_position=configs.target_position)
            except Exception as e:
                print(f"[Cluster {cluster_id:03d}] 绘图失败：{e}")

    print("\n所有簇已完成遍历。")

if __name__ == "__main__":
    main()
