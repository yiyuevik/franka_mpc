# scripts/collect_multimodal_trajectories.py

import numpy as np
import os
import time
import json
import shutil
from datetime import datetime
import plotly.graph_objs as go
import plotly.io as pio
from concurrent.futures import ProcessPoolExecutor, as_completed
import config

from utils.helpers import  ensure_dir, generate_grid_initial_guesses
from controllers.mpc_ocp import create_ocp_solver, simulate_closed_loop
from simulators.mujoco_simulator import MuJoCoSimulator, simulate_closed_loop_mujoco

# ------------------------- Configurations -------------------------
# Toggle batch mode: True for large dataset/saving memory, False for all in memory
USE_BATCH = False
BATCH_SIZE = 1000  # Only works if USE_BATCH == True

# Threshold for clustering: max allowed euclidean distance per time step for two trajectories to be considered the same
CLUSTER_POS_THRESHOLD = 0.05  # meters

# Save end-effector trajectory, joint states, etc
SAVE_POS_DIM = 3

N_SIM = 150  # Simulation horizon

def run_single_sim(i, u_guess, x0, N_sim):
    try:
        ocp, ocp_solver, _ = create_ocp_solver(x0)
        mujoco_sim = MuJoCoSimulator()
        t, simX, simU, simCost, success, pos = simulate_closed_loop_mujoco(
            ocp, ocp_solver, mujoco_sim, x0, u_guess, N_sim=N_sim
        )
        if success:
            simX_with_pos = np.hstack([simX, pos])
            return {
                "i": i,
                "success": True,
                "u_guess": u_guess,
                "simX": simX_with_pos,
                "simU": simU,
                "simCost": simCost
            }
        else:
            return {
                "i": i,
                "success": False,
                "u_guess": u_guess
            }
    except Exception as e:
        return {
            "i": i,
            "success": False,
            "u_guess": u_guess,
            "error": str(e)
        }
# ------------------------- Main Script -------------------------

def main():
    # --------- 0. Prepare output directory ---------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"data/mpc_multimodal_{timestamp}"
    ensure_dir(save_dir)
    print(f"Saving all output to: {save_dir}")

    # --------- 1. Prepare grid initial guess ---------
    all_initial_guesses = generate_grid_initial_guesses(config.U4_MIN, config.U4_MAX, config.STEP)
    total_combinations = len(all_initial_guesses)
    print(f"Total initial guess combinations: {total_combinations}")

    # ---------- 2. Run All Simulations in multiprocess ---------
    x0 = np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi], dtype=float)

    print("Starting simulation loop...")
    time_begin = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=(os.cpu_count()-2)) as executor:
        futures = [
            executor.submit(run_single_sim, i, u_guess, x0, N_SIM)
            for i, u_guess in enumerate(all_initial_guesses)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    # --------- 2. Storage Buffers ---------
    buffer_success = {
        'indexes': [],
        'initial_guesses': [],
        'simX': [],
        'simU': [],
        'simCost': [],
    }
    buffer_fail = {
        'indexes': [],
        'initial_guesses': [],
    }

    for r in results:
        if r["success"]:
            buffer_success['indexes'].append(r["i"])
            buffer_success['initial_guesses'].append(r["u_guess"])
            buffer_success['simX'].append(r["simX"])
            buffer_success['simU'].append(r["simU"])
            buffer_success['simCost'].append(r["simCost"])
        else:
            buffer_fail['indexes'].append(r["i"])
            buffer_fail['initial_guesses'].append(r["u_guess"])

    success_count = len(buffer_success['indexes'])
    fail_count = len(buffer_fail['indexes'])

    if not USE_BATCH:
        np.save(os.path.join(save_dir, "success_indexes.npy"), np.array(buffer_success['indexes']))
        np.save(os.path.join(save_dir, "success_initial_guesses.npy"), np.array(buffer_success['initial_guesses']))
        np.save(os.path.join(save_dir, "success_simX.npy"), np.array(buffer_success['simX']))
        np.save(os.path.join(save_dir, "success_simU.npy"), np.array(buffer_success['simU']))
        np.save(os.path.join(save_dir, "success_simCost.npy"), np.array(buffer_success['simCost']))

        np.save(os.path.join(save_dir, "fail_indexes.npy"), np.array(buffer_fail['indexes']))
        np.save(os.path.join(save_dir, "fail_initial_guesses.npy"), np.array(buffer_fail['initial_guesses']))

    time_used = time.time() - time_begin
    print(f"\nTotal finished: {success_count + fail_count}, Success: {success_count}, Fail: {fail_count}, Time used: {time_used/60:.1f} min")

    success_indexes = np.load(os.path.join(save_dir, "success_indexes.npy"))
    success_initial_guesses = np.load(os.path.join(save_dir, "success_initial_guesses.npy"))
    success_simX = np.load(os.path.join(save_dir, "success_simX.npy"))
    success_simU = np.load(os.path.join(save_dir, "success_simU.npy"))
    success_simCost = np.load(os.path.join(save_dir, "success_simCost.npy"))

    N_success = success_simX.shape[0]
    print("Clustering all successful trajectories...")

    pos_traj = success_simX[:, :, -SAVE_POS_DIM:]
    clusters = []
    rep_traj_list = []
    cluster_assignment = [-1] * N_success
    threshold_sq = CLUSTER_POS_THRESHOLD ** 2

    for i, traj in enumerate(pos_traj):
        assigned = False
        for cid, rep_traj in enumerate(rep_traj_list):
            if np.sum((rep_traj[-1] - traj[-1])**2) > threshold_sq:
                continue
            max_sq = np.max(np.sum((rep_traj - traj) ** 2, axis=1))
            if max_sq <= threshold_sq:
                clusters[cid].append(i)
                cluster_assignment[i] = cid
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            cluster_assignment[i] = len(clusters) - 1
            rep_traj_list.append(traj)

    cluster_count = len(clusters)
    print(f"Found {cluster_count} unique trajectory modes")

    cluster_info = {
        "num_clusters": cluster_count,
        "total_success": int(N_success),
        "total_failed": int(fail_count),
        "clusters": []
    }
    for cid, members in enumerate(clusters):
        rep_idx = members[0]
        rep_init = success_initial_guesses[rep_idx].tolist()
        rep_cost = float(np.sum(success_simCost[rep_idx]))
        cluster_info["clusters"].append({
            "cluster_id": cid,
            "count": len(members),
            "representative_initial_guess": rep_init,
            "representative_final_cost": rep_cost,
            "trajX_file": f"cluster_{cid:03d}_trajX.npy",
            "trajU_file": f"cluster_{cid:03d}_trajU.npy",
            "cost_file": f"cluster_{cid:03d}_cost.npy"
        })
        np.save(os.path.join(save_dir, f"cluster_{cid:03d}_trajX.npy"), success_simX[rep_idx])
        np.save(os.path.join(save_dir, f"cluster_{cid:03d}_trajU.npy"), success_simU[rep_idx])
        np.save(os.path.join(save_dir, f"cluster_{cid:03d}_cost.npy"), success_simCost[rep_idx])

    with open(os.path.join(save_dir, "cluster_info.json"), 'w') as f:
        json.dump(cluster_info, f, indent=2)
    np.save(os.path.join(save_dir, "cluster_assignment.npy"), np.array(cluster_assignment))

    print("Plotting 3D initial guess distribution (by trajectory mode)...")
    success_points = success_initial_guesses[:, [3,4,6]]
    fail_points = None
    if fail_count > 0:
        fail_initial_guesses = np.load(os.path.join(save_dir, "fail_initial_guesses.npy"))
        fail_points = fail_initial_guesses[:, [3,4,6]]

    colors = pio.templates['plotly'].layout.colorway
    cluster_colors = [colors[i % len(colors)] for i in range(cluster_count)]

    data_traces = []
    for cid, members in enumerate(clusters):
        pts = success_points[members]
        trace = go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=4, color=cluster_colors[cid]),
            name=f"Trajectory {cid}"
        )
        data_traces.append(trace)
    if fail_points is not None and fail_points.shape[0] > 0:
        trace_fail = go.Scatter3d(
            x=fail_points[:,0], y=fail_points[:,1], z=fail_points[:,2],
            mode='markers',
            marker=dict(size=3, color='gray'),
            name=f"Failed ({fail_points.shape[0]} pts)"
        )
        data_traces.append(trace_fail)
    fig = go.Figure(data=data_traces, layout=go.Layout(
        scene=dict(
            xaxis_title="u4",
            yaxis_title="u5",
            zaxis_title="u7",
        ),
        title="Initial Guess 3D Space - Trajectory Clusters"
    ))
    fig.write_html(os.path.join(save_dir, "initial_guess_clusters.html"), include_plotlyjs='cdn')
    print(f"Interactive 3D HTML plot saved to: {os.path.join(save_dir, 'initial_guess_clusters.html')}")
    print(f"All done. Results saved in: {save_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\nerror:", e)