# scripts/collect_multimodal_trajectories.py

import numpy as np
import os
import time
import json
import shutil
from datetime import datetime
import plotly.graph_objs as go
import plotly.io as pio
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

N_SIM = 400  # Simulation horizon


# ------------------------- Main Script -------------------------

def main():
    # --------- 0. Prepare output directory ---------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"mpc_multimodal_{timestamp}"
    ensure_dir(save_dir)
    print(f"Saving all output to: {save_dir}")

    # --------- 1. Prepare grid initial guess ---------
    all_initial_guesses = generate_grid_initial_guesses(config.U4_MIN, config.U4_MAX, config.STEP)
    total_combinations = len(all_initial_guesses)
    print(f"Total initial guess combinations: {total_combinations}")


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

    batch_id = 0
    success_count, fail_count = 0, 0

    # --------- 3. Run All Simulations ---------
    x0 = np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi,
                   0, 0, 0, 0, 0, 0, 0], dtype=float)
    ocp, ocp_solver, integrator = create_ocp_solver(x0)
    mujoco_sim = MuJoCoSimulator()
    print("Starting simulation loop...")
    time_begin = time.time()
    for i, u_guess in enumerate(all_initial_guesses):
        print(f"Sim {i+1}/{total_combinations} | u4={u_guess[3]:.1f}, u5={u_guess[4]:.1f}, u7={u_guess[6]:.1f}")
        try:
            ocp_solver.set(0, "x", x0)
            for j in range(config.Horizon):
                ocp_solver.set(j, "u", u_guess)

            t, simX, simU, simCost, success, pos = simulate_closed_loop_mujoco(
                ocp, ocp_solver, mujoco_sim, x0, N_sim=N_SIM
            )

            if success:
                # Compute FK for each step
                simX_with_pos = np.hstack([simX, pos])
                buffer_success['indexes'].append(i)
                buffer_success['initial_guesses'].append(u_guess)
                buffer_success['simX'].append(simX_with_pos)
                buffer_success['simU'].append(simU)
                buffer_success['simCost'].append(simCost)
                success_count += 1
            else:
                buffer_fail['indexes'].append(i)
                buffer_fail['initial_guesses'].append(u_guess)
                fail_count += 1

        except Exception as e:
            print(f"Simulation error: {e}")
            buffer_fail['indexes'].append(i)
            buffer_fail['initial_guesses'].append(u_guess)
            fail_count += 1
        ocp_solver.reset()
        # ----- Batch Save -----
        if USE_BATCH and ((i + 1) % BATCH_SIZE == 0 or (i + 1) == total_combinations):
            batch_id += 1
            np.save(os.path.join(save_dir, f"success_indexes_batch{batch_id}.npy"), np.array(buffer_success['indexes']))
            np.save(os.path.join(save_dir, f"success_initial_guesses_batch{batch_id}.npy"), np.array(buffer_success['initial_guesses']))
            np.save(os.path.join(save_dir, f"success_simX_batch{batch_id}.npy"), np.array(buffer_success['simX']))
            np.save(os.path.join(save_dir, f"success_simU_batch{batch_id}.npy"), np.array(buffer_success['simU']))
            np.save(os.path.join(save_dir, f"success_simCost_batch{batch_id}.npy"), np.array(buffer_success['simCost']))

            np.save(os.path.join(save_dir, f"fail_indexes_batch{batch_id}.npy"), np.array(buffer_fail['indexes']))
            np.save(os.path.join(save_dir, f"fail_initial_guesses_batch{batch_id}.npy"), np.array(buffer_fail['initial_guesses']))

            buffer_success = {k: [] for k in buffer_success}
            buffer_fail = {k: [] for k in buffer_fail}

    # Save final results if not batched (or leftovers from final batch)
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

    # --------- 4. Merge Batch Results if needed ---------


    # --------- 5. Load ALL successful data into memory for clustering ---------
    if not USE_BATCH:
        success_indexes = np.load(os.path.join(save_dir, "success_indexes.npy"))
        success_initial_guesses = np.load(os.path.join(save_dir, "success_initial_guesses.npy"))
        success_simX = np.load(os.path.join(save_dir, "success_simX.npy"))
        success_simU = np.load(os.path.join(save_dir, "success_simU.npy"))
        success_simCost = np.load(os.path.join(save_dir, "success_simCost.npy"))
    else:
        raise NotImplementedError("If you use batch mode, please implement a batch merge utility first!")

    N_success = success_simX.shape[0]

    # --------- 6. Cluster simX trajectories into modes ---------
    print("Clustering all successful trajectories...")

    # Use only end-effector xyz path for clustering (simX最后三维)
    pos_traj = success_simX[:, :, -SAVE_POS_DIM:]  # (N_success, T+1, 3)
    clusters = []
    rep_traj_list = []
    cluster_assignment = [-1] * N_success
    threshold_sq = CLUSTER_POS_THRESHOLD ** 2

    for i, traj in enumerate(pos_traj):
        assigned = False
        for cid, rep_traj in enumerate(rep_traj_list):
            # Compare final point first (fast reject)
            if np.sum((rep_traj[-1] - traj[-1])**2) > threshold_sq:
                continue
            # Check max deviation along entire trajectory
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

    # --------- 7. Save Cluster Information and Representative Trajectories ---------
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
        # Save representative trajectory to disk
        np.save(os.path.join(save_dir, f"cluster_{cid:03d}_trajX.npy"), success_simX[rep_idx])
        np.save(os.path.join(save_dir, f"cluster_{cid:03d}_trajU.npy"), success_simU[rep_idx])
        np.save(os.path.join(save_dir, f"cluster_{cid:03d}_cost.npy"), success_simCost[rep_idx])
    # Save cluster mapping and info
    with open(os.path.join(save_dir, "cluster_info.json"), 'w') as f:
        json.dump(cluster_info, f, indent=2)
    np.save(os.path.join(save_dir, "cluster_assignment.npy"), np.array(cluster_assignment))

    # --------- 8. Save a 3D Interactive Plot of Initial Guess Space Colored by Cluster ---------
    print("Plotting 3D initial guess distribution (by trajectory mode)...")
    success_points = success_initial_guesses[:, [3,4,6]]  # u4, u5, u7
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
    main()
