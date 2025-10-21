
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import time
from utils.helpers import sample_states_around, ensure_dir, generate_random_initial_guess
from utils.helpers import  get_guess_from_solver_result, clear_solver_state, get_traj
import configs
from controllers.mpc_ocp import create_ocp_solver
from simulators.mujoco_simulator import MuJoCoSimulator
import matplotlib.pyplot as plt

# --- Task function: Solve OCP for multiple disturbed points ---
def solve_branches_for_one_point(x_list, N_horizon):
    

    results = []
    x0 = x_list[0]  # Use the first point as x0 for solver initialization
    for x in x_list:
        try:
            ocp, solver, _   = create_ocp_solver(x)
            for j in range(configs.Horizon):
                solver.set(j, "x", x)
            solver.set(configs.Horizon, "x", x)
            solver.solve_for_x0(x0_bar=x)
            x_traj, u_traj, Pos_traj = get_traj(ocp_solver=solver, N=N_horizon, nx=configs.Num_State, nu=configs.Num_Input)
            # cost = solver.get_cost()
            results.append({
                "x0": x,
                "u_traj": u_traj,
                "x_traj": x_traj,
                "pos_traj": Pos_traj,
                # "cost": cost
            })
            solver.reset()
        except Exception as e:
            print(f"  Branch OCP failed: {e}")
    return results


# ===== Main trajectory rollout for each group (executed within each process) ====
# This function runs the main trajectory for a given group, sampling branch points and solving OCP
# for each sampled point asynchronously using a shared executor.
# It saves the main trajectory and branch results to a specified directory.

def run_main_group(x0_init, group_id, N_step, N_horizon, n_branch, n_branch_workers, save_root, nMaxGuess=3):

    save_dir = os.path.join(save_root, f"group_{group_id:02d}")
    ensure_dir(save_dir)

    mujoco_sim = MuJoCoSimulator()
    ocp, ocp_solver, _ = create_ocp_solver(x0_init)
    mujoco_sim.reset(q_init=x0_init, qd_init=np.zeros(7))

    simX_main = []
    simU_main = []
    X_traj = np.zeros((N_step+1, configs.Num_State))
    # simCost_main = []
    # pos = []
    
    branch_pool = ProcessPoolExecutor(max_workers=n_branch_workers)
    branch_pool_th = ThreadPoolExecutor(max_workers=n_branch_workers)
    branch_features = []


    u_guess = generate_random_initial_guess()
    for j in range(configs.Horizon):
        ocp_solver.set(j, "x", x0_init)
        ocp_solver.set(j, "u", u_guess)
    ocp_solver.set(configs.Horizon, "x", x0_init)
    x_curr = x0_init.copy()
    X_traj[0, :] = x0_init
    branch_points = sample_states_around(x_curr, n_branch)
    future = branch_pool_th.submit(solve_branches_for_one_point, branch_points, N_horizon)
    branch_features.append((0, future))
    # ======================== generate data for control step loop =============================
    for step in range(N_step):
        retries = 0
        success = True
        while retries < nMaxGuess:
            try:
                # Solve MPC for current state
                u_opt = ocp_solver.solve_for_x0(x0_bar=x_curr)
                # Prepare warm-start for next iteration
                x_traj, u_traj, Pos_traj = get_traj(ocp_solver=ocp_solver, N=N_horizon, nx=configs.Num_State, nu=configs.Num_Input)
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, configs.Horizon)
                clear_solver_state(ocp_solver, configs.Horizon)
                for j in range(configs.Horizon):
                    ocp_solver.set(j, "u", u_guess[:, j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(configs.Horizon, "x", x_guess[:, -1])
                # Apply control and simulate one step in MuJoCo
                simU_main.append(u_traj)
                simX_main.append(x_traj)
                x_curr = mujoco_sim.step(u_traj[0])
                X_traj[step+1, :] = x_curr
                # create a plot of Pos_traj and save to file
                # plt.figure()
                # plt.plot(x_traj, label='State Trajectory')
                # plt.xlabel('Time step')
                # plt.ylabel('Position')
                # plt.title(f'Predicted Position Trajectory at Step {step}')
                # plt.grid()
                # new_dir_x_traj = os.path.join(save_dir, f"normal_x_traj")
                # os.makedirs(new_dir_x_traj, exist_ok=True)
                # plt.savefig(os.path.join(new_dir_x_traj, f'predicted_position_step_{step:03d}.png'))
                # plt.close() 


                # plt.figure()
                # plt.plot(u_traj, label='input Trajectory')
                # plt.xlabel('Time step')
                # plt.ylabel('Position')
                # plt.title(f'Predicted Input Trajectory at Step {step}')
                # plt.grid()
                # new_dir_u_traj = os.path.join(save_dir, f"normal_u_traj")
                # os.makedirs(new_dir_u_traj, exist_ok=True)
                # plt.savefig(os.path.join(new_dir_u_traj, f'predicted_input_step_{step:03d}.png'))
                # plt.close()   
                # pos.append(mujoco_sim.get_end_effector_pos())
                # simCost_main.append(ocp_solver.get_cost())
                # Asynchronously submit branch tasks (using shared process pool)
                branch_points = sample_states_around(x_curr, n_branch)
                future = branch_pool.submit(solve_branches_for_one_point, branch_points, N_horizon)
                branch_features.append((step+1, future))
                break
            except Exception as e:
                ocp_solver.reset()
                u_guess = generate_random_initial_guess()
                for j in range(configs.Horizon):
                    ocp_solver.set(j, "x", x_curr)
                    ocp_solver.set(j, "u", u_guess)
                ocp_solver.set(configs.Horizon, "x", x_curr)
                print(f"Error in MPC solve: {e}. Retrying with a new initial guess u_guess: {u_guess}...")
                print(f"  Step {step}, retry {retries}")

            retries += 1
            if retries == nMaxGuess - 1:
                success = False
                print(f"  Step {step}, failed after {nMaxGuess} retries.")
        if not success:
            print(f"[Group {group_id} | Step {step}] MPC failed after {nMaxGuess} retries, stopping trajectory.")
            break
        

    # Save main trajectory
    np.save(os.path.join(save_dir, "main_simX.npy"), np.array(simX_main))
    np.save(os.path.join(save_dir, "main_simU.npy"), np.array(simU_main))
    # np.save(os.path.join(save_dir, "main_simCost.npy"), np.array(simCost_main))
    print(f"[Group {group_id}] Main trajectory completed, {len(simX_main)-1} steps.")

    import yaml
    success = len(simX_main) == N_step
    print(f"[len(simX_main), N_step]: {[len(simX_main), N_step]}")
    config_dict = {
        "N_step": N_step,
        "Horizon": N_horizon,
        "Num_Branch": n_branch,
        "Num_Group": configs.Num_Group,
        "ts": configs.Ts,
        "success": success,
        "actual_steps": len(simX_main) - 1
    }
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)


    # Collect and save branches
    branch_save_dir = os.path.join(save_dir, "branches_data")
    os.makedirs(branch_save_dir, exist_ok=True)
    for step, f in branch_features:
        try:
            branch_data = f.result()
            np.save(os.path.join(branch_save_dir, f"branch_step_{step:03d}.npy"), branch_data)
        except Exception as e:
            print(f"[Group {group_id}] Branch {step} failed: {e}")
    branch_pool.shutdown(wait=True)

    return x_traj

# --- Main process ---
def main():

    # Initial state
    x0 = configs.x0
    main_x0_list = sample_states_around(x0, configs.Num_Group)

    N_step = 100
    N_horizon = configs.Horizon
    n_branch = configs.Num_Noisy_Data

    save_dir = f"data/multimodality/parallel_rollout_{time.strftime('%Y%m%d_%H%M')}"
    ensure_dir(save_dir)

    # === Run multiple main trajectories in parallel ===
    max_group_parallel = 10
    n_branch_workers = 8 # Number of workers for branch solving within each group
    x_traj_all = []
    with ProcessPoolExecutor(max_workers=max_group_parallel) as group_pool:
        futures = []
        for gid, x0_init in enumerate(main_x0_list):
            f = group_pool.submit(run_main_group,
                                  x0_init, gid, N_step, N_horizon,
                                  n_branch, n_branch_workers, save_dir)
            futures.append(f)

        # Wait for all groups to complete
        for f in as_completed(futures):
            try:
                x_traj_all.append(f.result())
            except Exception as e:
                print(f"One group failed: {e}")

    x_traj_all = np.array(x_traj_all)
    clusters = []
    rep_traj_list = []
    cluster_assignment = [-1] * configs.Num_Group
    CLUSTER_POS_THRESHOLD = 1
    threshold_sq = CLUSTER_POS_THRESHOLD ** 2

    for i, traj in enumerate(x_traj_all):
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

    print(f"\n✅ All Group rollout + branch sampling completed, saved in: {save_dir}")
    import glob
    for ext in ('*.o', '*.so', '*.c'):
        for f in glob.glob(ext):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to remove {f}: {e}")
if __name__ == "__main__":
    main()
