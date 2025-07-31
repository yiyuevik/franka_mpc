
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import time
from utils.helpers import sample_states_around, ensure_dir, generate_random_initial_guess
import config

# --- Task function: Solve OCP for multiple disturbed points ---
def solve_branches_for_one_point(x_list, N_horizon):
    from controllers.mpc_ocp import create_ocp_solver
    from utils.helpers import get_guess_from_solver_result, clear_solver_state
    results = []
    for x in x_list:
        try:
            ocp, solver, _   = create_ocp_solver(x)
            ocp_solver.set(0, "x", x)
            u_guess = generate_random_initial_guess()
            for j in range(config.Horizon):
                ocp_solver.set(j, "u", u_guess)
            solver.solve_for_x0(x0_bar=x)
            u_guess, x_guess = get_guess_from_solver_result(solver, N_horizon)
            cost = solver.get_cost()
            results.append({
                "x0": x,
                "u_traj": u_guess,
                "x_traj": x_guess,
                "cost": cost
            })
            clear_solver_state(solver, N_horizon)
        except Exception as e:
            print(f"  Branch OCP failed: {e}")
    return results


# ===== Main trajectory rollout for each group (executed within each process) ====
# This function runs the main trajectory for a given group, sampling branch points and solving OCP
# for each sampled point asynchronously using a shared executor.
# It saves the main trajectory and branch results to a specified directory.

def run_main_group(x0_init, group_id, N_step, N_horizon, n_branch, shared_executor, save_root, nMaxGuess=3):
    from controllers.mpc_ocp import create_ocp_solver
    from simulators.mujoco_simulator import MuJoCoSimulator

    save_dir = os.path.join(save_root, f"group_{group_id:02d}")
    ensure_dir(save_dir)

    mujoco_sim = MuJoCoSimulator()
    ocp, ocp_solver, _ = create_ocp_solver(x0_init)
    mujoco_sim.reset(q_init=x0_init[:7], qd_init=x0_init[7:])

    simX_main = [x0_init]
    simU_main = []
    simCost_main = []
    # pos = []
    futures = []
    

    # ========================= first step of the main trajectory ========================

    ocp_solver.set(0, "x", x0_init)
    u_guess = generate_random_initial_guess()
    for j in range(config.Horizon):
        ocp_solver.set(j, "u", u_guess)
    
    retries = 0
    success = True
    while retries < nMaxGuess: 
        try:
            # Solve MPC for the initial state
            u_opt = ocp_solver.solve_for_x0(x0_bar=x0_init)
            # Prepare warm-start for next iteration
            u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
            clear_solver_state(ocp_solver, config.Horizon)
            for j in range(config.Horizon):
                ocp_solver.set(j, "u", u_guess[:, j])
                ocp_solver.set(j, "x", x_guess[:, j])
            ocp_solver.set(config.Horizon, "x", x_guess[:, -1])
            
            # Apply control and simulate one step in MuJoCo
            simU_main.append(u_opt)
            simX_main.append(mujoco_sim.step(u_opt))
            # pos.append(mujoco_sim.get_end_effector_pos())
            simCost_main.append(ocp_solver.get_cost())
            # Asynchronously submit branch tasks (using shared process pool)
            branch_points = sample_states_around(x0_init, n=n_branch)
            future = shared_executor.submit(solve_branches_for_one_point, branch_points, N_horizon)
            futures.append((0, future))
            break
        except Exception as e:
            ocp_solver.reset()
            ocp_solver.set(0, "x", x0_init)
            u_guess = generate_random_initial_guess()
            for j in range(config.Horizon):
                ocp_solver.set(j, "u", u_guess)
            retries += 1
            if retries == nMaxGuess - 1:
                success = False

    # ======================== generate data for control step loop =============================
    if success:
        for step in range(1, N_step):
            retries = 0
            success = True
            while retries < nMaxGuess:
                try:
                    # Solve MPC for current state
                    u_opt = ocp_solver.solve_for_x0(x0_bar=simX_main[-1])
                    # Prepare warm-start for next iteration
                    u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
                    clear_solver_state(ocp_solver, config.Horizon)
                    for j in range(config.Horizon):
                        ocp_solver.set(j, "u", u_guess[:, j])
                        ocp_solver.set(j, "x", x_guess[:, j])
                    ocp_solver.set(config.Horizon, "x", x_guess[:, -1])
                    
                    # Apply control and simulate one step in MuJoCo
                    simU_main.append(u_opt)
                    simX_main.append(mujoco_sim.step(u_opt))
                    # pos.append(mujoco_sim.get_end_effector_pos())
                    simCost_main.append(ocp_solver.get_cost())
                    # Asynchronously submit branch tasks (using shared process pool)
                    branch_points = sample_states_around(simX_main[-1], n=n_branch)
                    future = shared_executor.submit(solve_branches_for_one_point, branch_points, N_horizon)
                    futures.append((step, future))
                    break
                except Exception as e:
                    ocp_solver.reset()
                    ocp_solver.set(0, "x", simX_main[-1])
                    u_guess = generate_random_initial_guess()
                    for j in range(config.Horizon):
                        ocp_solver.set(j, "u", u_guess)
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
    np.save(os.path.join(save_dir, "main_simCost.npy"), np.array(simCost_main))
    print(f"[Group {group_id}] Main trajectory completed, {len(simX_main)-1} steps.")

    import yaml
    success = len(simX_main) == N_step + 1
    config_dict = {
        "N_step": N_step,
        "Horizon": N_horizon,
        "Num_Branch": n_branch,
        "Num_Group": config.Num_Group,
        "ts": config.Ts,
        "success": success,
        "actual_steps": len(simX_main) - 1
    }
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)


    # Collect and save branches
    for step, f in futures:
        try:
            branch_data = f.result()
            np.save(os.path.join(save_dir, f"branch_step_{step:03d}.npy"), branch_data)
        except Exception as e:
            print(f"[Group {group_id}] Branch {step} failed: {e}")


# --- Main process ---
def main():
    from controllers.mpc_ocp import create_ocp_solver
    from simulators.mujoco_simulator import MuJoCoSimulator

    # Initial state
    x0 = np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi,
                   0, 0, 0, 0, 0, 0, 0], dtype=float)
    main_x0_list = sample_states_around(x0, n= config.Num_Group)

    N_step = 100
    N_horizon = config.Horizon
    n_branch = config.Num_Noisy_Data

    save_dir = f"output/parallel_rollout_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(save_dir)

    # === Create shared branch process pool ===
    max_branch_workers = 8
    shared_executor = ProcessPoolExecutor(max_workers=max_branch_workers)

    # === Run multiple main trajectories in parallel ===
    max_group_parallel = 4
    with ProcessPoolExecutor(max_workers=max_group_parallel) as group_pool:
        futures = []
        for gid, x0_init in enumerate(main_x0_list):
            f = group_pool.submit(run_main_group,
                                  x0_init, gid, N_step, N_horizon,
                                  n_branch, shared_executor, save_dir)
            futures.append(f)

        # Wait for all groups to complete
        for f in as_completed(futures):
            pass

    shared_executor.shutdown(wait=True)

    print(f"\n✅ All Group rollout + branch sampling completed, saved in: {save_dir}")
if __name__ == "__main__":
    main()
