"""
Main script to set up and run the closed-loop MPC simulation for the Franka Panda arm.
This initializes the solver and simulator, runs the simulation, and produces visualization of the results.
"""
import os
import time
import numpy as np
import urdf2casadi.urdfparser as u2c

import config
from controllers.mpc_ocp import create_ocp_solver, simulate_closed_loop
from simulators.mujoco_simulator import MuJoCoSimulator, simulate_closed_loop_mujoco
from utils.helpers import generate_random_initial_guess
from utils.plotting import plot_trajectories, animate_trajectory

def main():
    # (Optional) set random seed for reproducibility
    # NUM_SEED = 7
    # np.random.seed(NUM_SEED)
    # random.seed(NUM_SEED)
    
    # 1) Initial state (7 joint angles + 7 joint velocities)
    x0 = np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi], dtype=float)
    # 2) Simulation parameters
    N_sim = 200  # number of simulation steps
      
    # 3) Initialize simulator and MPC controller
    mujoco_sim = MuJoCoSimulator()
    ocp, ocp_solver, integrator = create_ocp_solver(x0)

    # 4) Set initial control guess for the solver
    u_guess = generate_random_initial_guess()
    # If desired, one can manually specify a particular initial guess, e.g.:
    u_guess = np.array([0.0, 0.0, 0.0, 0, 0, 0.0, 20])
    ocp_solver.set(0, "x", x0)
    # for j in range(config.Horizon):
    #     ocp_solver.set(j, "u", u_guess)
   
    # 5) Run closed-loop simulation using MuJoCo physics
    start_time = time.time()
    t, simX, simU, simCost, success, pos, simX_mj = simulate_closed_loop_mujoco(ocp, ocp_solver, mujoco_sim, x0, N_sim=N_sim)
    end_time = time.time()
    print("first value of simX:", simX[0, :])  # initial state
    print("second value of simX:", simX[1, :])  
    print("second value of simU:", simU[0, :])
    # 6) Compare end-effector positions from MuJoCo vs. CasADi model
    print("\n=== End-effector position comparison ===")
    mujoco_ee, casadi_ee = mujoco_sim.compare_end_effector_pos()

    # 7) Output final state and timing
    if success:
        elapsed_time = end_time - start_time
        print("Final state:", simX[-1, :])
        print(f"Simulation with initial control guess {u_guess} took {elapsed_time:.4f} seconds, time/step: {elapsed_time/N_sim:.4f} seconds")
    else:
        print("Simulation failed to converge to the target within the given steps.")
    
    # 8) Visualization of results
    plot_trajectories(simX_mj[:, :7], simU, pos, target_position=config.target_position)
    # To view an animation of the end-effector trajectory, you may use:
    # anim = animate_trajectory(pos, target_position=config.target_position)
    # plt.show()  # or anim.save('trajectory_animation.gif') to save
    
if __name__ == "__main__":
    main()
