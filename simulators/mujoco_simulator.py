import numpy as np
import mujoco
import os
import time

import configs
from utils.helpers import clear_solver_state, get_guess_from_solver_result, compute_end_effector_position, generate_random_initial_guess

class MuJoCoSimulator:
    """
    MuJoCo simulator for the Franka Panda arm. Loads a MuJoCo model and provides step and reset functionality.
    """
    def __init__(self, xml_path=None):
        # Load the MuJoCo model (MJCF XML)
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "xml","panda_arm_modified.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # Print diagnostic info about the model
        print("dof_damping =", self.model.dof_damping)        # expected to be all zeros (no damping)
        print("dof_frictionloss =", self.model.dof_frictionloss)
        self.n_joints = self.model.nv       # number of joints (DOF)
        self.n_actuators = self.model.nu    # number of actuators
        print("MuJoCo model loaded successfully:")
        print(f"  Number of joints: {self.n_joints}")
        print(f"  Number of actuators: {self.n_actuators}")
        print(f"  Base simulation timestep: {self.model.opt.timestep}")
        # Set simulation parameters for stepping
        self.dt = configs.Ts
        self.substeps = max(1, int(self.dt / self.model.opt.timestep))
        print(f"  MPC timestep: {self.dt}")
        print(f"  Sub-steps per MPC step: {self.substeps}")
        self.joint_history = []  # 存储所有步骤的关节位置数据
        self.command_history = []   # 存储所有步骤的控制命令
        self.step_count = 0 
    
    def reset(self, q_init=None, qd_init=None):
        """
        Reset the simulation to an initial state. Optionally sets initial joint positions and velocities.
        """
        mujoco.mj_resetData(self.model, self.data)
        if q_init is not None:
            self.data.qpos[:len(q_init)] = q_init
        if qd_init is not None:
            self.data.qvel[:len(qd_init)] = qd_init
        mujoco.mj_forward(self.model, self.data)
        self.velocity_history = []
        self.command_history = []
        self.step_count = 0
    
    def step(self, u):
        """
        Apply control input u (joint velocity) and advance the simulation by one MPC timestep.
        Returns the next state (concatenated joint positions and velocities).
        """
        # Apply control torques
        joint_data = np.zeros((self.substeps, 7))
        current_q = self.data.qpos[:7].copy()  # current joint positions
        target_q = current_q + u * self.dt
        # Advance the simulation for the duration of one MPC time step
        for i in range(self.substeps):
            alpha = (i + 1) / self.substeps
            q_cmd = (1 - alpha) * current_q + alpha * target_q
            self.data.ctrl[:len(u)] = q_cmd
            mujoco.mj_step(self.model, self.data)
            joint_data[i, :] = self.data.qpos[:7].copy()
            # q  = self.data.qpos[:7].copy()
            # qd = self.data.qvel[:7].copy()
            # tau_act    = self.data.qfrc_actuator[:7].copy()   # 执行器力矩（已包含 -kv*qd）
            # tau_bias   = self.data.qfrc_bias[:7].copy()       # 重力+科氏+离心
            # tau_pass   = self.data.qfrc_passive[:7].copy()    # 被动项(关节阻尼/摩擦)
            # tau_net = tau_act + tau_pass - tau_bias

            # J = 0  # joint1
            # print(f"J1 q={q[J]:+.4f}, qd={qd[J]:+.4f}, "
            #     f"act={tau_act[J]:+.2f}, pass={tau_pass[J]:+.2f}, "
            #     f"bias={tau_bias[J]:+.2f}, net≈{tau_net[J]:+.2f}")
        self.joint_history.append(joint_data)  # 保存子步骤位置数据
        self.command_history.append(target_q.copy())        # 保存控制命令
        # self.plot_velocity_comparison(target_q, joint_data)
        self.step_count += 1

        # Retrieve resulting state
        q = self.data.qpos[:7].copy()
        # qd = self.data.qvel[:7].copy()
        return q
    
    def get_state(self):
        """
        Get the current state [q (7), qdot (7)] from the simulator.
        """
        q = self.data.qpos[:7].copy()
        qd = self.data.qvel[:7].copy()
        return np.concatenate([q, qd])
    
    def get_end_effector_pos(self):
        """
        Get the current end-effector position (x, y, z) from the simulator state.
        """
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_link8")
            return self.data.xpos[body_id].copy()
        except mujoco.MjError:
            # Fallback if named body not found
            print("Warning: manually computing end-effector position.")
            return self.data.xpos[-1].copy()
    
    def compare_end_effector_pos(self):
        """
        Compare MuJoCo's end-effector position with the position from the CasADi forward kinematics.
        Prints both positions and their difference.
        """
        mujoco_ee_pos = self.get_end_effector_pos()
        q_current = self.data.qpos[:7]
        casadi_ee_pos = compute_end_effector_position(q_current)
        print(f"MuJoCo end-effector position: {mujoco_ee_pos}")
        print(f"CasADi end-effector position: {casadi_ee_pos}")
        print(f"Position difference: {np.linalg.norm(mujoco_ee_pos - casadi_ee_pos):.4f} m")
        return mujoco_ee_pos, casadi_ee_pos
    
    def plot_velocity_comparison(self, u_command, velocity_data):
        """
        绘制控制输入与实际速度的对比图
        """
        import matplotlib.pyplot as plt
        
        time_steps = np.arange(self.substeps)
        
        plt.figure(figsize=(12, 8))
        
        for j in range(7):
            plt.subplot(3, 3, j+1)
            
            # 
            plt.plot(time_steps, velocity_data[:, j], 'b-', linewidth=2, label=f'Actual qd{j+1}')
            
            
            plt.axhline(u_command[j], linestyle='--', color='r', linewidth=2, label=f'Command u{j+1}')
            
            plt.xlabel("Sub-step")
            plt.ylabel(f"Joint {j+1} position (rad)")
            plt.title(f"Joint {j+1} Position Tracking")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("velocity_tracking_comparison.png", dpi=150, bbox_inches='tight')
        # 等待 4s
        time.sleep(4)
        plt.close() 

    def plot_complete_velocity_tracking(self, save_path="complete_velocity_tracking.png"):
        import matplotlib.pyplot as plt
        n_steps = len(self.joint_history)
        n_substeps = self.substeps
        
        # 创建时间轴
        dt_substep = self.model.opt.timestep
        total_time_points = n_steps * n_substeps
        time_axis = np.linspace(0, n_steps * self.dt, total_time_points)
        
        all_velocities = np.concatenate(self.joint_history, axis=0)  # (total_time_points, 7)
        all_commands = np.array(self.command_history)  # (n_steps, 7)
        
        expanded_commands = np.repeat(all_commands, n_substeps, axis=0)
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()
        
        for j in range(7):
            ax = axes[j]
            
            # 实际速度
            ax.plot(time_axis, all_velocities[:, j], 'b-', linewidth=1.5, 
                   label=f'Actual qd{j+1}', alpha=0.8)
            
            # u（分段常数）
            ax.plot(time_axis, expanded_commands[:, j], 'r--', linewidth=2, 
                   label=f'Command u{j+1}', alpha=0.9)
            
            for step in range(1, n_steps):
                ax.axvline(step * self.dt, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"Joint {j+1} position (rad)")
            ax.set_title(f"Joint {j+1} Position Tracking (Complete)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
def simulate_closed_loop_mujoco(ocp, ocp_solver, mujoco_sim, x0, u_guess, N_sim=50, nMaxGuess=3):
    """
    Simulate closed-loop control using the MuJoCo simulator for physics.
    Returns (t, simX, simU, simCost, success).
    """
    nx = ocp.model.x.size()[0]  
    nu = ocp.model.u.size()[0]  

    # Initialize storage for trajectory data
    simX = np.zeros((N_sim + 1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    pos = np.zeros((N_sim + 1, 3)) 
    simX[0, :] = x0
    pos_mpc = np.zeros((N_sim + 1, 3))
    # Reset the MuJoCo simulator to the initial state
    mujoco_sim.reset(q_init=x0, qd_init=np.zeros(7))
    pos[0, :] = mujoco_sim.get_end_effector_pos()
    pos_mpc[0, :] = compute_end_effector_position(x0)

   
    success = True
    # =================================== first step ===================================
    ocp_solver.reset()
    ocp_solver.set(0, "x", x0)
    for j in range(configs.Horizon):
        ocp_solver.set(j, "x", x0)
        ocp_solver.set(j, "u", u_guess)
    ocp_solver.set(configs.Horizon, "x", x0)
    retries = 0
    while retries < nMaxGuess: 
        try:
            # Solve MPC for the initial state
            u_opt = ocp_solver.solve_for_x0(x0_bar=x0)
            # Prepare warm-start for next iteration
            u_guess, x_guess = get_guess_from_solver_result(ocp_solver, configs.Horizon)
            clear_solver_state(ocp_solver, configs.Horizon)
            for j in range(configs.Horizon):
                ocp_solver.set(j, "u", u_guess[:, j])
                ocp_solver.set(j, "x", x_guess[:, j])
            ocp_solver.set(configs.Horizon, "x", x_guess[:, -1])
            # x_mpc = ocp_solver.get(0, "x")
            # print("x_mpc:", x_mpc)
            # Apply control and simulate one step in MuJoCo
            simU[0, :] = u_opt
            simX[1, :] = mujoco_sim.step(u_opt)
            # pos_mpc[1, :] = compute_end_effector_position(x_mpc)
            # print("x_mujoco:", simX[1, :])
            simCost[0, :] = ocp_solver.get_cost()
            pos[1, :] = mujoco_sim.get_end_effector_pos()
            break
        except Exception as e:
            ocp_solver.reset()
            ocp_solver.set(0, "x", x0)
            for j in range(configs.Horizon):
                ocp_solver.set(j, "u", u_guess)
            retries += 1
            if retries == nMaxGuess - 1:
                success = False

    # ======================== generate data for control step loop =============================
    if success:
        
        for i in range(1, N_sim):
            retries = 0
            success = True
            while retries < nMaxGuess:
                try:
                    # Solve MPC for current state
                    u_opt = ocp_solver.solve_for_x0(x0_bar=simX[i, :])
                    # Prepare warm-start for next iteration
                    u_guess, x_guess = get_guess_from_solver_result(ocp_solver, configs.Horizon)
                    ocp_solver.reset()
                    # u_guess = generate_random_initial_guess()
                    for j in range(configs.Horizon):
                        ocp_solver.set(j, "u", u_guess[:, j])
                        ocp_solver.set(j, "x", x_guess[:, j])
                    ocp_solver.set(configs.Horizon, "x", x_guess[:, -1])
                    # ocp_solver.set(config.Horizon//2, "u", u_guess)
                    
                    # Apply control and simulate one step in MuJoCo
                    simU[i, :] = u_opt
                    simX[i+1, :] = mujoco_sim.step(u_opt)
                    simCost[i, :] = ocp_solver.get_cost()
                    pos[i+1, :] = mujoco_sim.get_end_effector_pos()

                    # x_mpc = ocp_solver.get(0, "x")
                    # print("x_mpc:", x_mpc)
                    # print("x_mujoco:", simX[i+1, :])
                    # pos_mpc[i+1, :] = compute_end_effector_position(x_mpc)
                    break
                except Exception as e:
                    ocp_solver.reset()
                    ocp_solver.set(0, "x", simX[i, :])
                    u_guess = generate_random_initial_guess()
                    for j in range(configs.Horizon):
                        ocp_solver.set(j, "u", u_guess)
                    print(f"Error in MPC solve: {e}. Retrying with a new initial guess u_guess: {u_guess}...")
                    print(f"  Step {i}, retry {retries}")
                retries += 1
                if retries == nMaxGuess - 1:
                    success = False
                    print(f"  Step {i}, failed after {nMaxGuess} retries.")
            if not success:
                print("MPC solve failed after maximum retries.")
                break
   
    
    # mujoco_sim.plot_complete_velocity_tracking("complete_velocity_tracking.png")
    # Clear solver to free memory
    clear_solver_state(ocp_solver, configs.Horizon)
    t = np.linspace(0, N_sim * configs.Ts, N_sim + 1)
    return t, simX, simU, simCost, success, pos
