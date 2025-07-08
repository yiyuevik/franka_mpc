import numpy as np
import mujoco
import os
import time

import config
from utils.helpers import clear_solver_state, get_guess_from_solver_result

class MuJoCoSimulator:
    """
    MuJoCo simulator for the Franka Panda arm. Loads a MuJoCo model and provides step and reset functionality.
    """
    def __init__(self, xml_path=None):
        # Load the MuJoCo model (MJCF XML)
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "xml","panda_arm.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # Print diagnostic info about the model
        print("dof_damping =", self.model.dof_damping)        # expected to be all zeros (no damping)
        print("dof_frictionloss =", self.model.dof_frictionloss)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.n_joints = self.model.nv       # number of joints (DOF)
        self.n_actuators = self.model.nu    # number of actuators
        print("MuJoCo model loaded successfully:")
        print(f"  Number of joints: {self.n_joints}")
        print(f"  Number of actuators: {self.n_actuators}")
        print(f"  Base simulation timestep: {self.model.opt.timestep}")
        # Set simulation parameters for stepping
        self.dt = config.Ts
        self.substeps = max(1, int(self.dt / self.model.opt.timestep))
        print(f"  MPC timestep: {self.dt}")
        print(f"  Sub-steps per MPC step: {self.substeps}")
    
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
    
    def step(self, u):
        """
        Apply control input u (joint torques) and advance the simulation by one MPC timestep.
        Returns the next state (concatenated joint positions and velocities).
        """
        # Apply control torques
        self.data.qfrc_applied[:len(u)] = u
        mujoco.mj_forward(self.model, self.data)  # update internal state with applied forces
        # Advance the simulation for the duration of one MPC time step
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        # Retrieve resulting state
        q = self.data.qpos[:7].copy()
        qd = self.data.qvel[:7].copy()
        return np.concatenate([q, qd])
    
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
    
    def compare_end_effector_pos(self, T_fk_fun):
        """
        Compare MuJoCo's end-effector position with the position from the CasADi forward kinematics.
        Prints both positions and their difference.
        """
        mujoco_ee_pos = self.get_end_effector_pos()
        q_current = self.data.qpos[:7]
        casadi_ee_pos = self.fk_position_casadi(T_fk_fun, q_current)
        print(f"MuJoCo end-effector position: {mujoco_ee_pos}")
        print(f"CasADi end-effector position: {casadi_ee_pos}")
        print(f"Position difference: {np.linalg.norm(mujoco_ee_pos - casadi_ee_pos):.4f} m")
        return mujoco_ee_pos, casadi_ee_pos
    
    def fk_position_casadi(self, T_fk_fun, q_row):
        """
        Compute end-effector position using the CasADi forward kinematics function given joint positions q_row.
        """
        T = T_fk_fun(q_row)
        p = T[:3, 3]
        return np.array(p).reshape(3)

def simulate_closed_loop_mujoco(ocp, ocp_solver, mujoco_sim, x0, N_sim=50, nMaxGuess=1):
    """
    Simulate closed-loop control using the MuJoCo simulator for physics.
    Returns (t, simX, simU, simCost, success).
    """
    nx = ocp.model.x.size()[0]  # e.g., 14
    nu = ocp.model.u.size()[0]  # e.g., 7
    
    # Initialize storage for trajectory data
    simX = np.zeros((N_sim + 1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    simX[0, :] = x0
    
    # Reset the MuJoCo simulator to the initial state
    mujoco_sim.reset(q_init=x0[:7], qd_init=x0[7:])
    
    success = True
    for i in range(N_sim):
        retries = 0
        success = True
        while retries < nMaxGuess:
            try:
                # Solve MPC for current state
                u_opt = ocp_solver.solve_for_x0(x0_bar=simX[i, :])
                # Prepare warm-start for next iteration
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
                clear_solver_state(ocp_solver, config.Horizon)
                for j in range(config.Horizon):
                    ocp_solver.set(j, "u", u_guess[:, j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(config.Horizon, "x", x_guess[:, -1])
                
                # Apply control and simulate one step in MuJoCo
                simU[i, :] = u_opt
                simX[i+1, :] = mujoco_sim.step(u_opt)
                simCost[i, :] = ocp_solver.get_cost()
                break
            except Exception as e:
                success = False
                print(f"Error in MPC solve: {e}. Retrying with a new initial guess...")
                print(f"  Step {i}, retry {retries}")
                time.sleep(2)
            retries += 1
            if retries == nMaxGuess - 1:
                print("Trying alternative initial state guess (e.g., 2*pi offsets).")
                for j in range(0, config.Horizon, 20):
                    ocp_solver.set(j, "x", np.zeros(6))
                ocp_solver.set(0, "x", np.zeros(6))
        if not success:
            print("MPC solve failed after maximum retries.")
            break
    
    # Clear solver to free memory
    clear_solver_state(ocp_solver, config.Horizon)
    t = np.linspace(0, N_sim * config.Ts, N_sim + 1)
    return t, simX, simU, simCost, success
