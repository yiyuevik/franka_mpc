import numpy as np
import random
import config
import os
_fk_function = None


def clear_solver_state(ocp_solver, N_horizon):
    """
    Clear the solver's stored trajectories by setting all predicted states and inputs to zero.
    """
    for i in range(N_horizon):
        ocp_solver.set(i, "x", np.zeros_like(ocp_solver.get(i, "x")))
        ocp_solver.set(i, "u", np.zeros_like(ocp_solver.get(i, "u")))
    # Also clear the terminal state at the end of the horizon
    ocp_solver.set(N_horizon, "x", np.zeros_like(ocp_solver.get(N_horizon, "x")))

def get_guess_from_solver_result(ocp_solver, N_horizon):
    """
    Extract the current solver solution (states and inputs over the horizon) to use as an initial guess for the next solve.
    Returns (u_guess, x_guess) arrays for inputs and states.
    """
    u_guess = np.zeros((config.Num_Input, N_horizon))
    x_guess = np.zeros((config.Num_State, N_horizon + 1))
    for i in range(N_horizon - 1):
        u_guess[:, i] = ocp_solver.get(i + 1, "u")
        x_guess[:, i] = ocp_solver.get(i + 1, "x")
    # For the last stage, include the final control and state
    u_guess[:, N_horizon - 1] = ocp_solver.get(N_horizon - 1, "u")
    x_guess[:, N_horizon - 1] = ocp_solver.get(N_horizon, "x")
    x_guess[:, N_horizon] = ocp_solver.get(N_horizon, "x")
    return u_guess, x_guess

def generate_random_initial_guess(min_random=None, max_random=None):
    """
    Generate a random initial guess for the control input (joint torques).
    By default, this randomizes a subset of joint torques (the 4th, 5th, and 7th joints) within the configured range, while others are set to zero.
    """
    if min_random is None:
        min_random = config.initial_guess_min
    if max_random is None:
        max_random = config.initial_guess_max
    u_guess = np.zeros(config.Num_Input)
    # Randomize specific joint indices 3, 4, 6 (0-based) corresponding to joints 4, 5, 7
    random_indices = [3, 4, 6]
    for j in random_indices:
        u_guess[j] = round(random.uniform(min_random, max_random), 2)
    return u_guess

def generate_grid_initial_guesses(u4_min, u4_max, step=2.5):
    u4_range = np.arange(u4_min, u4_max + step, step)
    u5_range = np.arange(config.U5_MIN, config.U5_MAX + step, step)
    u7_range = np.arange(config.U7_MIN, config.U7_MAX + step, step)
    u4_grid, u5_grid, u7_grid = np.meshgrid(u4_range, u5_range, u7_range)
    u4_flat = u4_grid.flatten()
    u5_flat = u5_grid.flatten()
    u7_flat = u7_grid.flatten()
    all_guesses = []
    for i in range(len(u4_flat)):
        u_guess = np.array([0, 0, 0, u4_flat[i], u5_flat[i], 0, u7_flat[i]])
        all_guesses.append(u_guess)
    return np.array(all_guesses)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _initialize_forward_kinematics():
    global _fk_function
    if _fk_function is None:
        import urdf2casadi.urdfparser as u2c
        import os
        franka = u2c.URDFparser()
        path_to_franka = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'urdf/panda_arm.urdf')
        franka.from_file(path_to_franka)
        fk_dict = franka.get_forward_kinematics(config.root, config.tip)
        _fk_function = fk_dict["T_fk"]
    return _fk_function

def compute_end_effector_position(q):
    T_fk_fun = _initialize_forward_kinematics()
    T = T_fk_fun(q)
    position = np.array(T[:3, 3]).flatten()
    return position