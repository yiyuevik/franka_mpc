import numpy as np
import random
import config

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
