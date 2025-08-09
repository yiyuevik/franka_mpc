import numpy as np
import random
import config
import os
import casadi as ca
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

def generate_random_initial_guess(u4_min=None, u4_max=None, u5_min = None, u5_max=None, u7_min=None, u7_max=None):
    """
    Generate a random initial guess for the control input (joint torques).
    By default, this randomizes a subset of joint torques (the 4th, 5th, and 7th joints) within the configured range, while others are set to zero.
    """
    if u4_min is None:
        u4_min = config.U4_MIN
    if u4_max is None:
        u4_max = config.U4_MAX
    if u5_min is None:
        u5_min = config.U5_MIN
    if u5_max is None:
        u5_max = config.U5_MAX
    if u7_min is None:
        u7_min = config.U7_MIN
    if u7_max is None:
        u7_max = config.U7_MAX

    u_guess = np.zeros(config.Num_Input)
    # Randomize specific joint indices 3, 4, 6 (0-based) corresponding to joints 4, 5, 7
    u_guess[3] = round(random.uniform(u4_min, u4_max), 2)
    u_guess[4] = round(random.uniform(u5_min, u5_max), 2)
    u_guess[6] = round(random.uniform(u7_min, u7_max), 2)
    return u_guess

def generate_grid_initial_guesses(u4_min, u4_max, step=5):
    
    u4_range = np.linspace(u4_min, u4_max, num=step)
    u5_range = np.linspace(config.U5_MIN, config.U5_MAX, num=step)
    u7_range = np.linspace(config.U7_MIN, config.U7_MAX, num=step)
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

def compute_end_effector_position_symbolic(q):
    """
    CasADi MX symbolic version of end-effector position.
    """
    T_fk_fun = _initialize_forward_kinematics()  # This returns CasADi function!
    T = T_fk_fun(q)
    return T[:3, 3]  # symbolic 3x1 MX

def log_SO3_vee(R):
    """
    CasADi-compatible SO(3) logarithm map.
    Inputs:
        R : 3×3 CasADi SX rotation matrix
    Returns:
        3×1 SX vector (axis-angle) = vee(log(R))
    """
    trace_R = ca.trace(R)
    cos_theta = (trace_R - 1) / 2
    # Clamp to avoid NaN from acos
    cos_theta = ca.fmin(ca.fmax(cos_theta, -1 + 1e-9), 1 - 1e-9)
    theta = ca.acos(cos_theta)

    # Anti-symmetric part
    omega_hat = 0.5 * (R - R.T)
    vee = ca.vertcat(
        omega_hat[2, 1],
        omega_hat[0, 2],
        omega_hat[1, 0]
    )

    # Small-angle safeguard: use first-order Taylor when θ ≈ 0
    eps = 1e-6
    A = ca.if_else(theta < eps,
                   1 + 0 * theta,
                   theta / (2 * ca.sin(theta)))
    return A * vee

def skew(v):
    return ca.vertcat(
        ca.horzcat( 0,     -v[2],  v[1]),
        ca.horzcat( v[2],   0,    -v[0]),
        ca.horzcat(-v[1],  v[0],  0)
    )

def SO3_target_from_log(phi):
    angle = ca.norm_2(phi)
    I = ca.SX.eye(3)
    eps = 1e-6
    K = skew(phi / (angle + 1e-12))
    R = I + ca.if_else(angle < eps,
                       I + K,               # 1st-order
                       I + (ca.sin(angle)/angle) * K +
                       ((1 - ca.cos(angle))/(angle**2)) * ca.mtimes(K, K))
    return R

def obstacle_constraint_expr(q, o_p, o_s):
    """ Compute the squared distance from the end-effector position to an obstacle.
    Args:
        q : 7x1 CasADi MX vector of joint angles
        o_p : 3x1 CasADi MX vector of obstacle position
        o_s : 3x1 CasADi MX vector of obstacle size (radius)
    Returns:
        CasADi MX scalar expression representing the squared distance.
    """
    ee_pos = compute_end_effector_position_symbolic(q)  # CasADi MX 3x1 vector
    scaled_diff = (ee_pos - o_p) * o_s         # element-wise multiply
    return ca.sumsqr(scaled_diff)              # return scalar expression