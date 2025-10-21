import numpy as np
import random
import configs
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
    u_guess = np.zeros((configs.Num_Input, N_horizon))
    x_guess = np.zeros((configs.Num_State, N_horizon + 1))
    for i in range(N_horizon - 1):
        u_guess[:, i] = ocp_solver.get(i + 1, "u")
        x_guess[:, i] = ocp_solver.get(i + 1, "x")
    # For the last stage, include the final control and state
    u_guess[:, N_horizon - 1] = ocp_solver.get(N_horizon - 1, "u")
    x_guess[:, N_horizon - 1] = ocp_solver.get(N_horizon, "x")
    x_guess[:, N_horizon] = ocp_solver.get(N_horizon, "x")
    return u_guess, x_guess

def generate_random_initial_guess(u3_min=None, u3_max=None, u5_min = None, u5_max=None, u6_min=None, u6_max=None):
    """
    Generate a random initial guess for the control input (joint torques).
    By default, this randomizes a subset of joint torques (the 4th, 5th, and 7th joints) within the configured range, while others are set to zero.
    """
    if u3_min is None:
        u3_min = configs.U3_MIN
    if u3_max is None:
        u3_max = configs.U3_MAX
    if u5_min is None:
        u5_min = configs.U5_MIN
    if u5_max is None:
        u5_max = configs.U5_MAX
    if u6_min is None:
        u6_min = configs.U6_MIN
    if u6_max is None:
        u6_max = configs.U6_MAX

    u_guess = np.zeros(configs.Num_Input)
    # Randomize specific joint indices 2, 4, 5 (0-based) corresponding to joints 3, 5, 6
    u_guess[2] = round(random.uniform(u3_min, u3_max), 2)
    u_guess[4] = round(random.uniform(u5_min, u5_max), 2)
    u_guess[5] = round(random.uniform(u6_min, u6_max), 2)
    return u_guess

def generate_grid_initial_guesses(u3_min, u3_max, step=5):

    u3_range = np.linspace(u3_min, u3_max, num=step)
    u5_range = np.linspace(configs.U5_MIN, configs.U5_MAX, num=step)
    u6_range = np.linspace(configs.U6_MIN, configs.U6_MAX, num=step)
    u3_grid, u5_grid, u6_grid = np.meshgrid(u3_range, u5_range, u6_range)
    u3_flat = u3_grid.flatten()
    u5_flat = u5_grid.flatten()
    u6_flat = u6_grid.flatten()
    all_guesses = []
    for i in range(len(u3_flat)):
        u_guess = np.array([0, 0, u3_flat[i], 0,  u5_flat[i], u6_flat[i], 0])
        # u_guess = np.random.uniform(-3, 3, size=7)
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
        fk_dict = franka.get_forward_kinematics(configs.root, configs.tip)
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

def sample_states_around(x0, num_samples):
    samples = []
    for _ in range(num_samples):
        dx = np.random.normal(configs.Noise_Mean, configs.Noise_Std, size= configs.Num_State)
        x_sample = x0 + dx
        samples.append(x_sample)
    return samples

def get_traj(ocp_solver, N, nx, nu):
    X = np.zeros((N+1, nx))
    U = np.zeros((N, nu))
    Pos = np.zeros((N + 1, 3))  

    for i in range(N):
        X[i, :] = ocp_solver.get(i, "x")
        U[i, :] = ocp_solver.get(i, "u")
        Pos[i, :] = compute_end_effector_position(X[i, :])
    X[N, :] = ocp_solver.get(N, "x")
    Pos[N, :] = compute_end_effector_position(X[N, :])
    return X, U, Pos

def log_SO3_vee(R):
    """
    CasADi-compatible SO(3) logarithm map.
    Inputs:
        R : 3×3 CasADi SX rotation matrix
    Returns:
        3×1 SX vector (axis-angle) = vee(log(R))
    """
    R = ca.SX(R)

    trace_R = R[0,0] + R[1,1] + R[2,2]
    eps_small = 1e-4
    delta_pi  = 1e-6

    omega_hat = 0.5 * (R - R.T)
    vee = ca.vertcat(omega_hat[2,1], omega_hat[0,2], omega_hat[1,0])


    cos_theta = (trace_R - 1) / 2
    cos_theta = ca.fmin(ca.fmax(cos_theta, -1), 1)
    theta = ca.acos(cos_theta)

    A_small  = 1 + (theta*theta)/6
    A_normal = theta / (ca.sin(theta) + 1e-12)
    A = ca.if_else(theta < eps_small, A_small, A_normal)

    rotvec_normal = A * vee

    r00, r11, r22 = R[0,0], R[1,1], R[2,2]
    cond01 = r00 >= r11
    m01   = ca.if_else(cond01, r00, r11)
    idx01 = ca.if_else(cond01, 0,   1)
    cond  = m01 >= r22
    idx   = ca.if_else(cond, idx01, 2)

    def axis_from_R(i):
        if i == 0:
            x = ca.sqrt(ca.fmax(1 + r00 - r11 - r22, 0))
            v = ca.vertcat(
                x,
                (R[0,1] + R[1,0]) / (x + 1e-12),
                (R[0,2] + R[2,0]) / (x + 1e-12)
            )
        elif i == 1:
            y = ca.sqrt(ca.fmax(1 + r11 - r00 - r22, 0))
            v = ca.vertcat(
                (R[0,1] + R[1,0]) / (y + 1e-12),
                y,
                (R[1,2] + R[2,1]) / (y + 1e-12)
            )
        else:
            z = ca.sqrt(ca.fmax(1 + r22 - r00 - r11, 0))
            v = ca.vertcat(
                (R[0,2] + R[2,0]) / (z + 1e-12),
                (R[1,2] + R[2,1]) / (z + 1e-12),
                z
            )
        n = ca.norm_2(v)
        return ca.if_else(n > 0, v / n, ca.vertcat(1,0,0))

    axis0 = axis_from_R(0)
    axis1 = axis_from_R(1)
    axis2 = axis_from_R(2)
    axis_pi = ca.if_else(idx == 0, axis0, ca.if_else(idx == 1, axis1, axis2))

    is_pi = trace_R < -1 + delta_pi
    rotvec = ca.if_else(is_pi, ca.pi * axis_pi, rotvec_normal)
    return rotvec

def SO3_target_from_log(phi):
    angle = ca.norm_2(phi)
    epsilon = 1e-12
    condition = angle < epsilon
    k = phi / ca.if_else(condition, 1.0, angle)

    K = ca.vertcat(
        ca.horzcat(0, -k[2], k[1]),
        ca.horzcat(k[2], 0, -k[0]),
        ca.horzcat(-k[1], k[0], 0)
    )
    sin_angle = ca.sin(angle)
    cos_angle = ca.cos(angle)
    Rm = ca.DM.eye(3) + sin_angle * K + (1 - cos_angle) * ca.mtimes(K, K)
    return Rm


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

def random_rotvec():
    """
    Generate a random rotation vector (axis-angle representation) in 3D space.
    Returns:
        3x1 numpy array representing the rotation vector.
    """
   
    # 1. Randomly generate rotation axis u (uniform direction)
    v = np.random.randn(3)       # Gaussian distribution
    u = v / np.linalg.norm(v)

    # 2. Uniformly sample angle from [0, π]
    theta = np.random.uniform(0, np.pi)

    # 3. Construct rotvec
    rotvec = theta * u
    return rotvec