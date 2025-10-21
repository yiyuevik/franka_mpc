# mpc_ipopt_franka.py
import os
from time import time
import numpy as np
import casadi as ca
import urdf2casadi.urdfparser as u2c

import configs
from utils.helpers import log_SO3_vee, SO3_target_from_log, obstacle_constraint_expr, compute_end_effector_position, _initialize_forward_kinematics
from utils.plotting import plot_trajectories


# ------------------------------
# 2) 构建基于多重射击的 NLP
# ------------------------------
def build_ipopt_mpc(T_fk_fun, N=None, Ts=None, use_u_stage_cost=True, use_obstacle=True):
    """
    NLP 决策变量:
        X = [x_0, x_1, ..., x_N],  x_k ∈ R^7
        U = [u_0, ..., u_{N-1}],   u_k ∈ R^7
    参数 p:
        p = [x0(7), y_ref(6)]  其中 y_ref = [p_ref(3), rotvec_ref(3)]
    目标:
        J = sum_{k} u_k^T R u_k  +  [pos_N - p_ref, rot_err_N]^T W_e [ ... ]
    约束:
        动力学: x_{k+1} - x_k - Ts*u_k = 0
        盒约束: q_min <= x_k <= q_max, u_min <= u_k <= u_max
        初值: x_0 == x0
        (可选)障碍: h(x_k) >= 1
    """
    nq = configs.Num_State
    nu = configs.Num_Input

    if N  is None: N  = configs.Horizon
    if Ts is None: Ts = configs.Ts

    # 决策变量堆叠
    X  = ca.SX.sym('X',  nq, N+1)   # 每列一个状态
    U  = ca.SX.sym('U',  nu, N)     # 每列一个控制
    z  = ca.vertcat(ca.reshape(X, nq*(N+1), 1),
                    ca.reshape(U, nu*N, 1))

    # 参数：x0(7) + 目标(6)
    p  = ca.SX.sym('p', nq + 6)

    # 代价权重
    Qpos = np.array(configs.Q_pos)        # 3x3
    Prol = np.array(configs.P_rot)        # 3x3（终端）
    Ppos = np.array(configs.P_pos)        # 3x3（终端）
    Rw   = np.array(configs.R)            # 7x7（阶段对 u 的正则）

    # 取出参数
    x0_param  = p[0:nq]
    pref_pos  = p[nq:nq+3]
    pref_rv   = p[nq+3:nq+6]  # 期望旋转的 log（轴角）

    # 目标旋转矩阵（由 rotvec 指定）
    R_ref = SO3_target_from_log(pref_rv)  # 3x3 SX

    J = 0
    g_list = []

    # 初值约束: x_0 = x0_param
    g_list.append(X[:, 0] - x0_param)

    # 路径盒约束（放到 lbx/ubx，不进 g）
    q_min = np.array([
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])
    q_max = np.array([
         2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973
    ])
    u_min = np.array([-2.618, -2.618, -2.618, -2.618, -3.142, -3.142, -3.142])
    u_max = np.array([ 2.618,  2.618,  2.618,  2.618,  3.142,  3.142,  3.142])

    # 动力学 + 路径代价 + (可选)障碍约束
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        xk1= X[:, k+1]

        # 动力学等式约束: x_{k+1} - x_k - Ts*u_k = 0
        g_dyn = xk1 - (xk + Ts*uk)
        g_list.append(g_dyn)

        # 路径代价（可选）
        if use_u_stage_cost:
            J += ca.mtimes([uk.T, Rw, uk])

        # 障碍约束: h(x) >= 1  ->  g_obs = h(x) - 1 >= 0
        if use_obstacle and getattr(configs, "Obstacle_Avoidance", False):
            o_p = ca.DM(configs.Obstacle_Position)  # 3
            o_s = ca.DM(configs.Obstacle_Scale)     # 3
            h_expr = obstacle_constraint_expr(xk, o_p, o_s)     # 标量或(1,)
            g_list.append(h_expr - 1.0)

    # 终端代价：位置 + 旋转
    xN = X[:, N]
    T_N = T_fk_fun(xN)             # 4x4
    posN = T_N[:3, 3]              # 3x1
    R_N  = T_N[:3, :3]             # 3x3

    # 旋转误差：rot_err = vee( log( R_ref^T * R_N ) )
    rot_err = log_SO3_vee(ca.mtimes(R_ref.T, R_N))   # 3x1

    pos_err = posN - pref_pos     # 3x1

    J_terminal = ca.mtimes([pos_err.T, Ppos, pos_err]) + ca.mtimes([rot_err.T, Prol, rot_err])
    J += J_terminal

    # NLP 组装
    g = ca.vertcat(*g_list)

    nlp = {'x': z, 'f': J, 'g': g, 'p': p}

    # IPOPT 选项
    # 注意：LM(Levenberg–Marquardt) 不会“自动”启用；这里给两套常用选项
    ipopt_opts_exact = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 200,
        "ipopt.tol": 1e-4,
        "ipopt.linear_solver": "mumps",     # 若无 mumps，可改 ma27/ma57/ma86 等
        # 用精确 Jacobian/Hessian（CasADi 会给 IPOPT）
        # 这对带旋转的非线性问题往往更稳
    }
    ipopt_opts_lbfgs = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 200,
        "ipopt.tol": 1e-4,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",  # L-BFGS，内存小/快，但可能需要更多迭代
    }

    solver_exact = ca.nlpsol('solver_exact', 'ipopt', nlp, ipopt_opts_exact)
    solver_lbfgs = ca.nlpsol('solver_lbfgs', 'ipopt', nlp, ipopt_opts_lbfgs)

    # 变量上下界（盒约束）
    nx = nq*(N+1)
    nu = nu = configs.Num_Input*N

    # 对 X的每一个时刻加 q 的盒约束
    lbx = []
    ubx = []
    for k in range(N+1):
        lbx.extend(q_min.tolist())
        ubx.extend(q_max.tolist())
    # 对 U的每一个时刻加 u 的盒约束
    for k in range(N):
        lbx.extend(u_min.tolist())
        ubx.extend(u_max.tolist())
    lbx = np.array(lbx)
    ubx = np.array(ubx)

    # 约束 g 的上下界：等式 0；障碍 >=0
    # g 的结构： [ (x0-x0_param),  以及每步: dyn等式(7) , (可选)障碍(1) ]
    n_eq = nq*(1 + N)          # 初值等式 7 + 每步动力学等式 7
    n_ineq = 0
    if use_obstacle and getattr(configs, "Obstacle_Avoidance", False):
        n_ineq = N             # 每步 1 个障碍不等式

    lbg = np.zeros(n_eq + n_ineq)
    ubg = np.zeros(n_eq)       # 等式上界 0
    if n_ineq > 0:
        # 不等式: g_obs >= 0  ->  lbg=0, ubg=+inf
        ubg = np.concatenate([ubg, np.full(n_ineq, np.inf)])

    # 打包一些形状工具
    def pack_z(X_val, U_val):
        return np.concatenate([X_val.reshape(-1), U_val.reshape(-1)])

    def unpack_z(z_val):
        X_val = z_val[:nq*(N+1)].reshape(nq, N+1)
        U_val = z_val[nq*(N+1):].reshape(nu//N, N)
        return X_val, U_val

    tools = dict(pack_z=pack_z, unpack_z=unpack_z)

    return solver_exact, solver_lbfgs, (lbx, ubx, lbg, ubg), tools

# ------------------------------
# 3) 一次求解 + 滚动优化接口
# ------------------------------
def solve_one_shot(solver, bounds, tools, x0, y_ref, z0=None):
    """
    solver: IPOPT 求解器
    bounds: (lbx, ubx, lbg, ubg)
    tools:  pack/unpack 函数
    x0:     (7,)
    y_ref:  (6,) = [p_ref(3), rotvec_ref(3)]
    z0:     初值 (可选)，形如 pack_z(X_guess, U_guess)
    """
    lbx, ubx, lbg, ubg = bounds

    # 组装参数 p = [x0, y_ref]
    p_val = np.concatenate([x0.reshape(-1), y_ref.reshape(-1)])

    # 没有初值的话，给个简单的
    if z0 is None:
        N  = configs.Horizon
        nq = configs.Num_State
        nu = configs.Num_Input
        X_guess = np.tile(x0.reshape(-1,1), (1, N+1))
        U_guess = np.zeros((nu, N))
        z0 = tools['pack_z'](X_guess, U_guess)

    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=z0, p=p_val)
    z_opt = np.array(sol['x']).reshape(-1)
    X_opt, U_opt = tools['unpack_z'](z_opt)
    pos = np.zeros((X_opt.shape[1], 3))
    for i, X in enumerate(X_opt.T):
        pos[i, :] = compute_end_effector_position(X)
    return X_opt, U_opt, z_opt, pos

def shift_warm_start(z_opt, tools):
    """
    把上次最优解平移一个步长，用作下一次的 warm-start
    """
    X_opt, U_opt = tools['unpack_z'](z_opt)
    X_ws = np.hstack([X_opt[:,1:], X_opt[:,[-1]]])    # 丢弃最前，复制末端
    U_ws = np.hstack([U_opt[:,1:], U_opt[:,[-1]]])    # 同理
    return tools['pack_z'](X_ws, U_ws)

# ------------------------------
# 4) 简单示例
# ------------------------------
def main():
    # 目标：终端位置 + 终端姿态（以 rotvec 给出）
    # 你也可以把下面这 6 个值替换为你的目标（比如从 config 或外部传入）
    y_ref = np.array([0.5068906, 0.2, 0.5902821, 2.38327818, 0.26047109, -1.37449648], dtype=float)
    x0    = np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi], dtype=float)

    T_fk_fun = _initialize_forward_kinematics()
    solver_exact, solver_lbfgs, bounds, tools = build_ipopt_mpc(
        T_fk_fun=T_fk_fun,
        N=configs.Horizon,
        Ts=configs.Ts,
        use_u_stage_cost=False,
        use_obstacle=getattr(configs, "Obstacle_Avoidance", False)
    )

    # 一次性解（open-loop）
    time_start = time()
    X_opt, U_opt, z_opt, pos = solve_one_shot(solver_exact, bounds, tools, x0, y_ref, z0=None)
    print(f"Solved in {time()-time_start:.4f} seconds")
    print("u_0* =", U_opt[:,0])
    # plot_trajectories(X_opt.T, U_opt.T, pos, target_position=config.target_position)

    
    # 如果要做闭环滚动：
    # for t in range(N_sim):
    #     apply u_0* to (真实/仿真)系统，得到新状态 x0_new
    #     z0 = shift_warm_start(z_opt, tools)
    #     X_opt, U_opt, z_opt = solve_one_shot(solver_exact, bounds, tools, x0_new, y_ref, z0=z0)

if __name__ == "__main__":
    main()