"""
config.py

集中管理 MPC 相关设置（Q, R, P, Horizon, Ts 等）
以及初始状态采样与随机初始猜测生成函数等。
"""
import  numpy as np

Horizon = 128          # 预测步数
Ts = 0.01              # 采样时间
Num_State = 6         # 6个状态（包括两个摆杆的角度和角速度）
Num_Input = 1         # 控制量维数（推力）
Fmax = 6000

# 状态和控制量的权重矩阵
Q = np.diag([1,    # x
             1,    # xdot
             1000,    # theta1
             1,    # omega1
             1000,    # theta2
             1])   # omega2

R = 0.001             # 控制量的权重

P = np.diag([1,    # x
             1,    # xdot
             100,    # theta1
             1,    # omega1
             100,    # theta2
             1, ])   # omega2

def GenerateRandomInitialGuess(sim_round = 0, min_random=-6000.0, max_random=6000.0):
    """
    生成一个随机的 (u_ini_guess, x_ini_guess)
    其中 u_ini_guess 在 [min_random, max_random] 里均匀随机取,范围我不清楚，问！
    """
    u_ini_guess = np.random.uniform(min_random, max_random, 1)[0]
    mode = True
    if mode == False :
        x_ini_guess = np.array([0.0, 0.0, 2*np.pi, 0.0, 2*np.pi, 0.0])
        # if u_ini_guess >= 0:
        #     x_ini_guess =  np.zeros(6)
        #     x_ini_guess[2] = 2 * np.pi
        #     x_ini_guess[4] = 2 * np.pi
        # else:
        #     x_ini_guess = np.zeros(6)
    else:
        theta1_values = np.linspace(0, 2 * np.pi, 25)
        theta2_values = np.linspace(0, 2 * np.pi, 25)

        theta1_grid, theta2_grid = np.meshgrid(theta1_values, theta2_values)
        theta1_flat = theta1_grid.flatten()  
        theta2_flat = theta2_grid.flatten()
        # 根据 sim_round 选择对应的组合

        idx = (sim_round - 1) % (25 * 25)

        x_ini_guess = np.zeros(6)
        x_ini_guess[2] = theta1_flat[idx]  # theta1 对应数组的第3个元素（index 2）
        x_ini_guess[4] = theta2_flat[idx]  # theta2 对应数组的第5个元素（index 4）

    return u_ini_guess, x_ini_guess


def compute_cost(simX, simU):
    """
    计算闭环仿真的成本

    参数:
      simX: array, 形状为 (N_sim+1, n_state)
      simU: array, 形状为 (N_sim, n_control)
      Q: 状态权重矩阵
      R: 控制输入权重，如果 u 是标量，也可以直接传入标量
      P: 终端状态权重矩阵

    返回:
      cost: 标量，表示整个轨迹的总成本
    """
    # 仿真步数, N_sim = simX.shape[0] - 1
    N_sim = simX.shape[0] - 1
    cost = 0.0

    # 遍历每个阶段
    for k in range(N_sim):
        xk = simX[k, :].copy()                   # 第 k 步状态
        # 判断 u 是否为标量，如果 simU 每一项为 array，可以直接用向量运算
        uk = simU[k]                      # 第 k 步控制，可能是标量或向量
        xk[2] = np.sin(xk[2]/2)
        xk[4] = np.sin(xk[4]/2)
        cost_x = np.dot(xk, Q @ xk)         # x^T Q x
        cost_u = R * (uk ** 2)
        cost += cost_x + cost_u

    # 终端成本：最后状态
    xN = simX[-1, :]
    cost_terminal = np.dot(xN, P @ xN)
    cost += cost_terminal

    return cost
