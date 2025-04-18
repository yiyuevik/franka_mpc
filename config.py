"""
config.py

集中管理 MPC 相关设置(Q, R, P, Horizon, Ts 等）
以及初始状态采样与随机初始猜测生成函数等。
"""
import  numpy as np

Horizon = 128          # 预测步数
Ts = 0.001             # 采样时间
Num_State = 17         #  状态维数（位置 + 速度）
Num_Q = 7         # 关节数目
Num_Velocity = 7         # 关节速度维数
Num_P = 3         # 末端执行器位置维数
Num_Input = 7         # 控制量维数（推力）
gravity = [0, 0, -9.81] # 重力加速度
root = "panda_link0"
tip = "panda_link8"
tau_max = 10000.0 # 最大关节力矩

# 状态和控制量的权重矩阵
Q = np.eye(7) * 10.0 

R = np.eye(7) * 0.5    

P = np.eye(7) * 10.0

def GenerateRandomInitialGuess(sim_round = 0, min_random=-6000.0, max_random=6000.0):
    """
    生成一个随机的 (u_ini_guess, x_ini_guess)
    其中 u_ini_guess 在 [min_random, max_random] 里均匀随机取,范围我不清楚，问！
    """
    u_ini_guess = np.random.uniform(min_random, max_random, 1)[0]
  
    x_ini_guess = np.array([ 0.85114759,0.22895749, -0.00465471, -1.69075889, -0.24523397,  0.20150836,
  0 , 0, 0, 0, 0, 0, 0, 0,    0.3, 0.3, 0.5])# 初始状态
      
    return u_ini_guess, x_ini_guess


