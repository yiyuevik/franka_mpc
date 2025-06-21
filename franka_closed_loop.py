"""
我没有写main.py
此即为主入口脚本：读取/设置模型参数(在 config.py)，构造并求解 OCP, 然后进行闭环仿真 + 可视化。
运行方式: python cartpole_closed_loop.py
"""

import config
from franka_ocp import create_ocp_solver, simulate_closed_loop  
from franka_utils import plot_cartpole_trajectories, animate_cartpole
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import random

def plot_and_save_trajectory(simX, simU, group_index, N_sim):
    """
    绘制末端执行器位置、关节力矩和关节位置，并将图像保存到文件。

    参数:
    simX (np.ndarray): 状态轨迹 (N_sim+1, Num_State)
    simU (np.ndarray): 控制输入轨迹 (N_sim, Num_Input)
    group_index (int): 当前组的索引，用于文件名
    N_sim (int): 仿真步数
    """
    pos = simX[:, 14:17]
    joint = simX[:, :7]
    # Ensure u has the correct dimensions for plotting if it comes from simU directly
    # If simU is (N_sim, Num_Input), and we want to plot it against time steps 0 to N_sim-1
    u = simU 

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot End-Effector Position
    axs[0].plot(np.arange(N_sim + 1), pos) # Assuming simX has N_sim+1 points
    axs[0].set_ylabel("End-Effector Position (m)")
    axs[0].legend(["x", "y", "z"])
    axs[0].grid(True)

    # Plot Joint Torque
    axs[1].plot(np.arange(N_sim), u) # Assuming simU has N_sim points
    axs[1].set_ylabel("Joint Torque (Nm)")
    axs[1].legend([f"τ{i+1}" for i in range(u.shape[1])]) # u.shape[1] should be Num_Input
    axs[1].grid(True)

    # Plot Joint Position
    axs[2].plot(np.arange(N_sim + 1), joint) # Assuming simX has N_sim+1 points
    axs[2].set_ylabel("Joint Position (rad)")
    axs[2].set_xlabel("Time step")
    axs[2].legend([f"q{i+1}" for i in range(joint.shape[1])]) # joint.shape[1] should be 7
    axs[2].grid(True)

    plt.tight_layout()
    filename = f"trajectory_group_{group_index}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close(fig) # Close the figure to free memory

def main():
    
    # 1) 生成初始状态样本
    x0 = np.array([ 0,0,0,0,0,0,0 , 0, 0, 0, 0, 0, 0,0, 0.088, -7.14902e-13, 0.926,0,0,0]) # 初始状态
    # 2) 闭环仿真
    ## 参数设置
    N_sim = 400  # 模拟步数
    sim_round = 2 # 模拟轮数
    all_simX = np.zeros((N_sim+1,config.Num_State,sim_round))
    all_simU = np.zeros((N_sim,config.Num_Input,sim_round))
    all_time = np.zeros((sim_round))

    groups = []

    for i in range(sim_round):
        # 初始化猜测
        ocp, ocp_solver, integrator = create_ocp_solver(x0)
        # u_guess, x_init_guess = config.GenerateRandomInitialGuess(i)
        # 初次initial guess设置
        u_guess = np.zeros(config.Num_Input)
        for j in range(config.Num_Input):
            u_guess[j] = np.round(np.random.uniform(-200, 200), 2)
        x_guess = np.zeros(config.Num_State)
#         x_guess = np.array([0.85114759,  0.22895749, -0.00465471, -1.69075889, -0.24523397,  0.20150836,
#   0., 0, 0, 0, 0, 0, 0,0, 0.3, 0.3, 0.5,0,0,0])
        for j in [0, 1, 2, 3, 4, 5, 6]:
            x_guess[j] = np.round(np.random.uniform(-2*np.pi, -2*np.pi), 2)
        for j in range(0,config.Horizon,20):
            # ocp_solver.set(j, "x", x_guess)
            ocp_solver.set(0, "u", u_guess)
        
        ocp_solver.set(0, "x", x0)
        # print("X_init_guess", x_init_guess)
        starttime = time.time()
        t, simX, simU, simCost, success= simulate_closed_loop(ocp, ocp_solver, integrator, x0, N_sim=N_sim)
        endtime = time.time()
        # print(config.compute_cost(simX, simU))
        elapsed_time = endtime - starttime
        
        if success:
            assigned = False
            for grp in groups:
                if np.allclose(simX, grp['rep'], atol=1e-2):
                    grp['indices'].append(i)
                    grp['simX_list'].append(simX)
                    grp['simU_list'].append(simU)
                    # grp['costs'].append(simCost)
                    assigned = True
                    break
            if not assigned:
                new_group_index = len(groups) # 新组的索引将是当前组的数量
                groups.append({
                    'rep': simX,
                    'indices': [i],
                    'simX_list': [simX],
                    'simU_list': [simU],
                })
                # 当创建新组时，调用绘图和保存函数
                plot_and_save_trajectory(simX, simU, new_group_index, N_sim)

            all_simX[:, :, i] = simX
            all_simU[:, :, i] = simU
            all_time[i] = elapsed_time
            print(f"Round {i}: success, u guess={u_guess}, time={elapsed_time:.4f}s")
            print("final x", simX[-1,:])
            # 5) 动画
            # animate()
        else:
            all_simX[:, :, i] = all_simX[:, :, 0]
            all_simU[:, :, i] = all_simU[:, :, 0]
            all_time[i] = 0
            print(f"Round {i}: failed")

    print("Distinct trajectory groups:", len(groups))
    savemat('franka_groups.mat', {'groups': groups})
        
   

    print("all_time: ", np.sum(all_time))
    print("time/turn: ", np.sum(all_time)/(N_sim*(sim_round)))
    
        
if __name__ == "__main__":
    main()