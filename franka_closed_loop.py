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


def main():
    
    # 1) 生成初始状态样本
    x0 = np.array([ 0,0,0,0,0,0,0 , 0, 0, 0, 0, 0, 0, 0]) # 初始状态
    # 2) 闭环仿真
    ## 参数设置
    N_sim = 250  # 模拟步数
    sim_round = 1
    all_simX = np.zeros((N_sim+1,config.Num_State,sim_round))
    all_simU = np.zeros((N_sim,config.Num_Input,sim_round))
    all_time = np.zeros((sim_round))


    for i in range(sim_round):
        # 初始化猜测
        ocp, ocp_solver, integrator = create_ocp_solver(x0)
        u_guess, x_init_guess = config.GenerateRandomInitialGuess(i)
        # 初次initial guess设置
        for j in range(0,config.Horizon,20):
            ocp_solver.set(j, "x", x_init_guess)
        ocp_solver.set(0, "x", x0)
        print("X_init_guess", x_init_guess)
        starttime = time.time()
        t, simX, simU, simCost, success= simulate_closed_loop(ocp, ocp_solver, integrator, x0, x_init_guess, N_sim=N_sim)
        endtime = time.time()
        # print(config.compute_cost(simX, simU))
        elapsed_time = endtime - starttime

        if success:
            all_simX[:, :, i] = simX
            all_simU[:, :, i] = simU
            all_time[i] = elapsed_time
            print("final x", simX[-1,:])
            print(f"Simulation for initial x guess {x_init_guess} took {elapsed_time:.4f} seconds.")
            # 5) 动画
            # animate()
        else:
            all_simX[:, :, i] = all_simX[:, :, 0]
            all_simU[:, :, i] = all_simU[:, :, 0]
            all_time[i] = 0

    print("all_time: ", np.sum(all_time))
    print("time/turn: ", np.sum(all_time)/(N_sim*(sim_round)))
    
        
if __name__ == "__main__":
    main()