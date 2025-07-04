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
import urdf2casadi.urdfparser as u2c
import os
import random

def fk_position(T_fk_fun,q_row):
    T = T_fk_fun(q_row[:7])       
    p = T[:3, 3]               
    return np.array(p).reshape(3)

def main():
    # NUM_SEED = 7
    # np.random.seed(NUM_SEED)
    # random.seed(NUM_SEED)
    # 1) 生成初始状态样本
    x0 = np.array([ 0,0,0,0,0,0,0 , 0, 0, 0, 0, 0, 0,0]) # 初始状态
    # 2) 闭环仿真
    ## 参数设置
    N_sim = 400  # 模拟步数
    sim_round = 1
    all_simX = np.zeros((N_sim+1,config.Num_State+3,sim_round))
    all_simU = np.zeros((N_sim,config.Num_Input,sim_round))
    all_time = np.zeros((sim_round))
    franka = u2c.URDFparser()
    path_to_franka = absPath = os.path.dirname(os.path.abspath(__file__)) + '/urdf/panda_arm.urdf' 
    franka.from_file(path_to_franka)
    fk_dict = franka.get_forward_kinematics(config.root, config.tip)
    T_fk_fun = fk_dict["T_fk"]


    for i in range(sim_round):
        # 初始化猜测
        ocp, ocp_solver, integrator = create_ocp_solver(x0)
        u_guess = config.GenerateRandomInitialGuess()
        u_guess= np.array([ 0.,     0.,     0.,     3.8,   -11.54,   0.,     6.89])
        # 初次initial guess设置
        ocp_solver.set(0, "x", x0)
        for j in range(0,config.Horizon):
            ocp_solver.set(j, "u", u_guess)
  
        starttime = time.time()
        t, simX, simU, simCost, success= simulate_closed_loop(ocp, ocp_solver, integrator, x0, N_sim=N_sim)
        endtime = time.time()

        pos = np.apply_along_axis(lambda q_i: fk_position(T_fk_fun,q_i), 1, simX) 
        simX = np.hstack((simX, pos))
        # print(config.compute_cost(simX, simU))
        elapsed_time = endtime - starttime
        # print("round:", i, "u", simU)
        # print("x", simX)
        if success:
            all_simX[:, :, i] = simX
            all_simU[:, :, i] = simU
            all_time[i] = elapsed_time
            print("final x", simX[-1,:])
            print(f"Simulation for initial u guess {u_guess} took {elapsed_time:.4f} seconds.")
            # 5) 动画
            # animate()
        else:
            all_simX[:, :, i] = all_simX[:, :, 0]
            all_simU[:, :, i] = all_simU[:, :, 0]
            all_time[i] = 0
        
        pos = simX[:, 14:17]
        joint = simX[:, :7]
        u = simU

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(pos)
        axs[0].set_ylabel("End-Effector Position (m)")
        axs[0].legend(["x", "y", "z"])
        axs[0].grid()

        axs[1].plot(u)
        axs[1].set_ylabel("Joint Torque (Nm)")
        axs[1].legend([f"τ{i+1}" for i in range(7)])
        axs[1].grid()

        axs[2].plot(joint)
        axs[2].set_ylabel("Joint Position (rad)")
        axs[2].set_xlabel("Time step")
        axs[2].legend([f"q{i+1}" for i in range(7)])
        axs[2].grid()

        plt.tight_layout()
        plt.savefig("trajectory.png")


        # 创建3D图展示末端轨迹
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 末端轨迹
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='End Effector Path')
        
        # 起点和终点
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], 
                c='g', s=100, label='Start')
        ax.scatter(0.3, 0.3, 0.5, 
                c='r', s=100, label='Target')
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], 
                c='g', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End Effector Trajectory')
        ax.legend()
        plt.savefig("trajectory_3d.png")

            
    pos = simX[:, 14:17]
    joint = simX[:, :7]
    u = simU

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(pos)
    axs[0].set_ylabel("End-Effector Position (m)")
    axs[0].legend(["x", "y", "z"])
    axs[0].grid()

    axs[1].plot(u)
    axs[1].set_ylabel("Joint Torque (Nm)")
    axs[1].legend([f"τ{i+1}" for i in range(7)])
    axs[1].grid()

    axs[2].plot(joint)
    axs[2].set_ylabel("Joint Position (rad)")
    axs[2].set_xlabel("Time step")
    axs[2].legend([f"q{i+1}" for i in range(7)])
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    # # 保存仿真数据
    # import datetime
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # # 创建保存目录
    # save_dir = "simulation_results"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # # 保存数据
    # np.save(f"{save_dir}/all_simX_{timestamp}.npy", all_simX)
    # np.save(f"{save_dir}/all_simU_{timestamp}.npy", all_simU)
    # np.save(f"{save_dir}/all_time_{timestamp}.npy", all_time)
    
    # # 保存仿真参数信息
    # sim_info = {
    #     'N_sim': N_sim,
    #     'sim_round': sim_round,
    #     'x0': x0.tolist(),
    #     'timestamp': timestamp,
    #     'total_time': np.sum(all_time),
    #     'avg_time_per_step': np.sum(all_time)/(N_sim*sim_round)
    # }
    
    # import json
    # with open(f"{save_dir}/sim_info_{timestamp}.json", 'w') as f:
    #     json.dump(sim_info, f, indent=2)
    
    # print(f"数据已保存到 {save_dir}/ 目录:")
    # print(f"  - all_simX_{timestamp}.npy")
    # print(f"  - all_simU_{timestamp}.npy") 
    # print(f"  - all_time_{timestamp}.npy")
    # print(f"  - sim_info_{timestamp}.json")

    print("all_time: ", np.sum(all_time))
    print("time/turn: ", np.sum(all_time)/(N_sim*(sim_round)))

        
if __name__ == "__main__":
    main()