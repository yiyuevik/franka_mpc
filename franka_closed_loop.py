"""
cartpole_closed_loop.py
我没有写main.py
此即为主入口脚本：读取/设置模型参数(在 config.py)，构造并求解 OCP，然后进行闭环仿真 + 可视化。
运行方式: python cartpole_closed_loop.py
"""


import config
from double_pendulum_ocp import create_ocp_solver, simulate_closed_loop  
from double_pendulum_utils import plot_cartpole_trajectories, animate_cartpole
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

def main():

    # 1) 生成初始状态样本
    x0 = np.array([-3, 0, np.pi, 0, 2.98, 0])
    # 2) 闭环仿真
    ## 参数设置
    N_sim = 120  # 模拟步数
    sim_round = 626 
    all_simX = np.zeros((N_sim+1,config.Num_State,sim_round))
    all_simU = np.zeros((N_sim,1,sim_round))
    all_time = np.zeros((sim_round))
    
    SimX_typ = []
    x_init_guess_typ = []
     # 用于归类相似结果的分组列表
    # 每个分组以一个字典形式存储：
    #   'simX_rep'         : 该分组的代表 simX（用作比较）
    #   'indices'          : 所属分组的仿真编号列表
    #   'simX_list'        : 当前组内所有 simX 数组（例如，list中每个元素形状为 (N_sim+1, num_state)）
    #   'simU_list'        : 当前组内所有 simU 数组
    #   'x_init_guess_list': 当前组内对应的初始猜测数组
    #   'cost'            : 当前组内所有成本值
    groups = []


    for i in range(sim_round):
        # 初始化猜测
        ocp, ocp_solver, integrator = create_ocp_solver(x0)
        u_guess, x_init_guess = config.GenerateRandomInitialGuess(i)
        u0 = np.random.uniform(-1e3, 1e3, config.Horizon)
        # 初次initial guess设置
        for j in range(0,config.Horizon,20):
            # ocp_solver.set(j, "u", u0[j])
            ocp_solver.set(j, "x", x_init_guess)
        ocp_solver.set(0, "x", x0)
        print("X_init_guess", x_init_guess)
        print("i: ",i)
        starttime = time.time()
        t, simX, simU, simCost, success= simulate_closed_loop(ocp, ocp_solver, integrator, x0, x_init_guess, N_sim=N_sim)
        endtime = time.time()
        # print(config.compute_cost(simX, simU))
        elapsed_time = endtime - starttime

        if success:
            found_group = False
            for group in groups:
                if np.allclose(simX, group['simX_rep'], atol=0.1):
                    # 如果相似，则添加到该分组
                    group['indices'].append(i)
                    group['simX_list'].append(simX)
                    group['simU_list'].append(simU)
                    group['x_init_guess_list'].append(x_init_guess)
                    group['cost'].append(simCost)
                    found_group = True
                    break

            if not found_group:
                # 没有相似的分组，则创建一个新的分组
                groups.append({
                    'simX_rep': simX,
                    'indices': [i],
                    'simX_list': [simX],
                    'simU_list': [simU],
                    'x_init_guess_list': [x_init_guess],
                    'cost' : [simCost]
                })


            all_simX[:, :, i] = simX
            all_simU[:, :, i] = simU
            all_time[i] = elapsed_time
            print("final x", simX[-1,:])
            print(f"Simulation for initial x guess {x_init_guess} took {elapsed_time:.4f} seconds.")

            # 5) 绘制曲线
            # plot_cartpole_trajectories(t, simX, simU)
            # 6) 动画
            # animate_cartpole(t, simX, interval=50)
        else:
            all_simX[:, :, i] = all_simX[:, :, 0]
            all_simU[:, :, i] = all_simU[:, :, 0]
            all_time[i] = 0

    print("all_time: ", np.sum(all_time))
    print("time/turn: ", np.sum(all_time)/(N_sim*(sim_round)))
    print("SimX_typ: ", len(SimX_typ))
    print("Number of distinct simX groups:", len(groups))



    np.savez('sim_results.npz', groups=groups)
    savemat('groups.mat', {'groups': groups})

    theta_values = all_simX[:, 2, :]

    # 计算每个步骤的最大值、最小值和中位数
    theta_max = np.max(theta_values, axis=1)
    theta_min = np.min(theta_values, axis=1)
    theta_median = np.median(theta_values, axis=1)

    # 绘制 theta 的范围图
    plt.figure(figsize=(12, 6))

    # 填充最大值和最小值之间的区域，表示范围
    plt.fill_between(range(N_sim+1), theta_min, theta_max, color='lightblue', label='Range', alpha=0.5)

    # 绘制中位数线
    plt.plot(range(N_sim+1), theta_median, color='blue', label='Median', linewidth=2)

    # 设置图形标题和标签
    plt.title('Theta (State 3) Range and Median Over Time')
    plt.xlabel('Step')
    plt.ylabel('Theta Value')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()   
    # steps = 50
    # num_x0 = 10  # 10个x0

    # # 1. 将数据重新整理为适合绘制Boxplot的格式
    # data_for_boxplot = []

    # for step in range(steps):
    #     # 提取每个步长对应的10个控制输入值
    #     data_for_boxplot.append(all_simU[step, 0, :])

    # # 将数据转化为numpy数组，以便 matplotlib 进行绘制
    # data_for_boxplot = np.array(data_for_boxplot).T  # 转置，确保每一列为一个控制输入的样本

    # # 2. 使用matplotlib绘制Boxplot
    # plt.figure(figsize=(12, 6))

    # # 使用matplotlib的boxplot绘制
    # plt.boxplot(data_for_boxplot, widths=0.6)

    # # 3. 设置标签和标题
    # plt.title('Control Input in Dataset')
    # plt.xlabel('Step')
    # plt.ylabel('Control Input (ctrl)')

    # # 设置x轴标签
    # xticks = [i*10 + 5 for i in range(steps // 10)]  # 每10个步长在x轴上显示一个标签
    # plt.xticks(xticks, [str(i) for i in range(0, steps, 10)])

    # # 显示图形
    # plt.show()
        
if __name__ == "__main__":
    main()