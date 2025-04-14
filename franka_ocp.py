"""
cartpole_ocp.py

一个示例：从 config.py 中读取 Q, R, P, Horizon, Ts 等，来搭建 OCP。
"""

import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import config  # 引用 config.py
import scipy.linalg
import time
# 导入 CartPole 模型
from double_pendulum_model import export_cartpole_ode_model

def clear_solver_state(ocp_solver, N_horizon):
    # 清空状态和控制输入
    for i in range(N_horizon):
        ocp_solver.set(i, "x", np.zeros_like(ocp_solver.get(i,"x")))
        ocp_solver.set(i, "u", np.zeros_like(ocp_solver.get(i,"u")))
    ocp_solver.set(N_horizon, "x", np.zeros_like(ocp_solver.get(N_horizon,"x")))

def get_guess_from_solver_result(ocp_solver, N_horizon):
    u_guess = np.zeros(N_horizon)
    x_guess = np.zeros((config.Num_State, N_horizon+1))
    for i in range(N_horizon-1):
        u_guess[i] = ocp_solver.get(i+1, "u")
        x_guess[:, i] = ocp_solver.get(i+1, "x")
    u_guess[N_horizon-1] = ocp_solver.get(N_horizon-1, "u")
    x_guess[:, N_horizon-1] = ocp_solver.get(N_horizon, "x")
    x_guess[:, N_horizon] = ocp_solver.get(N_horizon, "x")
    return u_guess, x_guess

def create_ocp_solver(x0):
    ocp = AcadosOcp()

    # 读取 config 里的各种参数
    Nx = config.Num_State
    Nu = config.Num_Input
    N  = config.Horizon
    tf = N * config.Ts   # 总时域 tf = N_horizon * Ts

    # 设置 OCP 参数
    ocp.solver_options.N_horizon = N  # 设置预测步数
    ocp.solver_options.tf = tf       # 设置总时域


    # 加载 CartPole 模型
    model = export_cartpole_ode_model()
    ocp.model = model
    ocp.model.x = model.x
    ocp.model.u = model.u

    # 成本函数设置
    ocp.cost.cost_type = 'NONLINEAR_LS'
    
    ocp.model.cost_y_expr = ca.vertcat(model.x[0],  # x (位置)
                             model.x[1],  # xdot (速度)
                             np.sin(model.x[2]/2),  # theta1
                             model.x[3],  # omega1
                             np.sin(model.x[4]/2), # theta2
                             model.x[5], # omega2
                             model.u) 
    # ocp.model.cost_y_expr =  ca.vertcat(model.x, model.u)
    ocp.cost.W = scipy.linalg.block_diag(config.Q, config.R)

    ocp.cost.yref = np.zeros(Nx + Nu)  # (6维)

    # 终端成本
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ca.vertcat(model.x[0],  # x (位置)
                             model.x[1],  # xdot (速度)
                             np.sin(model.x[2]/2),  # cos(theta1)
                             model.x[3],  # omega1
                             np.sin(model.x[4]/2),  # cos(theta2)
                             model.x[5])  # omega2
    # ocp.model.cost_y_expr_e =  ca.vertcat(model.x)
    ocp.cost.W_e = config.P

    ocp.cost.yref_e = np.zeros(Nx)     # (5维)
    
    # 约束条件
    ocp.constraints.x0 = x0
    ocp.constraints.lbu = np.array([-config.Fmax])
    ocp.constraints.ubu = np.array([+config.Fmax])
    # ocp.constraints.ubx = np.array([20])
    # ocp.constraints.lbx = np.array([-20])
    ocp.constraints.idxbu = np.array([0])  # 控制量u只有1维
    # ocp.constraints.idxbx = np.array([0])

    # 求解器设置
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 400
    ocp.solver_options.nlp_solver_tol_stat = 5e-3
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    # 构造 OCP 求解器
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_double_pendulum.json")

    acados_integrator = AcadosSimSolver(ocp, json_file = "acados_ocp_double_pendulum.json")

    return ocp, acados_solver, acados_integrator


def simulate_closed_loop(ocp, ocp_solver, integrator, x0, x_init_guess, N_sim=50,nMaxGuess: int = 1):
    
    nx = ocp.model.x.size()[0]  # Should be 6
    nu = ocp.model.u.size()[0]  # Should be 1

    # 初始状态
    simX = np.zeros((N_sim+1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    simX[0, :] = x0  # 初始化状态为传入的 x0
    success = True
    # 闭环仿真
    for i in range(N_sim):
        retries = 0
        while retries < nMaxGuess:
            try:
                u_opt = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
                #设置下一个sim的初始猜测
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
                clear_solver_state(ocp_solver, config.Horizon) #按道理不太需要
                
                for j in range(config.Horizon):
                    ocp_solver.set(j, "u", u_guess[j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(config.Horizon, "x", x_guess[:, -1])

                simU[i, :] = u_opt
                # print("u_opt", u_opt)
                # 更新状态
                x_next = integrator.simulate(x=simX[i, :], u=u_opt)
                simX[i+1, :] = x_next
                simCost[i,:] = ocp_solver.get_cost()
                # print("i:",i,"x:",x_next)
                break
            except Exception as e:
                success = False
                print(f"Error in MPC_solve: {str(e)}, change guess")
                print("current step:", i, ", retries=", retries)
                time.sleep(2)
                if i == 0 :
                    clear_solver_state(ocp_solver, config.Horizon)
                    for j in range(0,config.Horizon,20):
                        ocp_solver.set(j, "x", x_init_guess)
                    ocp_solver.set(0, "x", x0)
                else:
                    u_range = np.max(u_guess) - np.min(u_guess)
                    x_range = np.max(x_guess) - np.min(x_guess)
                    u_noise_range = 0.1 * u_range
                    x_noise_range = 0.1 * x_range
                    u_increment = np.random.uniform(-u_noise_range, u_noise_range, size=u_guess.shape)
                    x_increment = np.random.uniform(-x_noise_range, x_noise_range, size=x_guess.shape)
                    u_guess = u_guess + u_increment
                    x_guess = x_guess + x_increment
                    for j in range(config.Horizon):
                        ocp_solver.set(j, "u", u_guess[j])
                        ocp_solver.set(j, "x", x_guess[:, j])
                    ocp_solver.set(config.Horizon, "x", x_guess[:, -1])
            
            retries += 1
            if retries == nMaxGuess-1:
                print("set initial guess = 2pi")
                for j in range(0,config.Horizon,20):
                        ocp_solver.set(j, "x", np.array([0,0,0,0,0,0]))
                ocp_solver.set(0, "x", np.array([0,0,0,0,0,0]))
        if success == False:
            print("MPC solve failed after max retries")
            break
            
    

    # print("x_final:", simX[-1,:])
    clear_solver_state(ocp_solver, config.Horizon)
    t = np.linspace(0, N_sim*config.Ts, N_sim+1)
    return t, simX, simU, simCost, success
