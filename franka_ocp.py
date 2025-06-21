import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import config  # 引用 config.py
import scipy.linalg
import time
# 导入 CartPole 模型
from franka_model import export_franka_ode_model


def get_guess_from_solver_result(ocp_solver, N_horizon):
    u_init_guess = np.zeros((config.Num_Input, N_horizon))
    x_init_guess = np.zeros((config.Num_State, N_horizon+1))
    for i in range(N_horizon-1):
        u_init_guess[:, i] = ocp_solver.get(i+1, "u")
        x_init_guess[:, i] = ocp_solver.get(i+1, "x")
    u_init_guess[:, N_horizon-1] = ocp_solver.get(N_horizon-1, "u")
    x_init_guess[:, N_horizon-1] = ocp_solver.get(N_horizon, "x")
    x_init_guess[:, N_horizon] = ocp_solver.get(N_horizon, "x")
    return u_init_guess, x_init_guess

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
    ocp.dims.N = config.Horizon

    # 加载 CartPole 模型
    model = export_franka_ode_model()
    ocp.model = model
    ocp.model.x = model.x
    ocp.model.u = model.u

    # 成本函数设置
    p_target = np.array([0.3,0.3,0.5])  # 你可以自定义目标位置
    # q_target = np.array([-0.23626529,   2.57177868,   2.4392286 ,  -2.50680006,   4.65672233,
#    1.68178735, -23.22776577   ])  # 关节角度目标
    # q_target = np.array(-0.71053519, -0.03359194,  1.59255456, -2.67065679, -2.76585929,  0.29034389,
#   3.14156497)
    u_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 控制输入目标
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = model.cost_y_expr
    ocp.cost.W = scipy.linalg.block_diag(config.Q_p, config.R)
    ocp.cost.yref = np.concatenate((p_target, u_target))
    ocp.dims.ny = ocp.cost.yref.shape[0]
    

    # 终端成本
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = model.cost_y_expr_e
    ocp.cost.W_e = config.P
    ocp.cost.yref_e = p_target
    ocp.dims.ny_e = ocp.cost.yref_e.shape[0]
    
    # 约束条件

    ocp.constraints.idxbu = np.arange(Nu)
    ocp.constraints.lbu = -config.tau_max*np.ones(Nu)
    ocp.constraints.ubu = +config.tau_max*np.ones(Nu)

    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6])   # 只列出前 7 个索引
    
    ocp.constraints.lbx = -1*np.pi * np.ones(7)  # 下界
    ocp.constraints.ubx = 1* np.pi * np.ones(7)

    #初始状态约束
    ocp.constraints.x0 = x0


    # 求解器设置
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.scaling = 10
    ocp.solver_options.integrator_type = 'DISCRETE'  # 改为离散
    ocp.solver_options.collocation_type = 'GAUSS_RADAU_IIA'  # 设置配置法类型
    ocp.solver_options.sim_method_num_stages = 4  # 配置点数
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 400
    ocp.solver_options.nlp_solver_tol_stat = 5e-3
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    # 构造 OCP 求解器
    isConstruct = False
    
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_double_pendulum.json", build=isConstruct, generate=isConstruct)

    acados_integrator = AcadosSimSolver(ocp, json_file = "acados_ocp_double_pendulum.json", build=isConstruct, generate=isConstruct)
    return ocp, acados_solver, acados_integrator


def simulate_closed_loop(ocp, ocp_solver, integrator, x0, N_sim=50,nMaxGuess: int = 1, M=10):
    
    nx = ocp.model.x.size()[0]  # Should be 14
    nu = ocp.model.u.size()[0]  # Should be 7

    # 初始状态
    simX = np.zeros((N_sim+1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    simX[0, :] = x0  # 初始化状态为传入的 x0

    # 闭环仿真
    for i in range(N_sim):
        retries = 0
        success = True
        while retries < nMaxGuess:
            try:
                u_opt = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
                #设置下一个sim的初始猜测
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)

                # print(ocp_solver.get(config.Horizon, "x"))

                if i % M == 0:
                    ocp_solver.reset()
                    for j in range(config.Horizon):
                        ocp_solver.set(j, "u", u_guess[:, j] + 0.1 * np.random.randn(nu))
                        ocp_solver.set(j, "x", x_guess[:, j] + 0.1 * np.random.randn(nx))
                    ocp_solver.set(config.Horizon, "x", x_guess[:, -1]+ 0.2 * np.random.randn(nx))
                else:
                    ocp_solver.reset()
                    for j in range(config.Horizon):
                        ocp_solver.set(j, "u", u_guess[:, j] + 0.02 * np.random.randn(nu))
                        ocp_solver.set(j, "x", x_guess[:, j] + 0.02 * np.random.randn(nx))
                    ocp_solver.set(config.Horizon, "x", x_guess[:, -1]+ 0.05 * np.random.randn(nx))

                simU[i, :] = u_opt
                # if i == 0:
                #     print("u_opt", u_opt)
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
                ocp_solver.reset()
                for j in range(config.Horizon):
                    ocp_solver.set(j, "u", u_guess[:, j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(config.Horizon, "x", x_guess[:, -1])
            retries += 1
            
        if success == False and retries >= nMaxGuess:
            print("MPC solve failed after max retries")
            break
            

    # print("x_final:", simX[-1,:])
    ocp_solver.reset()
    t = np.linspace(0, N_sim*config.Ts, N_sim+1)
    success = True 
    return t, simX, simU, simCost, success
