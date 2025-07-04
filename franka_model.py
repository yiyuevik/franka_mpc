"""
cartpole_model.py

定义 CartPole + 虚拟状态 (theta_stat) 的动力学模型，
供 ACADOS OCP 使用。
"""

import numpy as np
import casadi as ca
from acados_template import AcadosModel
from urdf_parser_py.urdf import URDF, Pose
import urdf2casadi.urdfparser as u2c
import config
import os

def export_franka_ode_model():
    """
    状态: q : 7个关节角度 + 7个关节角速度
    控制: tau : 7个关节力矩
    机械臂franka_emika_panda的动力学模型
    
    """

    franka = u2c.URDFparser()
    path_to_franka = absPath = os.path.dirname(os.path.abspath(__file__)) + '/urdf/panda_arm.urdf' 
    franka.from_file(path_to_franka)
    # 常量定义
    nx = config.Num_State  # 状态数量
    nq = config.Num_Q  # 关节数量
    nv = config.Num_Velocity  # 关节速度数量
    nu = config.Num_Input  # 控制输入数量
    x_sym = ca.SX.sym('x', nx)
    u_sym = ca.SX.sym('u', nu)         # [tau(7)]
    xdot_sym = ca.SX.sym('xdot', nx)
    
    
    q     = x_sym[:7] # 关节角度
    qdot  = x_sym[7:14]  # 关节角速度
    p     = x_sym[14:17]
    pdot  = x_sym[17:20] 
    tau   = u_sym   # 关节力矩

    root = config.root
    tip = config.tip
    M_sym = franka.get_inertia_matrix_crba(root, tip)
    C_sym = franka.get_coriolis_rnea(root, tip)
    G_sym = franka.get_gravity_rnea(root, tip, config.gravity)
    q0 = np.zeros((nq, 1)) # 初始关节角度
    qdot0 = np.zeros((nv, 1)) # 初始关节速度
    M = M_sym(q0)
    C = C_sym(q0, qdot0)
    G = G_sym(q0)
    eps   = 1e-4 
    qddot = ca.solve(M + eps*ca.SX.eye(7), tau - C - G)  # 计算加速度

    fk_dict = franka.get_forward_kinematics(root, tip)
    T_fk_fun = fk_dict["T_fk"]  # 4x4 CasADi Function
    T_fk_expr = T_fk_fun(q)        # 4x4 齐次矩阵表达式
    p_expr  = T_fk_expr[:3, 3]          # 末端位置 p(q)
    J_pos = ca.jacobian(p_expr, q) 
    
    # pddot = ca.mtimes(J_pos, qdot) # 错误

    # J_dot * qdot 的近似实现
    J_dot_qdot = ca.jacobian(J_pos @ qdot, q) @ qdot
    # 正确的末端线加速度
    pddot = J_pos @ qddot + J_dot_qdot


    f_expl = ca.vertcat(qdot, qddot, pdot, pddot) # 状态的导数
    f_impl = xdot_sym - f_expl # 隐式方程

    
    # 封装到 AcadosModel
    model = AcadosModel()
    model.name = 'franka_20dof'
    model.x = x_sym
    model.xdot = xdot_sym
    model.u = u_sym
    model.p = []
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.cost_y_expr = ca.vertcat(p, u_sym) # 目标位置 + 力矩
    model.cost_y_expr_e = p
    return model
