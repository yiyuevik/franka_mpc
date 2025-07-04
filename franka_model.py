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
    x_sym = ca.SX.sym('x', nx)       # [q(7), qdot(7)]
    u_sym = ca.SX.sym('u', nu)         # [tau(7)]
    xdot_sym = ca.SX.sym('xdot', nx) # [qdot(7), qddot(7)]

    
    q     = x_sym[:7] # 关节角度
    qdot  = x_sym[7:14]  # 关节角速度
    tau   = u_sym   # 关节力矩

    root = config.root
    tip = config.tip
    M_sym = franka.get_inertia_matrix_crba(root, tip)
    C_sym = franka.get_coriolis_rnea(root, tip)
    G_sym = franka.get_gravity_rnea(root, tip, config.gravity)

    # Dynamics
    M = M_sym(q)
    C = C_sym(q, qdot)  
    G = G_sym(q)
    qddot = ca.solve(M + 1e-3*ca.SX.eye(7), tau - C - G)
    
    # 运动学作为输出，不作为状态
    fk_dict = franka.get_forward_kinematics(config.root, config.tip)
    T_fk_fun = fk_dict["T_fk"]
    T_fk_expr = T_fk_fun(q)
    p_expr = T_fk_expr[:3, 3]  # 末端位置
    
    f_expl = ca.vertcat(qdot, qddot)
    f_impl = xdot_sym - f_expl

    model = AcadosModel()
    model.name = 'franka_14dof'
    model.x = x_sym
    model.xdot = xdot_sym
    model.u = u_sym
    model.p = []
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.cost_y_expr = ca.vertcat(p_expr, tau) 
    model.cost_y_expr_e = ca.vertcat(p_expr)     
    
    return model
