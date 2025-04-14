"""
cartpole_model.py

定义 CartPole + 虚拟状态 (theta_stat) 的动力学模型，
供 ACADOS OCP 使用。
"""

import numpy as np
import casadi as ca
from acados_template import AcadosModel

def export_cartpole_ode_model():
    """
    状态: x = [ pos, xdot, theta, theta_dot, theta_stat ]
    控制: u = [ F ]

    动力学方程参考: 老师给的 dynamic_update_virtual_Casadi
    """
    # 常量定义
    M_CART = 2.0       # 小车质量
    M_PEND1 = 1.0       # 摆杆1质量
    M_PEND2 = 1.0       # 摆杆2质量
    l1 = 0.5       # 摆杆1质心长度
    L1 = 1.0       # 摆杆1总长度
    l2 = 0.5       # 摆杆2质心长度
    L2 = 1.0       # 摆杆2总长度
    G = 9.81        # 重力加速度
    I1 = 0.0126    # 摆杆1惯性矩
    I2 = 0.0185    # 摆杆2惯性矩


    # CasADi 符号
    x_sym = ca.SX.sym('x', 6)  # [x0, x1, x2, x3, x4, x5] 假设状态数量为6
    u_sym = ca.SX.sym('u', 1)  # 控制输入

    # 提取各个变量
    pos = x_sym[0]
    vel = x_sym[1]
    theta1 = x_sym[2]
    omega1 = x_sym[3]
    theta2 = x_sym[4]
    omega2 = x_sym[5]
    # theta1_stat = x_sym[6]
    # theta2_stat = x_sym[7]
    F = u_sym[0]

    # 动力学
     # 计算矩阵 Mt (3×3)
    Mt = ca.vertcat(
            ca.horzcat(M_CART + M_PEND1 + M_PEND2, L1*(0.5*M_PEND1+M_PEND2)*ca.cos(theta1),       0.5*M_PEND2*L2*ca.cos(theta2)),
            ca.horzcat(L1*(0.5*M_PEND1+M_PEND2)*ca.cos(theta1), L1**2*((1/3)*M_PEND1+M_PEND2),         0.5*L1*L2*M_PEND2*ca.cos(theta1-theta2)),
            ca.horzcat(0.5*L2*M_PEND2*ca.cos(theta2),    0.5*L1*L2*M_PEND2*ca.cos(theta1-theta2), (1/3)*L2**2*M_PEND2)
         )

    # 计算 f1, f2, f3
    f1 = L1*(0.5*M_PEND1+M_PEND2)*omega1**2 * ca.sin(theta1) + 0.5*M_PEND2*L2*omega2**2 * ca.sin(theta2) + F
    f2 = -0.5* L1*L2*M_PEND2*omega2**2 * ca.sin(theta1-theta2) + G*(0.5*M_PEND1+M_PEND2)*L1 * ca.sin(theta1)
    f3 =  0.5 * L1*L2*M_PEND2*omega1**2 * ca.sin(theta1-theta2) + 0.5*G*L2*M_PEND2 * ca.sin(theta2)
   

    # 状态的导数
    dx_pos = vel
    dx_theta1 = omega1
    dx_theta2 = omega2
    sol = ca.solve(Mt, ca.vertcat(f1, f2, f3))
    dx_vel    = sol[0]
    dx_omega1 = sol[1]
    dx_omega2 = sol[2]
    # dx_theta1_stat = -(2.0/np.pi)*(theta1-np.pi)*omega1
    # dx_theta2_stat = -(2.0/np.pi)*(theta2-np.pi)*omega2
    # 合并成一个列向量
    f_expl = ca.vertcat(dx_pos, dx_vel, dx_theta1, dx_omega1, dx_theta2, dx_omega2)

    # 隐式方程
    xdot_sym = ca.SX.sym('xdot', 6)
    f_impl = xdot_sym - f_expl

    # 封装到 AcadosModel
    model = AcadosModel()
    model.name = "double_pendulum_6states"
    model.x = x_sym
    model.xdot = xdot_sym
    model.u = u_sym
    model.p = []
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    return model
