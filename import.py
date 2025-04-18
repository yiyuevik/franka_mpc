import casadi as ca
import numpy as np
import urdf2casadi.urdfparser as u2c

# 加载模型
parser = u2c.URDFparser()
parser.from_file("urdf/panda_arm.urdf")
fk = parser.get_forward_kinematics("panda_link0", "panda_link8")
T_fk_fun = fk["T_fk"]

# 设定符号变量
q = ca.SX.sym("q", 7)
T_fk = T_fk_fun(q)
p_fk = T_fk[:3, 3]

# 目标位置
p_target = np.array([0.3, 0.3, 0.5])
loss = ca.sumsqr(p_fk - p_target)

# 构建优化器
opti = ca.Opti()
q_var = opti.variable(7)
opti.minimize(ca.sumsqr(T_fk_fun(q_var)[:3, 3] - p_target))
opti.subject_to(opti.bounded(-2*np.pi, q_var, 2*np.pi))
opti.solver("ipopt")

# 求解
sol = opti.solve()
q_sol = sol.value(q_var)
print("IK 解:", q_sol)

q_var = np.array([1.18321348e-03 , 6.65923211e-04, -5.94291066e-03,  1.20477369e-03,
  1.23731236e-02, -3.37901204e-03,  1.73378993e-02])
print(T_fk_fun(q_var)[:3, 3])