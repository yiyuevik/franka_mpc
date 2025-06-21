import casadi as ca
import numpy as np
import urdf2casadi.urdfparser as u2c
import time

parser = u2c.URDFparser()
parser.from_file("urdf/panda_arm.urdf")
root = "panda_link0"
tip  = "panda_link8"

# 设置符号变量
q   = ca.SX.sym("q", 7)
qdot = ca.SX.sym("qdot", 7)
x = ca.vertcat(q, qdot)
u = ca.SX.sym("u", 7)

# 动力学函数
M = parser.get_inertia_matrix_crba(root, tip)(q)
G = parser.get_gravity_rnea(root, tip, [0,0,-9.81])(q)
C = parser.get_coriolis_rnea(root, tip)(q, qdot)
qddot = ca.solve(M, u - C - G)
xdot = ca.vertcat(qdot, qddot)
xdot_fun = ca.Function("xdot_fun", [x, u], [xdot])

# 末端位姿
T_fk_fun = parser.get_forward_kinematics(root, tip)["T_fk"]
T_fk = T_fk_fun(q)
p_fk = T_fk[:3,3]

# 离散化参数
N = 40
T = 4.0
h = T / N

# 创建 OCP
opti = ca.Opti()
x_var = opti.variable(14, N+1)
u_var = opti.variable(7, N)

# 初始条件
x0 = np.zeros(14)
x0[:7] = [0, 0, 0, -1.5, 0, 1.5, 0]
opti.subject_to(x_var[:,0] == x0)

# 动力学约束 (Euler)
for k in range(N):
    x_next = x_var[:,k] + h * xdot_fun(x_var[:,k], u_var[:,k])
    opti.subject_to(x_var[:,k+1] == x_next)

# 控制约束
opti.subject_to(opti.bounded(-87, u_var, 87))

# 成本函数：追踪末端目标位置 + 控制正则项
p_target = np.array([0.3, 0.3, 0.5])
Qp = np.eye(3)*100
R  = np.eye(7)*1
cost = 0
for k in range(N):
    qk = x_var[:7,k]
    pk = T_fk_fun(qk)[:3,3]
    cost += ca.mtimes([(pk - p_target).T, Qp, (pk - p_target)])
    cost += ca.mtimes([u_var[:,k].T, R, u_var[:,k]])
# 终端 cost
qN = x_var[:7,N]
pN = T_fk_fun(qN)[:3,3]
cost += ca.mtimes([(pN - p_target).T, Qp*5, (pN - p_target)])

opti.minimize(cost)

# solver
opti.solver("ipopt", {"ipopt.print_level":0, "print_time":False})
starttime = time.time()
sol = opti.solve()
endtime = time.time()
print("求解时间:", endtime - starttime, "秒")

# 提取末态关节位置 q_final
x_sol = sol.value(x_var)
q_final = x_sol[:7, -1]
print("q_ref =", np.array2string(q_final, separator=', '))