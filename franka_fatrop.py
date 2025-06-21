import casadi as ca
import numpy as np
import config
from franka_model import export_franka_ode_model

# 设置参数
N = config.Horizon
nx = config.Num_State
nu = config.Num_Input
Ts = config.Ts
target_pos = np.array([0.3, 0.3, 0.5])
tau_max = config.tau_max

model = export_franka_ode_model()
x_sym = model.x
u_sym = model.u
f_expl = model.f_expl_expr

# opti参数
opti = ca.Opti()
X = opti.variable(nx, N+1)
U = opti.variable(nu, N)
x0_param = opti.parameter(nx)
opti.subject_to(X[:, 0] == x0_param)

def f_disc(x, u):
    return x + Ts * ca.Function('f', [x_sym, u_sym], [f_expl])(x, u)

for k in range(N):
    opti.subject_to(X[:, k+1] == f_disc(X[:, k], U[:, k]))

# 成本函数
W = ca.blockcat([[config.Q, ca.MX.zeros((3, nu))],
                 [ca.MX.zeros((nu, 3)), config.R]])
cost = 0
for k in range(N):
    yk = ca.vertcat(X[14:17, k], U[:, k])
    yref_k = ca.vertcat(target_pos, ca.DM.zeros(nu, 1))
    cost += (yk - yref_k).T @ W @ (yk - yref_k)

y_e = X[14:17, N]
cost += (y_e - target_pos).T @ config.P @ (y_e - target_pos)
opti.minimize(cost)


for k in range(N):
    # 每个阶段单独加力矩上下界
    opti.subject_to(opti.bounded(-tau_max, U[:, k], tau_max))
    # 每个阶段单独加关节角度上下界
    opti.subject_to(opti.bounded(-np.pi,    X[:7, k],   np.pi))


options = {}
# options["expand"] = True
options["fatrop"] = {"mu_init": 0.1}
# options["structure_detection"] = "auto"
# options["debug"] = True
opti.solver("fatrop", options)

# simulation
def simulate_closed_loop_fatrop(x0_init, N_sim):
    simX = np.zeros((N_sim+1, nx))
    simU = np.zeros((N_sim, nu))
    simX[0, :] = x0_init

    f_step = ca.Function('f', [x_sym, u_sym], [f_expl])

    for t in range(N_sim):
        opti.set_value(x0_param, simX[t, :])
        sol = opti.solve()
        u_opt = sol.value(U[:, 0])
        simU[t, :] = u_opt
        x_next = f_step(simX[t, :], u_opt).full().flatten() * Ts + simX[t, :]
        simX[t+1, :] = x_next

    return simX, simU


if __name__ == "__main__":
    x0 = np.array([ 0,0,0,0,0,0,0 , 0, 0, 0, 0, 0, 0,0, 0.088, -7.14902e-13, 0.926,0,0,0])
    simX, simU = simulate_closed_loop_fatrop(x0, N_sim=1000)
    print("Final state:", simX[-1])
