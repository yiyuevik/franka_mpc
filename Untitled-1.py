import os, time, random, numpy as np, casadi as ca, matplotlib.pyplot as plt
from urdf_parser_py.urdf import URDF
import urdf2casadi.urdfparser as u2c

#  global config
HORIZON   = 128
Ts        = 0.01
NX        = 14            # 7 q + 7 dq
NU        = 7
GRAVITY   = [0, 0, -9.81]
TAU_MAX   = 100.0
Qp        = np.eye(3)*500.0
R         = np.eye(NU)*1
P         = Qp
P_TARGET  = np.array([0.3, 0.3, 0.5])

# dynamic model
print("Building Franka symbolic model ...")
franka    = u2c.URDFparser()
urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "panda_arm.urdf")
franka.from_file(urdf_path)
root, tip = "panda_link0", "panda_link8"

M_fun  = franka.get_inertia_matrix_crba(root, tip)
C_fun  = franka.get_coriolis_rnea(root, tip)
G_fun  = franka.get_gravity_rnea(root, tip, GRAVITY)
fk_fun = franka.get_forward_kinematics(root, tip)["T_fk"]

def f_dyn(x, u):
    q, dq = x[:7], x[7:]
    ddq   = ca.solve(M_fun(q), u - C_fun(q, dq) - G_fun(q))
    return ca.vertcat(dq, ddq)

def rk4(x, u, dt=Ts):
    k1 = f_dyn(x, u)
    k2 = f_dyn(x+0.5*dt*k1, u)
    k3 = f_dyn(x+0.5*dt*k2, u)
    k4 = f_dyn(x+dt*k3, u)
    return x + dt/6*(k1+2*k2+2*k3+k4)

# 构建opti优化器
def build_opti():
    opti = ca.Opti()

    # 决策变量
    U = opti.variable(NU, HORIZON)
    X = opti.variable(NX, HORIZON+1)
    x0_par = opti.parameter(NX)              # 初始状态参数

    obj = 0 # 目标函数
    for k in range(HORIZON):
        qk = X[:7, k]
        uk = U[:, k]
        pk = fk_fun(qk)[:3, 3]
        dp = pk - P_TARGET
        obj += ca.mtimes([dp.T, Qp, dp]) + ca.mtimes([uk.T, R, uk])
        # dynamics约束
        opti.subject_to(X[:, k+1] == rk4(X[:, k], uk))

        # constraint on joint torque
        opti.subject_to(opti.bounded(-TAU_MAX, uk, TAU_MAX))
        # constraint on joint angles
        opti.subject_to(opti.bounded(-np.pi, X[:7, k], np.pi))

    # 终端成本
    qN = X[:7, -1]
    pN = fk_fun(qN)[:3, 3]
    obj += ca.mtimes([(pN-P_TARGET).T, P, (pN-P_TARGET)])

    opti.minimize(obj)

    # 初始状态等式约束
    opti.subject_to(X[:, 0] == x0_par)

    # IPOPT 设置
    opti.solver('ipopt', {'ipopt.print_level': 0,
                          'print_time': False,
                          'ipopt.max_iter': 500,
                          'ipopt.tol': 1e-4})

    return opti, U, X, x0_par

#
def simulate_closed_loop(x0_np, N_sim=80):
    opti, U, X, x0_par = build_opti()

    simX = np.zeros((N_sim+1, NX))
    simU = np.zeros((N_sim,   NU))
    simX[0, :] = x0_np

    # 首次 warm‑start 初始化为 0
    opti.set_initial(U, 0)
    opti.set_initial(X, np.tile(x0_np.reshape(-1,1), (1,HORIZON+1)))

    for k in range(N_sim):
        opti.set_value(x0_par, simX[k, :])

        try:
            start_time = time.time()
            sol = opti.solve()
            end_time = time.time()
            print(f"Step {k}: Solve time = {end_time - start_time:.3f} seconds")
        except RuntimeError as e:
            # 调试不收敛可用 opti.debug.show_infeasibilities()
            raise RuntimeError(f"MPC failed at step {k}: {e}")

        U_opt = sol.value(U)
        X_opt = sol.value(X)
        u0    = U_opt[:, 0]

        simU[k, :]   = u0
        simX[k+1, :] = rk4(ca.DM(simX[k, :]), ca.DM(u0)).full().ravel()

        # 热启动：用本次解移位作为下一次初猜
        opti.set_initial(U, np.hstack([U_opt[:,1:], U_opt[:,-1:]]))
        opti.set_initial(X, np.hstack([X_opt[:,1:], X_opt[:,-1:]]))

    return simX, simU

# 
if __name__ == "__main__":
    x0_np = np.array([ 0.0, 0.0, 0.0, 0, 0.0, 0, 0.0,
                       0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0 ])
    print("Running closed‑loop MPC with Opti ...")
    t0 = time.time()
    simX, simU = simulate_closed_loop(x0_np, N_sim=80)
    print(f"Done in {time.time()-t0:.1f}s")

    # ---------- 可视化 ----------
    pos = np.zeros((simX.shape[0], 3))
    for k in range(simX.shape[0]):
        pos[k, :] = fk_fun(ca.DM(simX[k, :7]))[:3, 3].full().ravel()

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(pos); ax[0].set_ylabel("EE pos (m)")
    ax[1].plot(simU); ax[1].set_ylabel("Torque (Nm)")
    ax[2].plot(simX[:, :7]); ax[2].set_ylabel("Joint (rad)")
    ax[2].set_xlabel("step")
    for a in ax: a.grid()
    ax[0].legend(["x","y","z"]); ax[1].legend([f"τ{i+1}" for i in range(7)])
    ax[2].legend([f"q{i+1}" for i in range(7)])
    plt.tight_layout(); plt.show()
    print("Final EE pos:", pos[-1, :])
    print("Final joint pos:", simX[-1, :7])
    print("Final joint vel:", simX[-1, 7:])
    print("Final joint torque:", simU[-1, :])
