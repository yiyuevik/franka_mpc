import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg
import time

import config
from models.franka_model import export_franka_ode_model
from utils.helpers import clear_solver_state, get_guess_from_solver_result

def create_ocp_solver(x0):
    """
    Create and configure the ACADOS OCP and solver for the Franka Panda robot.
    Returns the OCP model, OCP solver, and simulation integrator.
    """
    ocp = AcadosOcp()
    
    # Load parameters from config
    Nx = config.Num_State
    Nu = config.Num_Input
    N  = config.Horizon
    tf = N * config.Ts  # total time horizon length (seconds)
    
    # Set prediction horizon length and total time
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = tf
    
    # Import the robot model dynamics
    model = export_franka_ode_model()
    ocp.model = model
    ocp.model.x = model.x
    ocp.model.u = model.u
    
    # Cost function setup (least-squares)
    p_target = config.target_position
    u_target = config.target_torque
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = model.cost_y_expr
    ocp.cost.W = scipy.linalg.block_diag(config.Q, config.R)
    ocp.cost.yref = np.concatenate((p_target, u_target))
    ocp.dims.ny = ocp.cost.yref.shape[0]
    
    # Terminal cost setup (least-squares)
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = model.cost_y_expr_e
    ocp.cost.W_e = config.P
    ocp.cost.yref_e = p_target
    ocp.dims.ny_e = ocp.cost.yref_e.shape[0]
    
    # Constraints: initial state is fixed to x0. 
    q_min = np.array([
        -2.8973,
        -1.7628,
        -2.8973,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973
    ])
    q_max = np.array([
        2.8973,
        1.7628,
        2.8973,
        -0.0698,
        2.8973,
        3.7525,
        2.8973
    ])
    ocp.constraints.idxbx = np.arange(7)
    ocp.constraints.lbx = q_min
    ocp.constraints.ubx = q_max
    ocp.constraints.x0 = x0
    
    # Solver settings
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 400
    ocp.solver_options.nlp_solver_tol_stat = 5e-4
    ocp.solver_options.levenberg_marquardt = 1.0
    
    # Create ACADOS solver and integrator
    # True for generate, build, and compile the C code
    # False for just loading the pre-generated JSON file
    is_generate = False
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_franka.json", generate=is_generate, build=is_generate)
    acados_integrator = AcadosSimSolver(ocp, json_file="acados_ocp_franka.json", generate=is_generate, build=is_generate)
    return ocp, acados_solver, acados_integrator

def simulate_closed_loop(ocp, ocp_solver, integrator, x0, N_sim=50, nMaxGuess=1):
    """
    Simulate the closed-loop system using the ACADOS integrator (model-based simulation).
    Returns (t, simX, simU, simCost, success).
    """
    nx = ocp.model.x.size()[0]  # state dimension (e.g., 14)
    nu = ocp.model.u.size()[0]  # control dimension (e.g., 7)
    
    # Initialize storage for simulation data
    simX = np.zeros((N_sim + 1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    simX[0, :] = x0  # set initial state
    
    success = True
    # Closed-loop simulation
    for i in range(N_sim):
        retries = 0
        success = True
        while retries < nMaxGuess:
            try:
                # Solve MPC for the current state
                u_opt = ocp_solver.solve_for_x0(x0_bar=simX[i, :])
                # Prepare warm-start guess for next step
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
                clear_solver_state(ocp_solver, config.Horizon)
                for j in range(config.Horizon):
                    ocp_solver.set(j, "u", u_guess[:, j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(config.Horizon, "x", x_guess[:, -1])
                
                # Apply control and simulate one step with the model integrator
                simU[i, :] = u_opt
                x_next = integrator.simulate(x=simX[i, :], u=u_opt)
                simX[i+1, :] = x_next
                simCost[i, :] = ocp_solver.get_cost()
                break  # success, exit retry loop
            except Exception as e:
                success = False
                print(f"Error in MPC solve: {e}. Retrying with a new initial guess...")
                print(f"  Step {i}, retry {retries}")
                time.sleep(2)
            retries += 1
            if retries == nMaxGuess - 1:
                # Before final attempt, perturb initial guess (e.g., add 2*pi offset to angles)
                print("Trying alternative initial state guess (e.g., 2*pi offsets).")
                for j in range(0, config.Horizon, 20):
                    ocp_solver.set(j, "x", np.zeros(6))
                ocp_solver.set(0, "x", np.zeros(6))
        if not success:
            print("MPC solve failed after maximum retries.")
            break
    
    # Clear solver state (cleanup)
    clear_solver_state(ocp_solver, config.Horizon)
    # Time vector for each sample
    t = np.linspace(0, N_sim * config.Ts, N_sim + 1)
    return t, simX, simU, simCost, success
