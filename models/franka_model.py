"""
franka_model.py

Defines the dynamics model for the Franka Emika Panda robotic arm (7-DOF).
Provides continuous dynamics equations and cost output expressions for use in the ACADOS OCP.
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
    Construct and return a CasADi AcadosModel for the Franka Panda robot.
    State: x = [q (7 joint angles), qdot (7 joint angular velocities)]
    Control: u = [tau (7 joint torques)]
    """
    franka_parser = u2c.URDFparser()
    # Load the URDF model of the robot
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(project_root, "urdf", "panda_arm.urdf")
    franka_parser.from_file(urdf_path)
    
    # Dimensions and constants
    nx = config.Num_State       # 14 state variables (7 positions + 7 velocities)
    nq = config.Num_Q           # 7 joint positions
    nv = config.Num_Velocity    # 7 joint velocities
    nu = config.Num_Input       # 7 control inputs
    gravity = config.gravity
    
    # Define CasADi symbolic variables
    x_sym = ca.SX.sym('x', nx)        # state vector [q; qdot]
    u_sym = ca.SX.sym('u', nu)        # control vector [tau]
    xdot_sym = ca.SX.sym('xdot', nx)  # state derivative [qdot; qddot]
    
    # Split the state for readability
    q = x_sym[:nq]          # joint angles (7x1)
    qdot = x_sym[nq:nq+nv]  # joint angular velocities (7x1)
    tau = u_sym             # joint torques (7x1)
    
    # Get dynamics expressions from URDF (inertia matrix, Coriolis, gravity)
    root_link = config.root
    tip_link = config.tip
    M_sym = franka_parser.get_inertia_matrix_crba(root_link, tip_link)
    C_sym = franka_parser.get_coriolis_rnea(root_link, tip_link)
    G_sym = franka_parser.get_gravity_rnea(root_link, tip_link, gravity)
    
    # Dynamics: M(q)*qddot + C(q,qdot) + G(q) = tau  -> solve for qddot
    M = M_sym(q)
    C = C_sym(q, qdot)
    G = G_sym(q)
    # Add a small regularization to M for numerical stability, then solve for qddot
    qddot = ca.solve(M + 1e-3 * ca.SX.eye(nq), tau - C - G)
    
    # Forward kinematics for end-effector position (for cost output, not part of state)
    fk_dict = franka_parser.get_forward_kinematics(root_link, tip_link)
    T_fk_fun = fk_dict["T_fk"]
    T_fk_expr = T_fk_fun(q)
    p_expr = T_fk_expr[:3, 3]  # end-effector position (3x1)
    
    # Formulate explicit and implicit dynamics
    f_expl = ca.vertcat(qdot, qddot)
    f_impl = xdot_sym - f_expl
    
    # Create and populate the AcadosModel
    model = AcadosModel()
    model.name = "franka_14dof"
    model.x = x_sym
    model.xdot = xdot_sym
    model.u = u_sym
    model.p = []  # no parameters
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    # Cost outputs: include end-effector position and control in stage cost, end-effector position in terminal cost
    model.cost_y_expr = ca.vertcat(p_expr, tau)   # dimension: 3 (position) + 7 (torques) = 10
    model.cost_y_expr_e = ca.vertcat(p_expr)      # dimension: 3 (position only)
    return model
