"""
franka_model.py

Defines the dynamics model for the Franka Emika Panda robotic arm (7-DOF).
Provides continuous dynamics equations and cost output expressions for use in the ACADOS OCP.
"""
import numpy as np
import casadi as ca
from acados_template import AcadosModel
import urdf2casadi.urdfparser as u2c
import config
import os
from utils.helpers import log_SO3_vee, SO3_target_from_log

def export_franka_ode_model():
    """
    Export a simplified 7-DOF kinematic model of Franka Panda for ACADOS:
    State x: joint positions [q] ∈ R^7
    Input u: joint velocities [qdot] ∈ R^7
    Dynamics: q_next = q + dt * qdot
    """
    franka_parser = u2c.URDFparser()
    # Load the URDF model of the robot
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(project_root, "urdf", "panda_arm.urdf")
    franka_parser.from_file(urdf_path)
    
    # Dimensions and constants
    nq = config.Num_State       # 7 joint angles
    nu = config.Num_Input       # 7 control inputs
    
    # Define CasADi symbolic variables
    x_sym = ca.SX.sym('x', nq)        # state vector [q]
    u_sym = ca.SX.sym('u', nu)        # control vector
    xdot_sym = ca.SX.sym('xdot', nq)  # state derivative [qdot]
    p_sym = ca.SX.sym('p', 6)         # end-effector pose (3 position + 3 orientation)

    root_link = config.root
    tip_link = config.tip

    # Forward kinematics for end-effector position (for cost output, not part of state)
    fk_dict = franka_parser.get_forward_kinematics(root_link, tip_link)
    T_fk_fun = fk_dict["T_fk"]
    T_fk_expr = T_fk_fun(x_sym)

    pos = T_fk_expr[:3, 3]  # end-effector position (3x1)
    Rot  = T_fk_expr[:3, :3]

    rot_err = log_SO3_vee(ca.mtimes(SO3_target_from_log(p_sym[3:]).T, Rot))  # SO(3) logarithm map for orientation error

    # Formulate explicit and implicit dynamics
    f_expl = u_sym
    f_impl = xdot_sym - f_expl
    
    # Create and populate the AcadosModel
    model = AcadosModel()
    model.name = "franka_7dof"
    model.x = x_sym
    model.xdot = xdot_sym
    model.u = u_sym
    model.p = p_sym
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    # Cost outputs: include end-effector position and control in stage cost, end-effector position in terminal cost
    model.cost_y_expr = ca.vertcat(pos - p_sym[:3], rot_err, u_sym)   # dimension: (3 position + 3 orientation + 7 control)
    model.cost_y_expr_e = ca.vertcat(pos - p_sym[:3], rot_err)      # dimension:  (3 position + 3 orientation)
    return model
