import numpy as np
import casadi as ca
import do_mpc
import mujoco
import time
SAMPLING_TIME = 0.001
CONTROL_RATE = 10
HORIZON = 128
SUM_CTL_STEPS = 200

# CONTROLLER_SAMPLE_TIME = 0.01
FIXED_TARGET = np.array([[0.3], [0.3], [0.5]])
TARGET_POS = np.array([0.3, 0.3, 0.5]).reshape(3, 1)

Q = np.diag([10,10,10])
R = 0.5
P = np.diag([10,10,10])


class Cartesian_Collecting_MPC:

    def __init__(self, panda, data):
        self.panda = panda
        self.data = data
        mujoco.mj_step(self.panda, data)
        self.model = self.create_model()
        self.mpc = self.create_mpc(self.model)



    def get_inertia_matrix(self):
        nv = self.data.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.data.model, M, self.data.qM)
        return M[:7, :7]

    def get_coriolis(self):
        return self.data.qfrc_bias

    def get_gravity_forces(self):
        return self.data.qfrc_gravcomp
    
    # Jacobian Matrix
    def compute_jacobian(self, model, data, tpoint):
        """Compute the Jacobian for the end-effector."""
        # Jacobian matrices for position (jacp) and orientation (jacr)
        jacp = np.zeros((3, model.nv))  # Position Jacobian
        jacr = np.zeros((3, model.nv))  # Orientation Jacobian
        
        body_id = 9

        # mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        mujoco.mj_jac(model, data, jacp, jacr, tpoint, body_id)
        
        return jacp, jacr
    

    def create_model(self):
        model_type = "continuous"
        model = do_mpc.model.Model(model_type)

        # Define the states (joint positions and velocities)
        q = model.set_variable(var_type="_x", var_name="q", shape=(7, 1))
        q_dot = model.set_variable(var_type="_x", var_name="q_dot", shape=(7, 1))

        x = model.set_variable(var_type="_x", var_name="x", shape=(3, 1))
        x_dot = model.set_variable(var_type="_x", var_name="x_dot", shape=(3, 1))

        # R = model.set_variable(var_type='_u', var_name='R')

        # Position Jacobian
        jacp, _ = self.compute_jacobian(self.panda, self.data, TARGET_POS)
        jacp = jacp[:, :7]

        # Define the state target variable
        target_x_states = model.set_variable(var_type="_tvp", var_name="target_x_states", shape=(3, 1))

        # Define the control inputs (joint torques)
        tau = model.set_variable(var_type="_u", var_name="tau", shape=(7, 1))

        M = self.get_inertia_matrix()
        C = self.get_coriolis()[:7].reshape(1, 7)
        G = self.get_gravity_forces()[:7]

        q_ddot = ca.mtimes(ca.inv(M), (tau - ca.mtimes(C, q_dot) - G))

        model.set_rhs("x",x_dot)
        model.set_rhs("x_dot", jacp@q_dot)

        model.set_rhs("q", q_dot)
        model.set_rhs("q_dot", q_ddot)

        model.setup()
        return model

    def create_mpc(self, model):
        mpc = do_mpc.controller.MPC(model)
        n_horizon = HORIZON 
        t_step = SAMPLING_TIME

        setup_mpc = {
            "n_horizon": n_horizon,
            "t_step": t_step,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 3,
            "collocation_ni": 2,
            "store_full_solution": True,
            "nlpsol_opts": {'ipopt.max_iter': 10, 'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        }
        mpc.set_param(**setup_mpc)

        # target position

        mterm = P[0,0]*(model.x["x"][0]-model.tvp["target_x_states"][0])**2 + P[1,1]*(model.x["x"][1]-model.tvp["target_x_states"][1])**2 + P[2,2]*(model.x["x"][2]-model.tvp["target_x_states"][2])**2
        lterm = Q[0,0]*(model.x["x"][0]-model.tvp["target_x_states"][0])**2 + Q[1,1]*(model.x["x"][1]-model.tvp["target_x_states"][1])**2 + Q[2,2]*(model.x["x"][2]-model.tvp["target_x_states"][2])**2




        tvp_template = mpc.get_tvp_template()

        def tvp_fixed_fun(t_now):
            # Define a fixed target x position
            fixed_target = FIXED_TARGET

            # Set the same target for the whole prediction horizon
            for k in range(n_horizon + 1):
                tvp_template["_tvp", k, "target_x_states"] = fixed_target

            return tvp_template

        mpc.set_tvp_fun(tvp_fixed_fun)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(tau=R)  # Regularization term for control inputs

        # Define constraints
        mpc.bounds["lower", "_x", "q"] = -np.pi
        mpc.bounds["upper", "_x", "q"] = np.pi


        mpc.setup()
        return mpc
    
    def mpc_cost(self, predicted_states, predicted_controls, Q, R, P):
        cost = 0

        # initial cost
        x_0 = predicted_states[:,0,0]
        cost = Q[0,0]*(x_0[0]-TARGET_POS[0])**2 + Q[1,1]*(x_0[1]-TARGET_POS[1])**2 + Q[2,2]*(x_0[2]-TARGET_POS[2])**2

        # stage cost
        i = 0
        for i in range(predicted_controls.shape[1]-1):
            x_i = predicted_states[:,i+1,0]
            u_i_1 = predicted_controls[:,i,0]
            u_i = predicted_controls[:,i+1,0]
            delta_u = u_i - u_i_1
            # print(0.5*(ca.sumsqr(u_i)))
            cost += Q[0,0]*(x_i[0]-TARGET_POS[0])**2 + Q[1,1]*(x_i[1]-TARGET_POS[1])**2 + Q[2,2]*(x_i[2]-TARGET_POS[2])**2 + R*(ca.sumsqr(delta_u))

        # terminal cost
        x_end = predicted_states[:,-1,0]
        cost += P[0,0]*(x_end[0]-TARGET_POS[0])**2 + P[1,1]*(x_end[1]-TARGET_POS[1])**2 + P[2,2]*(x_end[2]-TARGET_POS[2])**2

        return cost

    def _simulate_mpc_mujoco(self, mpc, panda, data, u_initial_guess, initial_idx):
        control_steps = SUM_CTL_STEPS
        horizon = HORIZON

        # data collecting for 1 setting
        u_collecting_1_setting = np.zeros([control_steps,horizon,7])
        x0_collecting_1_setting = np.zeros([control_steps,20])

        x0 = np.zeros((20, 1))
        mpc.x0 = x0
        mpc.u0 = ca.DM(u_initial_guess)
        mpc.set_initial_guess()

        joint_states = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        joint_inputs = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        x_states = {1: [], 2: [], 3: []}
        mpc_cost = []
        abs_distance = []
        delta_t_list = []

        current_step = 0

        # control update step
        max_steps = SUM_CTL_STEPS*CONTROL_RATE
        control_step = 0

        while current_step < max_steps:

            if current_step % CONTROL_RATE == 0:
                end_position = self.data.body("hand").xpos.copy()
                print(f'-------------------------------------------------------------------------')
                print(f'initial_idx, current_step, end_position -- {initial_idx, current_step, end_position}')
                print(f'-------------------------------------------------------------------------')
                distance = np.linalg.norm(end_position.reshape(3, 1) - TARGET_POS)
                print(f'distance -- {distance}')
                abs_distance.append(distance)

                for i in range(3):
                    x_states[i + 1].append(end_position[i])

                # Position Jacobian
                jacp, _ = self.compute_jacobian(self.panda, self.data, TARGET_POS)
                jacp = jacp[:, :7]

                q_current = np.array(data.qpos).reshape(-1, 1)
                q_dot_current = np.array(data.qvel).reshape(-1, 1)
                x_current = np.array(data.xpos[9,:]).reshape(-1, 1)
                x0[:7] = q_current[:7]
                x0[7:14] = q_dot_current[:7]
                x0[14:17] = x_current
                x0[17:20] = ca.mtimes(jacp, q_dot_current[:7])

                # x0 data collecting
                x0_collecting_1_setting[control_step,:] = x0.reshape(1,20)

                for i in range(7):
                    joint_states[i + 1].append(q_current[i])

                start_time = time.time()
                u0 = mpc.make_step(x0)
                end_time = time.time()
                delta_t = end_time - start_time
                print(f'mpc_solving time -- {delta_t}')
                delta_t_list.append(delta_t)

                data.ctrl[:7] = u0.flatten()

                predicted_states = mpc.data.prediction(('_x', 'x'))
                predicted_controls = mpc.data.prediction(('_u', 'tau'))

                applied_inputs = predicted_controls[:,0,0].reshape(-1, 1)
                for i in range(7):
                    joint_inputs[i + 1].append(applied_inputs[i])

                # u data collecting
                u_collecting_array = predicted_controls.transpose(2,1,0)
                u_collecting_1_setting[control_step,:,:] = u_collecting_array

                # calculate mpc cost
                cost = self.mpc_cost(predicted_states, predicted_controls, Q, R, P)
                print(f'cost -- {cost}')
                cost = cost.toarray().reshape(-1)
                mpc_cost.append(cost)
                control_step = control_step + 1
                
            current_step += 1

            mujoco.mj_step(panda, data)

        return joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x0_collecting_1_setting, u_collecting_1_setting, delta_t_list




    def _simulate_single_step(self, mpc, panda, data, u_guess, initial_idx, ctl_step, n):
        control_steps = 1
        horizon = HORIZON

        # data collecting for 1 setting
        u_collecting_1_step = np.zeros([control_steps,horizon,7])
        x_collecting_1_step = np.zeros([control_steps,20])

        x0 = np.zeros((20, 1))
        mpc.x0 = x0
        mpc.u0 = ca.DM(u_guess)
        mpc.set_initial_guess()

        joint_states = np.zeros([1,7])
        joint_inputs = np.zeros([1,7])
        x_states = {1: [], 2: [], 3: []}
        mpc_cost = np.zeros([1,1])
        abs_distance = np.zeros([1,1])

        current_step = 0

        # control update step
        max_steps = SUM_CTL_STEPS*CONTROL_RATE
        control_step = 0

        end_position = self.data.body("hand").xpos.copy()
        print(f'-------------------------------------------------------------------------')
        print(f'noisy: initial_idx, ctl_step, n, end_position -- {initial_idx, ctl_step, n, end_position}')
        print(f'-------------------------------------------------------------------------')
        distance = np.linalg.norm(end_position.reshape(3, 1) - TARGET_POS)
        print(f'distance -- {distance}')
        abs_distance = distance

        for i in range(3):
            x_states[i + 1].append(end_position[i])

        # Position Jacobian
        jacp, _ = self.compute_jacobian(self.panda, self.data, TARGET_POS)
        jacp = jacp[:, :7]

        q_current = np.array(data.qpos).reshape(-1, 1)
        q_dot_current = np.array(data.qvel).reshape(-1, 1)
        x_current = np.array(data.xpos[9,:]).reshape(-1, 1)
        x0[:7] = q_current[:7]
        x0[7:14] = q_dot_current[:7]
        x0[14:17] = x_current
        x0[17:20] = ca.mtimes(jacp, q_dot_current[:7])

        # x0 data collecting
        x_collecting_1_step = x0

        joint_states[0,:] = q_current[:7].reshape(7)

        start_time = time.time()
        u0 = mpc.make_step(x0)
        end_time = time.time()
        delta_t = end_time - start_time
        print(f'mpc_solving time -- {delta_t}')

        data.ctrl[:7] = u0.flatten()

        predicted_states = mpc.data.prediction(('_x', 'x'))
        predicted_controls = mpc.data.prediction(('_u', 'tau'))

        applied_inputs = predicted_controls[:,0,0].reshape(-1, 1)
        joint_inputs[0,:] = applied_inputs[:7].reshape(7)

        # u data collecting
        u_collecting_array = predicted_controls.transpose(2,1,0)
        u_collecting_1_step[control_step,:,:] = u_collecting_array

        # calculate mpc cost
        cost = self.mpc_cost(predicted_states, predicted_controls, Q, R, P)
        print(f'cost -- {cost}')
        cost = cost.toarray().reshape(-1)
        mpc_cost = cost.item()
        control_step = control_step + 1
                
        current_step += 1

        mujoco.mj_step(panda, data)
        
        # new q state
        updated_joint_states = np.zeros(7)
        updated_q = np.array(data.qpos).reshape(-1, 1)

        for i in range(7):
            updated_joint_states[i] = (updated_q[i])       

        return joint_states, x_states, mpc_cost, joint_inputs, abs_distance, x_collecting_1_step, u_collecting_1_step, updated_joint_states


    def simulate(self,u_initial_guess,initial_state, initial_idx):
        self.data.qpos[:7] = initial_state
        mujoco.mj_step(self.panda, self.data)
        return self._simulate_mpc_mujoco(self.mpc, self.data.model, self.data, u_initial_guess, initial_idx)
    

    
    def single_simulate(self,u_guess,noisy_state, initial_idx, ctl_step, n):
        self.data.qpos[:7] = noisy_state
        mujoco.mj_step(self.panda, self.data)
        return self._simulate_single_step(self.mpc, self.data.model, self.data, u_guess, initial_idx, ctl_step, n)
   