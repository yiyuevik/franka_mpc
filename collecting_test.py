import mujoco
import numpy as np
import random
import os
import torch

from mpc_mujoco.collecting_test_model import Cartesian_Collecting_MPC

import matplotlib.pyplot as plt
# Setting
NUM_INI_STATES = 4
CONTROL_STEPS = 200
NUM_SEED = 14

SAMPLING_TIME = 0.001
TARGET_POS = np.array([0.3, 0.3, 0.5])

current_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(current_dir, 'collecting_test', 'collecting_6')
os.makedirs(FOLDER_PATH, exist_ok=True)

def main():
    # initial data generating
    ini_0_states, random_ini_u_guess, ini_data_idx = ini_data_generating()

    # memories for data
    u_ini_memory = np.zeros((1*CONTROL_STEPS, 128, 7))
    x_ini_memory = np.zeros((1*CONTROL_STEPS, 20))
    j_ini_memory = np.zeros((1*CONTROL_STEPS, 1))

    # initial data groups
    initial_data_groups = []
    for n in range(NUM_INI_STATES):
        initial_data_groups.append([random_ini_u_guess[n,:], ini_0_states[n,:], ini_data_idx[n], u_ini_memory, x_ini_memory, j_ini_memory])

    # 单进程执行
    for data_group in initial_data_groups:
        single_ini_process(*data_group)


def ini_data_generating():
    np.random.seed(NUM_SEED)
    random.seed(NUM_SEED)

    # Gaussian noise for initial states generating
    mean_ini = 0
    std_dev_ini = 0.1
    
    # initial states generating
    random_ini_states_list = []
    for i in range(NUM_INI_STATES):
            gaussian_noise_ini_joint_1 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_2 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_3 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_4 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_5 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            gaussian_noise_ini_joint_6 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
            random_ini_states_list.append([gaussian_noise_ini_joint_1,gaussian_noise_ini_joint_2,gaussian_noise_ini_joint_3,gaussian_noise_ini_joint_4,gaussian_noise_ini_joint_5,gaussian_noise_ini_joint_6, 0])
    random_ini_states = np.array(random_ini_states_list) # 50*7

    # random initial u guess
    random_u_guess_list = []
    for i in range(NUM_INI_STATES):
          u_guess_4 = np.round(random.uniform(-2, 2),2)
          u_guess_5 = np.round(random.uniform(-2, 2),2)
          u_guess_7 = np.round(random.uniform(-2, 2),2)
          random_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])
    random_ini_u_guess = np.array(random_u_guess_list) # 50*7

    # data idx
    idx_list = []
    for i in range(NUM_INI_STATES):
          idx = i
          idx_list.append(idx)
    data_idx = np.array(idx_list)

    return random_ini_states, random_ini_u_guess, data_idx



def single_ini_process(initial_guess, initial_state, initial_idx, u_ini_memory, x_ini_memory, j_ini_memory):
    try:
        # panda mujoco
        xml_path = os.path.join(current_dir, 'xml', 'mjx_scene.xml')
        panda = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(panda)
        mpc = Cartesian_Collecting_MPC(panda = panda, data=data)

        # normal simulation
        ini_joint_states, ini_x_states, ini_mpc_cost, ini_joint_inputs, ini_abs_distance, x_collecting_ini, u_collecting_ini, delta_t = mpc.simulate(initial_guess,initial_state,initial_idx)
        print(f'index {initial_idx.item()} initial data control loop finished!!!!!!')
        print(f'x_data size -- {x_collecting_ini.shape}')
        print(f'u_data size -- {u_collecting_ini.shape}')
        print(f'----------------------------------------------------')

        ini_mpc_t_memory = np.array(delta_t).reshape(CONTROL_STEPS, 1)
        ini_mpc_distance_memory = np.array(ini_abs_distance).reshape(CONTROL_STEPS, 1)

        time_mpc_path = os.path.join(FOLDER_PATH, f'time_mpc_' + 'idx-' + str(initial_idx.item()) + '.npy')
        np.save(time_mpc_path, ini_mpc_t_memory)
        distance_mpc_path = os.path.join(FOLDER_PATH, f'distance_mpc_' + 'idx-' + str(initial_idx.item()) + '.npy')
        np.save(distance_mpc_path, ini_mpc_distance_memory)
        print(f'------ solving time & distance trajectory saving finished! ------')

        u_ini_memory = u_collecting_ini
        x_ini_memory = x_collecting_ini.reshape(CONTROL_STEPS,20)
        j_ini_memory = np.array(ini_mpc_cost).reshape(CONTROL_STEPS,1)
        
        # data saving
        torch_u_ini_memory_tensor = torch.Tensor(u_ini_memory)
        torch_x_ini_memory_tensor = torch.Tensor(x_ini_memory)
        torch_j_ini_memory_tensor = torch.Tensor(j_ini_memory)
        
        u_data = torch_u_ini_memory_tensor
        x_data = torch_x_ini_memory_tensor
        j_data = torch_j_ini_memory_tensor

        # save data in PT file for training
        torch.save(u_data, os.path.join(FOLDER_PATH , f'u_data_' + 'idx-' + str(initial_idx.item()) + '_test6.pt'))
        torch.save(x_data, os.path.join(FOLDER_PATH , f'x_data_' + 'idx-' + str(initial_idx.item()) + '_test6.pt'))
        torch.save(j_data, os.path.join(FOLDER_PATH , f'j_data_' + 'idx-' + str(initial_idx.item()) + '_test6.pt'))

        # Plot results
        ts = SAMPLING_TIME
        n = len(ini_mpc_cost)
        t = np.arange(0, n*ts, ts)
        print(f't -- {len(t)}')

        # 3d trajectory plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(ini_x_states[1], ini_x_states[2], ini_x_states[3], label='Trajectory')

        # final & target point
        point = [ini_x_states[1][-1], ini_x_states[2][-1], ini_x_states[3][-1]]
        ax.scatter(point[0], point[1], point[2], color='green', s=10, label='Final Point')
        
        target = TARGET_POS
        ax.scatter(target[0], target[1], target[2], color='red', s=10, label='Target Point')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.legend()

        figure_name = 'idx-' + str(initial_idx.item()) + str(initial_guess) + '_' + str(initial_state) + '_3d' + '.png'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

        # joint space plot (7 joints)
        plt.figure()
        for i in range(7):
            plt.plot(t, ini_joint_states[i+1], label=f"Joint {i+1}")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.legend()

        figure_name = 'idx-' + str(initial_idx.item()) + str(initial_guess)  + '_' + str(initial_state) + '_joints' + '.png'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

        # distance cost plot
        plt.figure()
        plt.plot(t, ini_mpc_cost, color = 'red')
        plt.xlabel("Time [s]")
        plt.ylabel("mpc cost")

        figure_name = 'idx-' + str(initial_idx.item()) + str(initial_guess)  + '_' + str(initial_state) + '_cost' + '.png'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

        # u trajectory plot
        plt.figure()
        for i in range(7):
            plt.plot(t, ini_joint_inputs[i+1], label=f"Joint {i+1}")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint Control Inputs")
        plt.legend()

        figure_name = 'idx-' + str(initial_idx.item()) + str(initial_guess) + '_' + str(initial_state) + '_u' + '.png'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

        # absolute distance plot
        plt.figure()
        plt.plot(t, ini_abs_distance)
        plt.xlabel("Time [s]")
        plt.ylabel("absolute distance [m]")

        figure_name = 'idx-' + str(initial_idx.item()) + str(initial_guess)  + '_' + str(initial_state) + '_dis' + '.png'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)
    except Exception as e:
        print("wrong!")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()