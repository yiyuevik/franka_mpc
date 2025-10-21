import os
import yaml
import numpy as np

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "mpc_config.yaml")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Assign configuration parameters (with defaults if keys are missing)
x0 = cfg.get("x0", np.array([0, -0.25*np.pi, 0, -0.75*np.pi, 0, 0.5*np.pi, 0.25*np.pi], dtype=float))
Horizon = cfg.get("Horizon", 20)
Ts = cfg.get("Ts", 0.1)
Num_State = cfg.get("Num_State", 7)
Num_P = cfg.get("Num_P", 6)
Num_Input = cfg.get("Num_Input", 7)
root = cfg.get("root", "panda_link0")
tip = cfg.get("tip", "panda_link8")
tau_max = cfg.get("tau_max", 100.0)

# Parallel processing parameters
Num_Group = cfg.get("Num_Group", 4)

# Noisy data collection parameters
Num_Noisy_Data = cfg.get("Num_Noisy_Data", 15)
Noise_Mean = cfg.get("Noise_Mean", 0.0)
Noise_Std = cfg.get("Noise_Std", 0.05)

# Weight matrices for cost function
Q_pos = np.array(cfg.get("Q_pos", np.eye(3) * 10.0))    # position error weight
Q_rot = np.array(cfg.get("Q_rot", np.eye(3) * 5.0))  # orientation error weight
R = np.array(cfg.get("R", np.eye(7) * 0.01))
P_pos = np.array(cfg.get("P_pos", np.eye(3) * 10.0))    # terminal position error weight
P_rot = np.array(cfg.get("P_rot", np.eye(3) * 5.0))  # terminal orientation error weight

# Initial guess randomization range
initial_guess_min = cfg.get("initial_guess_min", -20.0)
initial_guess_max = cfg.get("initial_guess_max", 20.0)

# Target references for cost (desired end-effector position and target joint torques)
target_position = np.array(cfg.get("target_position", [0.5068906, 0.2, 0.5902821]))

# Initial guess grid
U3_MIN = cfg.get("U3_MIN", -20.0)
U3_MAX = cfg.get("U3_MAX", 20.0)
U5_MIN = cfg.get("U5_MIN", -20.0)
U5_MAX = cfg.get("U5_MAX", 20.0)
U6_MIN = cfg.get("U6_MIN", -20.0)
U6_MAX = cfg.get("U6_MAX", 20.0)
STEP = cfg.get("STEP", 5)

# Obstacle avoidance configuration
Obstacle_Avoidance = cfg.get("Obstacle_Avoidance", True)
Obstacle_Position = np.array(cfg.get("Obstacle_Position", [0.2566, 0.2671, 0.5460]))
Obstacle_Scale = np.array(cfg.get("Obstacle_Scale", [31.25, 31.25, 31.25]))
