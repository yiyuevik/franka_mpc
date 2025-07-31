import os
import yaml
import numpy as np

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "mpc_config.yaml")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Assign configuration parameters (with defaults if keys are missing)
Horizon = cfg.get("Horizon", 128)
Ts = cfg.get("Ts", 0.001)
Num_State = cfg.get("Num_State", 14)
Num_Q = cfg.get("Num_Q", 7)
Num_Velocity = cfg.get("Num_Velocity", 7)
Num_P = cfg.get("Num_P", 3)
Num_Input = cfg.get("Num_Input", 7)
gravity = cfg.get("gravity", [0.0, 0.0, -9.81])
root = cfg.get("root", "panda_link0")
tip = cfg.get("tip", "panda_link8")
tau_max = cfg.get("tau_max", 100.0)

# Noisy data collection parameters
Num_Noisy_Data = cfg.get("Num_Noisy_Data", 15)
Noise_Mean = cfg.get("Noise_Mean", 0.0)
Noise_Std = cfg.get("Noise_Std", 0.05)

# Weight matrices for cost function
Q = np.array(cfg.get("Q", np.eye(3) * 1000.0))
R = np.array(cfg.get("R", np.eye(7) * 0.01))
P = np.array(cfg.get("P", Q))  # terminal cost weight (defaults to Q if not provided)

# Initial guess randomization range
initial_guess_min = cfg.get("initial_guess_min", -20.0)
initial_guess_max = cfg.get("initial_guess_max", 20.0)

# Target references for cost (desired end-effector position and target joint torques)
target_position = np.array(cfg.get("target_position", [0.3, 0.3, 0.5]))
target_torque = np.array(cfg.get("target_torque", [0.0] * 7))

# Initial guess grid
U4_MIN = cfg.get("U4_MIN", -20.0)
U4_MAX = cfg.get("U4_MAX", 20.0)
U5_MIN = cfg.get("U5_MIN", -20.0)
U5_MAX = cfg.get("U5_MAX", 20.0)
U7_MIN = cfg.get("U7_MIN", -20.0)
U7_MAX = cfg.get("U7_MAX", 20.0)
STEP = cfg.get("STEP", 2.5)

# heuristic guess range
U4_MIN_HEURISTIC = cfg.get("U4_MIN_HEURISTIC", -11.0)
U4_MAX_HEURISTIC = cfg.get("U4_MAX_HEURISTIC", 20.0)
U5_MIN_HEURISTIC = cfg.get("U5_MIN_HEURISTIC", -14.0)
U5_MAX_HEURISTIC = cfg.get("U5_MAX_HEURISTIC", 12.0)
U7_MIN_HEURISTIC = cfg.get("U7_MIN_HEURISTIC", -12.0)
U7_MAX_HEURISTIC = cfg.get("U7_MAX_HEURISTIC", 13.0)