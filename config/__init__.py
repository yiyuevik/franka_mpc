import os
import yaml
import numpy as np

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "mpc_config.yaml")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Assign configuration parameters (with defaults if keys are missing)
Horizon = cfg.get("Horizon", 20)
Ts = cfg.get("Ts", 0.1)
Num_State = cfg.get("Num_State", 7)
Num_P = cfg.get("Num_P", 6)
Num_Input = cfg.get("Num_Input", 7)
root = cfg.get("root", "panda_link0")
tip = cfg.get("tip", "panda_link8")
tau_max = cfg.get("tau_max", 100.0)

# Weight matrices for cost function
Q_pos = np.array(cfg.get("Q_pos", np.eye(3) * 10.0))    # position error weight
Q_rot = np.array(cfg.get("Q_rot", np.eye(3) * 5.0))  # orientation error weight
R = np.array(cfg.get("R", np.eye(7) * 0.01))

# Initial guess randomization range
initial_guess_min = cfg.get("initial_guess_min", -20.0)
initial_guess_max = cfg.get("initial_guess_max", 20.0)

# Target references for cost (desired end-effector position and target joint torques)
target_position = np.array(cfg.get("target_position", [0.3, 0.3, 0.5]))

# Initial guess grid
U4_MIN = cfg.get("U4_MIN", -20.0)
U4_MAX = cfg.get("U4_MAX", 20.0)
U5_MIN = cfg.get("U5_MIN", -20.0)
U5_MAX = cfg.get("U5_MAX", 20.0)
U7_MIN = cfg.get("U7_MIN", -20.0)
U7_MAX = cfg.get("U7_MAX", 20.0)
STEP = cfg.get("STEP", 5)