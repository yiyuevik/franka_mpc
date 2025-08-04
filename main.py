"""
Entry point for the Franka MPC project. This script simply calls the main routine in `scripts/run_closed_loop.py`.
"""
from scripts.run_closed_loop import main as closed_loop_main
from scripts.collect_multimodal_trajectories import main as collect_trajectories
if __name__ == "__main__":
    # collect_trajectories()
    closed_loop_main()