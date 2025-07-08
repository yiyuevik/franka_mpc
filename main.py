"""
Entry point for the Franka MPC project. This script simply calls the main routine in `scripts/run_closed_loop.py`.
"""
from scripts.run_closed_loop import main as run_main

if __name__ == "__main__":
    run_main()