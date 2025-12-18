"""
Entry point for the Franka MPC project. This script simply calls the main routine in `scripts/run_closed_loop.py`.
"""
from scripts import (
    run_closed_loop,
    collect_multimodal_trajectories,
    mpc_ipopt_franka,
    calculate_axis_importance,
    data_collect,
    smoke_env,
    generate_dataset,
    train,
    run_closed_loop_flow,
    compare_real_vs_sim,
    analyze_modality,
)

if __name__ == "__main__":
    # collect_multimodal_trajectories()
    # run_closed_loop()
    # calculate_axis_importance()
    data_collect()
    # mpc_ipopt_franka()
    # smoke_env()
    # generate_dataset()
    # train()
    # run_closed_loop_flow()
    # compare_real_vs_sim()
    # analyze_modality()