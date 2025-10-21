#!/bin/bash
cd /app
# python main.py

python -m scripts.run_closed_loop_flow_multirun 
# python -m scripts.eval_dataset_closeness
# python plot_mpc_multimodality.py
# python plot_traj.py
chmod -R a+rw /app