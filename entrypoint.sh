#!/bin/bash
cd /app
python main.py
# python -m utils.plot_trajs_clusters
# python -m utils.plot_mpc_multimodality
# python -m scripts.plot_traj
# python -m scripts.run_closed_loop_flow_multirun 
# python -m scripts.eval_dataset_closeness
# python plot_mpc_multimodality.py
chmod -R a+rw /app