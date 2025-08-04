#!/bin/bash
cd /app
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv
python main.py
chmod -R 777 /app/output