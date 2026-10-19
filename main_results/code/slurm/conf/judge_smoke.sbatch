#!/bin/bash
#SBATCH --job-name=wm_judge_smoke
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
echo "host=$(hostname)"
# can a compute node reach the API at all?
$PY -c "import requests; r=requests.get('https://openrouter.ai/api/v1/models', timeout=20); print('network ok:', r.status_code)" || { echo "NO NETWORK FROM COMPUTE NODE"; exit 3; }
# 3 prompts x 2 arms x 2 orders = 12 comparisons, including the known-damaged arm (controller off)
$PY scripts/api_judge.py --config configs/pilot.yaml --run runs/cross_long --arms wm_cross30,wm_cross30f --limit 3 --workers 4
