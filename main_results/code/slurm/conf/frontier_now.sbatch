#!/bin/bash
#SBATCH --job-name=frontier_now
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# the four frontier arms, both measures, on the texts that exist right now (everything already judged is skipped)
$PY scripts/api_judge.py --config $CFG --run runs/tradeoff --arms wm_c4g,wm_c6,wm_c6g,wm_c8g --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/tradeoff --arms wm_c4g,wm_c6,wm_c6g,wm_c8g --workers 8
echo "end=$(date)"
