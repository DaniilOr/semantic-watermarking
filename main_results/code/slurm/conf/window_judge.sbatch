#!/bin/bash
#SBATCH --job-name=window_judge
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/window_dev --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/window_dev --workers 8
$PY scripts/quality_vs_clean.py --clean runs/tradeoff:base --clean runs/cross_llama:base --clean runs/clean_draws:base \
    --marked runs/window_dev --marked runs/strength_fill --marked runs/keylen10 --out runs/quality_vs_clean_window.json
