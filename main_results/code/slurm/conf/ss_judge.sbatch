#!/bin/bash
#SBATCH --job-name=ss_judge
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/ss_llama3b --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/ss_llama3b --workers 8
$PY scripts/quality_vs_clean.py --clean runs/ss_llama3b:base --clean runs/ss_llama3b:clean_seed1 --marked runs/ss_llama3b --out runs/quality_vs_clean_ss_llama3b.json
echo "end=$(date)"
