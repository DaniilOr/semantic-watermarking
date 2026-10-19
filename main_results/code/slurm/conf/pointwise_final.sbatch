#!/bin/bash
#SBATCH --job-name=pointwise_final
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
# the extra clean draws and whatever the sweep extension added, then the test against clean-to-clean variation
$PY scripts/api_pointwise.py --config $CFG --run runs/clean_draws --run runs/tradeoff --workers 8
$PY scripts/quality_vs_clean.py --clean runs/tradeoff:base --clean runs/cross_llama:base --clean runs/clean_draws:base \
    --marked runs/tradeoff --marked runs/cross_llama --out runs/quality_vs_clean.json
echo "end=$(date)"
