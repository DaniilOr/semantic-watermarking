#!/bin/bash
#SBATCH --job-name=baseline_judge
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
for R in baseline_kgw baseline_kgw_selfhash baseline_semstamp; do
  $PY scripts/api_judge.py --mode preference --config $CFG --run runs/$R --workers 8
  $PY scripts/api_pointwise.py --config $CFG --run runs/$R --workers 8
done
echo "##### flaws vs the SAME held-out clean band as our arms #####"
$PY scripts/quality_vs_clean.py --clean runs/conf_cross:base --clean runs/conf_pos:base \
    --marked runs/baseline_kgw --marked runs/baseline_kgw_selfhash --marked runs/baseline_semstamp \
    --marked runs/conf_cross --marked runs/conf_window_reg --out runs/quality_vs_clean_baselines.json
echo "end=$(date)"
