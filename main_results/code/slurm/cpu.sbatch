#!/bin/bash
#SBATCH --job-name=wm_cpu
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Generic CPU job for analysis scripts, tests, data preparation, key building.  Usage:
#   sbatch --job-name=wm_tests --export=ALL,CMD="python -m pytest -q tests" slurm/cpu.sbatch
#   sbatch --job-name=wm_diag  --export=ALL,CMD="python scripts/traj_diag.py runs/dev" slurm/cpu.sbatch
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PATH=/home/daniil.orel/anaconda3/envs/wmworld/bin:$PATH
CMD=${CMD:?set CMD}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date) cmd=$CMD"
bash -c "$CMD"
echo "exit=$? end=$(date)"
