#!/bin/bash
#SBATCH --job-name=wm_judge_dev
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
echo "host=$(hostname) start=$(date)"
# primary judge (another model family than generator and reference); cross_long holds the known-damaged arm
$PY scripts/api_judge.py --config configs/pilot.yaml --run runs/cross_llama --run runs/cross_long --run runs/cross_long2 --workers 8
# second opinion from an unrelated judge on one run: do two judges agree?
$PY scripts/api_judge.py --config configs/pilot.yaml --run runs/cross_llama --model qwen/qwen3.8-27b --workers 8
echo "end=$(date)"
