#!/bin/bash
#SBATCH --job-name=fill_quality
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY scripts/quality_vs_clean.py --clean runs/tradeoff:base --clean runs/cross_llama:base --clean runs/clean_draws:base \
    --marked runs/strength_fill --out runs/quality_vs_clean_fill.json
