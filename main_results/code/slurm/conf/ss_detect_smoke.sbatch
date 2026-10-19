#!/bin/bash
#SBATCH --job-name=ss_detect_smoke
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY scripts/semstamp_detect.py --marked runs/ss_smoke --null runs/ss_smoke --null runs/ss_llama3b --device cpu --alt-keys 2 --out runs/ss_smoke/detect.json
