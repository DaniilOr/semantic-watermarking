#!/bin/bash
#SBATCH --job-name=ob_judge
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Amendment 17: gemma-4-31b judge (preference vs base, pointwise flaws) and the clean-band test.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
ARMS=base,clean_seed1,semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3,chain_seg_push,chain_seg_push_m
MARKED=semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3,chain_seg_push,chain_seg_push_m
$PY scripts/api_judge.py --mode preference --config configs/pilot.yaml --run runs/ob_bench --arms clean_seed1,$MARKED --workers 16
$PY scripts/api_pointwise.py --config configs/pilot.yaml --run runs/ob_bench --arms base,clean_seed1,$MARKED --workers 16
$PY scripts/quality_vs_clean.py --clean runs/ob_bench:base --clean runs/ob_bench:clean_seed1 --marked runs/ob_bench --out runs/quality_vs_clean_ob_bench.json
echo "end=$(date)"
