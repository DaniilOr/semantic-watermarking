#!/bin/bash
#SBATCH --job-name=ob_tail
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Amendment 17 tail (CPU part): attacks, then the judge.  Detection runs as a separate GPU job after the attacks.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
ARMS=base,clean_seed1,semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3
MARKED=semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3
$PY scripts/reorder_attack.py --run runs/ob_bench --arms $ARMS --attacks reorder_sent,resegment
$PY scripts/api_paraphrase.py --run runs/ob_bench --arms $ARMS --level light --workers 16
$PY scripts/api_paraphrase.py --run runs/ob_bench --arms $ARMS --level strong --workers 16
echo "attacks done $(date)"
$PY scripts/api_judge.py --mode preference --config configs/pilot.yaml --run runs/ob_bench --arms clean_seed1,$MARKED --workers 16
$PY scripts/api_pointwise.py --config configs/pilot.yaml --run runs/ob_bench --arms base,clean_seed1,$MARKED --workers 16
$PY scripts/quality_vs_clean.py --clean runs/ob_bench:base --clean runs/ob_bench:clean_seed1 --marked runs/ob_bench --out runs/quality_vs_clean_ob_bench.json
echo "end=$(date)"
