#!/bin/bash
#SBATCH --job-name=ob_attacks
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Amendment 17: attacks on the benchmark run (reorder/resegment local; paraphrases via gemma-4-31b).
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
ARMS=base,clean_seed1,semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3,chain_seg_push,chain_seg_push_m
MARKED=semstamp,chain_reference,chain_grpo300,chain_grpo400,chain_rft3,chain_seg_push,chain_seg_push_m
$PY scripts/reorder_attack.py --run runs/ob_bench --arms $ARMS --attacks reorder_sent,resegment
$PY scripts/api_paraphrase.py --run runs/ob_bench --arms $ARMS --level light --workers 16
$PY scripts/api_paraphrase.py --run runs/ob_bench --arms $ARMS --level strong --workers 16
echo "end=$(date)"
