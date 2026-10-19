#!/bin/bash
#SBATCH --job-name=ss_judge_r2
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Round-2 arms (amendment 6) judged ahead of the full tail; the later ss_judge run resumes and skips these.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
ARMS=segA_b2,segA_b3,segB_b2,bestof4_seg,bestof4B_b1
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/ss_llama3b --arms $ARMS --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/ss_llama3b --arms $ARMS --workers 8
$PY scripts/quality_vs_clean.py --clean runs/ss_llama3b:base --clean runs/ss_llama3b:clean_seed1 --marked runs/ss_llama3b --out runs/quality_vs_clean_ss_llama3b_r2.json
$PY scripts/fact_probe.py --out runs/fact_probe_ss_llama3b_r2.json
echo "end=$(date)"
