#!/bin/bash
#SBATCH --job-name=ss_detect
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
read -r BETA W EMB PFX < <($PY -c "import json; c=json.load(open(\"assets/ss_choice.json\")); print(c[\"beta\"], c[\"W\"], c[\"embedder\"], c.get(\"embedder_prefix\") or \"-\")")
[ "$PFX" = "-" ] && PFX=""
M=""; for d in runs/ss_llama3b runs/ss_llama3b_reorder_sent runs/ss_llama3b_resegment runs/ss_llama3b_para_light runs/ss_llama3b_para_strong; do [ -d $d ] && M="$M --marked $d"; done
$PY scripts/semstamp_detect.py --null runs/null_calibration --null runs/ss_llama3b_null $M --W $W --embedder $EMB --embedder-prefix="$PFX" --device cuda --alt-keys 6 --out runs/detect_ss_llama3b.json
echo "end=$(date)"
