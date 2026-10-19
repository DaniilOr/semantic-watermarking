#!/bin/bash
#SBATCH --job-name=wm_spans
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# 1024-token stories: every span steered to one key, vs spans rotating through four keys
$PY -m wmworld.cli generate --config $CFG --stage ablation_spans --run-name spans --dry-run || exit 1
$PY -m wmworld.cli generate --config $CFG --stage ablation_spans --run-name spans || exit 1
# whole-text detection, per-span scan (spans are marked individually), and quality
$PY scripts/cross_detect.py --config $CFG --run runs/spans --span 128 --prefixes 256,512,1024
$PY -m wmworld.cli evaluate --config $CFG --run runs/spans --no-decoys
echo "end=$(date)"
