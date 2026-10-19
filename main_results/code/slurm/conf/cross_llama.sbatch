#!/bin/bash
#SBATCH --job-name=wm_xllama
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# stronger cross-model push: the vocabulary bridge attenuates the nominal beta (beta 10 cost only 0.02 nats/token)
$PY -m wmworld.cli generate --config $CFG --stage ablation_cross_llama --run-name cross_llama --dry-run || exit 1
$PY -m wmworld.cli generate --config $CFG --stage ablation_cross_llama --run-name cross_llama || exit 1
$PY scripts/cross_detect.py --config $CFG --run runs/cross_llama --decoys 40
$PY -m wmworld.cli evaluate --config $CFG --run runs/cross_llama --no-decoys
echo "end=$(date)"
