#!/bin/bash
#SBATCH --job-name=wm_code100
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
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname)"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 4)" 2>/dev/null || { echo "CUDA PREFLIGHT FAILED on $(hostname): no usable GPU (driver error); exclude this node"; exit 4; }
CFG=configs/pilot.yaml
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
# code: 400 MBPP tasks x (base + wm_neutral + wm_decoy), one shard of NUM_SHARDS
$PY -m wmworld.cli generate --config $CFG --stage code100 --run-name code100 --dry-run || exit 1
$PY -m wmworld.cli generate --config $CFG --stage code100 --run-name code100 --shard ${SHARD:-0} --num-shards ${NUM_SHARDS:-2} || exit 1
echo "end=$(date)"
$PY scripts/t_null_pool.py --config $CFG --run runs/code100
