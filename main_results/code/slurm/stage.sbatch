#!/bin/bash
#SBATCH --job-name=wm_stage
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Generic one-GPU shard of a generation stage.  Usage:
#   sbatch --export=ALL,STAGE=dev,SHARD=0,NUM_SHARDS=4,CFG=configs/pilot.yaml --job-name=wm_dev0 slurm/stage.sbatch
# The stage must have a budget manifest (run `generate --stage $STAGE --dry-run` first).
set -euo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
export PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 4)" 2>/dev/null || { echo "CUDA PREFLIGHT FAILED on $(hostname): no usable GPU (driver error); exclude this node"; exit 4; }
CFG=${CFG:-configs/pilot.yaml}
STAGE=${STAGE:?set STAGE}
SHARD=${SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
EXTRA=${EXTRA:-}
echo "host=$(hostname) job=$SLURM_JOB_ID stage=$STAGE shard=$SHARD/$NUM_SHARDS start=$(date)"
$PY -m wmworld.cli generate --config $CFG --stage $STAGE --shard $SHARD --num-shards $NUM_SHARDS $EXTRA
echo "end=$(date)"
