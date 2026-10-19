#!/bin/bash
#SBATCH --job-name=wm_cross
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
# cross-model steering: generation by the main model, keyed/neutral world by the reference model (pythia-410m)
$PY -m wmworld.cli generate --config $CFG --stage ablation_cross --run-name cross_rw --dry-run || exit 1
$PY -m wmworld.cli generate --config $CFG --stage ablation_cross --run-name cross_rw || exit 1
# detection with the REFERENCE model (its own tokenizer; no bridge) and, for contrast, with the MAIN model
$PY scripts/cross_detect.py --config $CFG --run runs/cross_rw
$PY scripts/t_null_pool.py --config $CFG --run runs/cross_rw
echo "end=$(date)"
