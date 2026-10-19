#!/bin/bash
#SBATCH --job-name=wm_fit_spec
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Amendment: sigma entries for the decoy-relative objective 'specific', reusing the scale_fit pools (decoy + neutral scorings only).
set -euo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=${CFG:-configs/pilot.yaml}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
$PY -m wmworld.cli fit-scorer --config $CFG --partition scale_fit --objective specific
echo "end=$(date)"
