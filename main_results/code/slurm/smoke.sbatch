#!/bin/bash
#SBATCH --job-name=wm_smoke
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/smoke_%j.out
# Smoke stage (spec Section 8, stage 1): selftest -> toy fit-scorer -> 5-arm smoke generation -> toy calibration/evaluation.
set -euo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
export PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=${CFG:-configs/smoke.yaml}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
$PY -m wmworld.cli selftest --config $CFG || echo "SELFTEST REPORTED FAILURES (see runs/selftest.json); continuing so the remaining smoke steps still produce artifacts"
$PY -m wmworld.cli fit-scorer --config $CFG --partition smoke --toy
$PY -m wmworld.cli generate --config $CFG --stage smoke --dry-run
$PY -m wmworld.cli generate --config $CFG --stage smoke
$PY -m wmworld.cli evaluate --config $CFG --run runs/smoke
echo "end=$(date)"
