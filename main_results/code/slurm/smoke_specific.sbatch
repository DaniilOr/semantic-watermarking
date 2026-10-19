#!/bin/bash
#SBATCH --job-name=wm_smoke_spec
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Small-sample check of the decoy-relative objective (amendment) in the smoke setting (12 prompts, 64 tokens, N=4):
#   toy 'specific' sigma on the smoke pools -> base + llm_global(specific, centred) -> llm_global(specific, greedy) -> evaluate both
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/smoke.yaml
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
$PY -m wmworld.cli fit-scorer --config $CFG --partition smoke --toy --objective specific || exit 1
$PY -m wmworld.cli generate --config $CFG --stage smoke --run-name smoke_specific --arms base,llm_global --override '{"objective": "specific", "center_gains": true}' --dry-run
$PY -m wmworld.cli generate --config $CFG --stage smoke --run-name smoke_specific --arms base,llm_global --override '{"objective": "specific", "center_gains": true}' || exit 1
$PY -m wmworld.cli evaluate --config $CFG --run runs/smoke_specific || exit 1
$PY -m wmworld.cli generate --config $CFG --stage smoke --run-name smoke_specific_greedy --arms base,llm_global --override '{"objective": "specific", "selection": "greedy"}' --dry-run
$PY -m wmworld.cli generate --config $CFG --stage smoke --run-name smoke_specific_greedy --arms base,llm_global --override '{"objective": "specific", "selection": "greedy"}' || exit 1
$PY -m wmworld.cli evaluate --config $CFG --run runs/smoke_specific_greedy || exit 1
echo "end=$(date)"
