#!/bin/bash
#SBATCH --job-name=wm_smoke_steer
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Small-sample "steer more" check on the standard objective (smoke setting: 12 prompts, 64 tokens):
#   greedy selection (N=4) | N=8 softmax beta=2 | N=8 greedy   -> each: base + llm_global, evaluated with both detectors
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/smoke.yaml
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
run_one () {
  NAME=$1; OVR=$2
  echo "=== variant $NAME override=$OVR $(date) ==="
  $PY -m wmworld.cli generate --config $CFG --stage smoke --run-name $NAME --arms base,llm_global --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage smoke --run-name $NAME --arms base,llm_global --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
run_one smoke_greedy    '{"selection": "greedy"}'
run_one smoke_N8        '{"candidates": 8}'
run_one smoke_N8_greedy '{"candidates": 8, "selection": "greedy"}'
echo "end=$(date)"
