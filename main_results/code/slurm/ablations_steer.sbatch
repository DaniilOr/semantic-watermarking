#!/bin/bash
#SBATCH --job-name=wm_abl_steer
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# "Steer more" ablations (spec Section 8, stage 4; one axis each) on the frozen 60-prompt dev subset, llm_global arm:
#   greedy selection | beta=4 | N=16 | rank-based selection (amendment)
# Each: dry-run budget -> generate -> evaluate (scores under target keys + decoy bank).  Sequential on one GPU.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=${CFG:-configs/pilot.yaml}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
run_one () {
  NAME=$1; OVR=$2
  echo "=== ablation $NAME override=$OVR $(date) ==="
  $PY -m wmworld.cli generate --config $CFG --stage ablation --run-name $NAME --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage ablation --run-name $NAME --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
run_one ablation_greedy '{"selection": "greedy"}'
run_one ablation_beta4  '{"beta": 4.0}'
run_one ablation_N16    '{"candidates": 16}'
run_one ablation_rank   '{"selection": "rank"}'
echo "end=$(date)"
