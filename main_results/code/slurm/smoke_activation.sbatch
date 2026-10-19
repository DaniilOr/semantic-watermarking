#!/bin/bash
#SBATCH --job-name=wm_smoke_act
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Extension, small sample first (smoke: 12 prompts, 64 tokens): activation steering toward the keyed world.
#   alpha=0.25 on all layers already collapses the model into repeating the key word, so this sweep is small:
#   all layers alpha 0.02 / 0.05 / 0.1, and middle layers (8-15) alpha 0.1 / 0.3; N=1 arm activation_steer, base as control.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/smoke.yaml
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
run_one () {
  NAME=$1; ARMS=$2; OVR=$3
  echo "=== variant $NAME arms=$ARMS override=$OVR $(date) ==="
  $PY -m wmworld.cli generate --config $CFG --stage smoke --run-name $NAME --arms $ARMS --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage smoke --run-name $NAME --arms $ARMS --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
run_one smoke_act002    base,activation_steer '{"steer_alpha": 0.02}'
run_one smoke_act005    base,activation_steer '{"steer_alpha": 0.05}'
run_one smoke_act01     base,activation_steer '{"steer_alpha": 0.1}'
run_one smoke_actmid01  base,activation_steer '{"steer_alpha": 0.1, "steer_layers": "8,9,10,11,12,13,14,15"}'
run_one smoke_actmid03  base,activation_steer '{"steer_alpha": 0.3, "steer_layers": "8,9,10,11,12,13,14,15"}'
echo "end=$(date)"
