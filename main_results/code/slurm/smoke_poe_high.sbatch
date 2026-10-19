#!/bin/bash
#SBATCH --job-name=wm_smoke_poeh
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# PoE steering, high-beta sweep with contrast-clip variants (smoke: 12 prompts, 64 tokens).
# beta=2 (clip 5) already doubled the selection method's shift with 17% key-word leakage; a small clip should block the
# "insert the key word" shortcut (contrast of several nats) while keeping the diffuse per-token signal (fractions of a nat).
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
  $PY -m wmworld.cli generate --config $CFG --stage smoke --run-name $NAME --arms base,poe_steer --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage smoke --run-name $NAME --arms base,poe_steer --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
run_one smoke_poe2c1     '{"poe_beta": 2.0, "poe_clip": 1.0}'
run_one smoke_poe5c1     '{"poe_beta": 5.0, "poe_clip": 1.0, "poe_entropy_min": 0.2}'
run_one smoke_poe5c05    '{"poe_beta": 5.0, "poe_clip": 0.5, "poe_entropy_min": 0.2}'
run_one smoke_poe10c05   '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_entropy_min": 0.2}'
run_one smoke_poe10c02   '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2}'
run_one smoke_poe5       '{"poe_beta": 5.0, "poe_entropy_min": 0.2}'
echo "end=$(date)"
