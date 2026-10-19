#!/bin/bash
#SBATCH --job-name=wm_smoke_poe
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Extension, small sample first (smoke: 12 prompts, 64 tokens): token-level product-of-experts steering
#   p'(t) ∝ p_G(t) exp(beta_t * clip(log p_key - log p_neutral)), beta_t entropy-gated (0 below 0.5 nats, full at 2 nats).
#   beta 0.5 / 1 / 2 gated, and beta 1 ungated; base as control; both detectors + leakage.
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
run_one smoke_poe05   '{"poe_beta": 0.5}'
run_one smoke_poe1    '{"poe_beta": 1.0}'
run_one smoke_poe2    '{"poe_beta": 2.0}'
run_one smoke_poe1ng  '{"poe_beta": 1.0, "poe_gate": "none"}'
echo "end=$(date)"
