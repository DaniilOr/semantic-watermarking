#!/bin/bash
#SBATCH --job-name=wm_smoke_poekl
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# PoE steering with the KL-constrained adaptive gate (smoke: 12 prompts, 64 tokens): beta_t chosen per token so that
# KL(steered || base) <= kappa; beta_max=10, clip 0.5.  Sweep kappa 0.02 / 0.05 / 0.1 / 0.2 nats per token.
# Evaluation now also reports the task-conditioned log-likelihood (destructiveness proxy) and category-level leakage.
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
run_one smoke_poekl002 '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_gate": "kl", "poe_kl_budget": 0.02}'
run_one smoke_poekl005 '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_gate": "kl", "poe_kl_budget": 0.05}'
run_one smoke_poekl01  '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_gate": "kl", "poe_kl_budget": 0.1}'
run_one smoke_poekl02  '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_gate": "kl", "poe_kl_budget": 0.2}'
# re-evaluate the earlier fixed-beta runs so they also get the quality proxy and category leakage (scores are cached)
for r in smoke_poe2c1 smoke_poe5c05 smoke_poe10c05 smoke_poe10c02 smoke_poe2 smoke_poe5; do
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$r || true
done
echo "end=$(date)"
