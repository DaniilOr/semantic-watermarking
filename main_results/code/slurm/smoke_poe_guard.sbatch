#!/bin/bash
#SBATCH --job-name=wm_smoke_guard
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# PoE steering with the leak guard (key words never pushed), the category guard, and the adaptive t-statistic controller
# (push only until the running within-text t reaches the target).  Smoke scale first: 12 prompts, 64 tokens.
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
# quality-proxy null at full length: a second unwatermarked draw (seed 1) of the same 60 prompts; the difference of its
# task-conditioned mean log-likelihood from the seed-0 base tells how much of a steered arm's paired delta is draw noise
$PY -m wmworld.cli generate --config configs/pilot.yaml --stage ablation --run-name abl_base_seed1 --arms base --seeds 1 --dry-run || exit 1
$PY -m wmworld.cli generate --config configs/pilot.yaml --stage ablation --run-name abl_base_seed1 --arms base --seeds 1 || exit 1
$PY -m wmworld.cli evaluate --config configs/pilot.yaml --run runs/abl_base_seed1 --no-decoys || exit 1
run_one smoke_poe10c02_g    '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true}'
run_one smoke_poe10c02_gc   '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_block_category": true}'
run_one smoke_poe10c02_ad   '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_adapt": "tstat", "poe_t_target": 4.0}'
run_one smoke_poe20c02_ad   '{"poe_beta": 20.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_adapt": "tstat", "poe_t_target": 4.0}'
run_one smoke_poe5c05_g     '{"poe_beta": 5.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true}'
echo "end=$(date)"
