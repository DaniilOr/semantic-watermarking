#!/bin/bash
#SBATCH --job-name=wm_abl_rw
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# 512-token PoE with SEED-PHRASE keys (random_words, key set randomwords; user direction): reference settings, EOS excluded, leak guard on.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
run_one () {
  NAME=$1; OVR=$2
  echo "=== variant $NAME override=$OVR $(date) ==="
  $PY -m wmworld.cli generate --config $CFG --stage ablation_rw --run-name $NAME --arms base,poe_steer --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage ablation_rw --run-name $NAME --arms base,poe_steer --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
# quality-proxy null: a second unwatermarked draw (seed 1) of the same 60 prompts
$PY -m wmworld.cli generate --config $CFG --stage ablation_rw --run-name rw_base_seed1 --arms base --seeds 1 || exit 1
$PY -m wmworld.cli evaluate --config $CFG --run runs/rw_base_seed1 --no-decoys || exit 1
run_one rw_poe5c05_g    '{"poe_beta": 5.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true}'
run_one rw_poe10c02_g   '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true}'
run_one rw_poe2c1_g     '{"poe_beta": 2.0, "poe_clip": 1.0, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true}'
RUNS=""; for r in rw_poe5c05_g rw_poe10c02_g rw_poe2c1_g; do [ -f runs/$r/generations.jsonl ] && RUNS="$RUNS --run runs/$r"; done
$PY scripts/t_null_pool.py --config $CFG $RUNS
echo "end=$(date)"
