#!/bin/bash
#SBATCH --job-name=wm_abl_rw8
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
# small-sample follow-ups on the best setting (adaptive beta 10 / clip 0.5, guards, EOS pinned, seed-phrase keys):
#   1. maintenance push: keep 30% of beta after the target so the tail of the text stays marked (crop-last / reorder)
#   2. clip 1.0 at beta 5 with the leak + repetition guards (re-open the clip now that key words cannot be inserted)
#   3. content-decision steering: push only where the generator is uncertain (entropy >= 1.5 nats, full strength at 3)
run_one rw_poe10c05_g_ad4_r4_f03 '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_block_repeat_n": 4, "poe_adapt": "tstat", "poe_t_target": 4.0, "poe_beta_floor": 0.3}'
run_one rw_poe5c1_g_r4           '{"poe_beta": 5.0, "poe_clip": 1.0, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_block_repeat_n": 4}'
run_one rw_poe10c05_g_ad4_r4_hi  '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_entropy_min": 1.5, "poe_entropy_ref": 3.0, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_block_repeat_n": 4, "poe_adapt": "tstat", "poe_t_target": 4.0}'
RUNS=""; for r in rw_poe10c05_g_ad4_r4_f03 rw_poe5c1_g_r4 rw_poe10c05_g_ad4_r4_hi; do [ -f runs/$r/generations.jsonl ] && RUNS="$RUNS --run runs/$r"; done
$PY scripts/t_null_pool.py --config $CFG $RUNS
$PY scripts/rank_false_attribution.py --bank decoy_rw runs/rw_poe10c05_g_ad4_r4_f03 runs/rw_poe5c1_g_r4 runs/rw_poe10c05_g_ad4_r4_hi
echo "end=$(date)"
