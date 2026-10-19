#!/bin/bash
#SBATCH --job-name=wm_abl_len
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
# length sweep for the most balanced setting (beta 10 decoy push, bang-bang t4, guards): outputs capped at 128 and 256
# tokens for BOTH arms (stage-level generation settings), so detection vs length is measured on genuinely short texts
OVR='{"poe_beta": 10.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_adapt": "bang", "poe_t_target": 4.0, "poe_block_repeat_n": 4, "poe_contrast": "decoys", "poe_n_decoys": 3}'
for L in 128 256; do
  NAME=rw${L}_poe10c05_g_bang4_r4_dec
  echo "=== variant $NAME stage=ablation_rw$L $(date) ==="
  $PY -m wmworld.cli generate --config $CFG --stage ablation_rw$L --run-name $NAME --arms base,poe_steer --override "$OVR" --dry-run || exit 1
  $PY -m wmworld.cli generate --config $CFG --stage ablation_rw$L --run-name $NAME --arms base,poe_steer --override "$OVR" || exit 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || exit 1
done
$PY scripts/t_null_pool.py --config $CFG --n-decoys 8 --run runs/rw128_poe10c05_g_bang4_r4_dec --run runs/rw256_poe10c05_g_bang4_r4_dec
$PY scripts/rank_false_attribution.py --bank decoy_rw runs/rw128_poe10c05_g_bang4_r4_dec runs/rw256_poe10c05_g_bang4_r4_dec
echo "end=$(date)"
