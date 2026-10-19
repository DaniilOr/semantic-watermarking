#!/bin/bash
#SBATCH --job-name=wm_abl_rw9
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
# decoy push follow-ups: (1) clip 1.0 (the wider clip gave +20 points detection at beta 5 with the neutral push but 17% key-word
# leakage; the decoy push leaks 1.7%, so test whether the gain transfers cleanly); (2) beta 15 at clip 0.5 (headroom: cost -0.20, leak 1.7%)
run_one rw_poe10c1_g_bang4_r4_dec  '{"poe_beta": 10.0, "poe_clip": 1.0, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_adapt": "bang", "poe_t_target": 4.0, "poe_block_repeat_n": 4, "poe_contrast": "decoys", "poe_n_decoys": 3}'
run_one rw_poe15c05_g_bang4_r4_dec '{"poe_beta": 15.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_adapt": "bang", "poe_t_target": 4.0, "poe_block_repeat_n": 4, "poe_contrast": "decoys", "poe_n_decoys": 3}'
$PY scripts/t_null_pool.py --config $CFG --n-decoys 8 --run runs/rw_poe10c1_g_bang4_r4_dec --run runs/rw_poe15c05_g_bang4_r4_dec
$PY scripts/rank_false_attribution.py --bank decoy_rw runs/rw_poe10c1_g_bang4_r4_dec runs/rw_poe15c05_g_bang4_r4_dec
echo "end=$(date)"
