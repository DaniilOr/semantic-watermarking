#!/bin/bash
#SBATCH --job-name=wm_abl_poe4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# 512-token PoE variants aimed at "detectable but not destructive", all with the leak guard and EOS excluded:
#   1. beta 10 / clip 0.5 (strongest non-leaking smoke setting; the guard handles the leakage seen at 512 tokens)
#   2. beta 5 / clip 0.5 with the adaptive t-statistic controller (stop pushing once the text is detectable)
#   3. beta 8 / clip 0.5 pushed only at uncertain positions (entropy gate 1.0 -> 3.0 nats)
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
  $PY -m wmworld.cli generate --config $CFG --stage ablation --run-name $NAME --arms base,poe_steer --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage ablation --run-name $NAME --arms base,poe_steer --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
run_one abl_poe10c05_g   '{"poe_beta": 10.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true}'
run_one abl_poe5c05_g_ad '{"poe_beta": 5.0, "poe_clip": 0.5, "poe_entropy_min": 0.2, "poe_steer_eos": false, "poe_block_key_tokens": true, "poe_adapt": "tstat", "poe_t_target": 4.0}'
run_one abl_poe8c05_hi   '{"poe_beta": 8.0, "poe_clip": 0.5, "poe_entropy_min": 1.0, "poe_entropy_ref": 3.0, "poe_steer_eos": false, "poe_block_key_tokens": true}'
# pooled-null t-detector diagnostic over every 512-token PoE run available (a few minutes on the GPU)
RUNS=""; for r in abl_poe5c05 abl_poe10c02 abl_poe20c02 abl_poe2c1 abl_poe10c02_noeos abl_poe5c05_noeos abl_poe10c05_g abl_poe5c05_g_ad abl_poe8c05_hi; do [ -f runs/$r/generations.jsonl ] && RUNS="$RUNS --run runs/$r"; done
$PY scripts/t_null_pool.py --config $CFG $RUNS
echo "end=$(date)"
