#!/bin/bash
#SBATCH --job-name=wm_abl_poe3
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# PoE at 512 tokens on the 60-prompt dev subset with EOS excluded from steering (poe_steer_eos=false),
# then the same steering with seed-phrase keys (random_words family, key set 'randomwords').
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
MODEL_SNAP=$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
[ -r "$MODEL_SNAP/config.json" ] || { echo "PREFLIGHT FAILED on $(hostname): model snapshot not readable at $MODEL_SNAP"; exit 3; }
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
run_one () {
  NAME=$1; STAGE=$2; OVR=$3
  echo "=== variant $NAME stage=$STAGE override=$OVR $(date) ==="
  $PY -m wmworld.cli generate --config $CFG --stage $STAGE --run-name $NAME --arms base,poe_steer --override "$OVR" --dry-run || return 1
  $PY -m wmworld.cli generate --config $CFG --stage $STAGE --run-name $NAME --arms base,poe_steer --override "$OVR" || return 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$NAME || return 1
}
run_one abl_poe10c02_noeos ablation '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false}'
run_one abl_poe5c05_noeos ablation '{"poe_beta": 5.0, "poe_clip": 0.5, "poe_steer_eos": false}'
# seed-phrase keys: build once (rarity scoring + guard on the GPU), then steer with the clip-0.2 setting
$PY -m wmworld.cli make-keys --config $CFG --set randomwords || exit 1
run_one abl_rw_poe10c02_noeos ablation_rw '{"poe_beta": 10.0, "poe_clip": 0.2, "poe_entropy_min": 0.2, "poe_steer_eos": false}'
echo "end=$(date)"
