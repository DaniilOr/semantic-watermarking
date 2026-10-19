#!/bin/bash
#SBATCH --job-name=wm_attack_crop
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
# front-loading check: crop and reorder attacks only (deterministic, fast) on the maintenance-push run and its twin without it
for r in rw_poe10c05_g_ad4_r4_f03 rw_poe10c05_g_ad4_r4; do
  $PY -m wmworld.cli attack --config $CFG --run-name $r --subset crop,reorder,resegment || exit 1
  $PY -m wmworld.cli evaluate --config $CFG --run runs/attacks_$r --no-decoys || exit 1
  $PY scripts/attack_table.py runs/attacks_$r runs/$r
done
echo "end=$(date)"
