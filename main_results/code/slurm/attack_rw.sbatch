#!/bin/bash
#SBATCH --job-name=wm_attack_rw
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
# robustness at development scale: the registered attack suite on the adaptive seed-phrase run (60 base + 60 marked parents;
# same-model paraphrases, resegmentation, reordering, crops, dilution with donor-pool text), then scoring under the 12 keys
$PY -m wmworld.cli attack --config $CFG --run-name rw_poe5c05_g_ad || exit 1
$PY -m wmworld.cli evaluate --config $CFG --run runs/attacks_rw_poe5c05_g_ad --no-decoys || exit 1
echo "end=$(date)"
