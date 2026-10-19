#!/bin/bash
#SBATCH --job-name=wm_reeval_rw
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
# same-family decoy bank (decoy_rw): re-score the seed-phrase runs under the 100 random-word decoys (rank detector),
# then the pooled-null diagnostic with the decoy-relative t-statistic (8 same-family decoys) and the rank false-attribution table
for r in rw_poe10c05_g_ad4 rw_poe5c05_g rw_poe10c05_g_ad6 rw_poe5c05_g_r4 rw_poe10c05_g; do
  $PY -m wmworld.cli evaluate --config $CFG --run runs/$r || exit 1
done
$PY scripts/t_null_pool.py --config $CFG --n-decoys 8 --run runs/rw_poe10c05_g_ad4 --run runs/rw_poe5c05_g --run runs/rw_poe10c05_g_ad6 --run runs/rw_poe5c05_g_r4 --run runs/rw_poe10c05_g
$PY scripts/rank_false_attribution.py --bank decoy_rw runs/rw_poe10c05_g_ad4 runs/rw_poe5c05_g runs/rw_poe10c05_g_ad6 runs/rw_poe5c05_g_r4 runs/rw_poe10c05_g
echo "end=$(date)"
