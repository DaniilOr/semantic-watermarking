#!/bin/bash
#SBATCH --job-name=lr_depths
#SBATCH --time=12:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# LR detector check (user 2026-11-14, futuremark.pdf): sentence depths of every text of the large-scale runs, 4 lanes.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/nfs-stor/daniil.orel/home_moved/huggingface_cache TOKENIZERS_PARALLELISM=false RUNS_ROOT=/nfs-stor/daniil.orel/home_moved/wmworld_runs   # /l metadata is very slow on 2026-11-14
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
echo "host=$(hostname) start=$(date)"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
P=(); for g in 0 1; do for l in 0 1; do k=$((g * 2 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/lr_detect.py depths --embedder /nfs-stor/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282 --shard $k --num-shards 4 > slurm_logs/lr_depths_$k.log 2>&1 & P+=($!); done; done
FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
echo "lanes done (failure: $FAIL) $(date)"; exit $FAIL
