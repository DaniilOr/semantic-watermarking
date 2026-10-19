#!/bin/bash
#SBATCH --job-name=ss_r2
#SBATCH --time=06:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# Round-2 arm (amendment 6).  Usage: sbatch --export=ALL,LABEL=segA_b2,ARGS="--steer table --beta 2" slurm/conf/ss_r2_arm.sbatch
LABEL=${LABEL:?set LABEL}; ARGS=${ARGS:?set ARGS}
S="$PY scripts/steered_semstamp.py --run-name ss_llama3b --arm steered_seg --arm-label $LABEL --W 16 --embedder /l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
echo "host=$(hostname) start=$(date) label=$LABEL args=$ARGS"
for I in 0 1 2; do $S $ARGS --shard $I --num-shards 3 > slurm_logs/ss_r2_${LABEL}_${I}.log 2>&1 & done; wait
echo "$LABEL done; end=$(date)"
