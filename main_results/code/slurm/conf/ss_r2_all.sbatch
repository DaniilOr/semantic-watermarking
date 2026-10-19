#!/bin/bash
#SBATCH --job-name=ss_r2_all
#SBATCH --time=08:00:00
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
# Round-2 arms A, B, E (amendment 6), sequential, 3 shards each.
S="$PY scripts/steered_semstamp.py --run-name ss_llama3b --arm steered_seg --W 16 --embedder /l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
echo "host=$(hostname) start=$(date)"
run() { L=$1; shift; for I in 0 1 2; do $S --arm-label $L "$@" --shard $I --num-shards 3 > slurm_logs/ss_r2_${L}_${I}.log 2>&1 & done; wait; echo "$L done ($(date))"; }
run segA_b2 --steer table --beta 2
run segA_b3 --steer table --beta 3
run segB_b2 --steer marginal --beta 2
run bestof4B_b1 --best-of 4 --steer marginal --beta 1
echo "end=$(date)"
