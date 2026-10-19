#!/bin/bash
#SBATCH --job-name=baseline_kgw
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f'free GPU memory {f/2**30:.1f} GiB'); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
echo "host=$(hostname) start=$(date)"
for I in 0 1 2; do $PY scripts/baselines.py --method kgw --run-name baseline_kgw --shard $I --num-shards 3 > slurm_logs/bl_kgw_$I.log 2>&1 & done; wait
echo "kgw simple_1 done ($(date))"
for I in 0 1 2; do $PY scripts/baselines.py --method kgw --seeding-scheme selfhash --run-name baseline_kgw_selfhash --shard $I --num-shards 3 > slurm_logs/bl_kgwsh_$I.log 2>&1 & done; wait
echo "kgw selfhash done ($(date))"
echo "end=$(date)"
