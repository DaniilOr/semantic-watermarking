#!/bin/bash
#SBATCH --job-name=ob_speedtest
#SBATCH --time=02:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Sync-free multilora: chain_grpo300 on the first 18 ArXiv prompts, 3 processes on one GPU (the production setup);
# compared offline with the old-code records of the same prompts in runs/ob_bench.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for k in 0 1 2; do $PY scripts/steered_semstamp.py --run-name ob_speedtest --prompts ob_bench --ob-domains ob_arxiv --limit 18 --max-new 200 --W 16 \
  --arm-label chain_grpo300 --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapter-index mask \
  --adapters /l/users/daniil.orel/wmworld_rl/v2/adapters_step300.pt --shard $k --num-shards 3 > slurm_logs/ob_speedtest_$k.log 2>&1 & done; wait
echo "end=$(date)"
