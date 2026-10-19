#!/bin/bash
#SBATCH --job-name=ob_gen_v4a800
#SBATCH --time=36:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# v4a_bigjudge step-800 adapters on the benchmark (amendment 19 evaluation): same flags as the other adapter arms
# (sentence chain, blocks of 4, cap 16, stop margin 0.02, adapter = mask id, no push); 3 shards on one GPU.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
C="$PY scripts/steered_semstamp.py --run-name ob_bench --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapter-index mask --adapters /l/users/daniil.orel/wmworld_rl/v4a_bigjudge/adapters_step800.pt --arm-label chain_v4a800"
for k in 0 1 2; do $C --shard $k --num-shards 3 > slurm_logs/ob_chain_v4a800_$k.log 2>&1 & done; wait
echo "end=$(date)"
