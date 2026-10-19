#!/bin/bash
#SBATCH --job-name=ob_gen_v3
#SBATCH --time=36:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:3
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
[ -f /home/daniil.orel/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/config.json ] || { echo "3B NOT VISIBLE"; exit 6; }
# GRPO run 3 checkpoints on the benchmark (amendment 18 evaluation): one arm per GPU, 3 shards each, same flags as the
# other adapter arms (sentence chain, blocks of 4, cap 16, stop margin 0.02, adapter = mask id, no push)
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
C="$PY scripts/steered_semstamp.py --run-name ob_bench --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapter-index mask"
arm() { g=$1; L=$2; A=$3; for k in 0 1 2; do CUDA_VISIBLE_DEVICES=${G[$g]} $C --arm-label $L --adapters $A --shard $k --num-shards 3 > slurm_logs/ob_${L}_${k}.log 2>&1 & done; }
arm 0 chain_v3a1000 /l/users/daniil.orel/wmworld_rl/v3a_patience/adapters_step1000.pt
arm 1 chain_v3a1400 /l/users/daniil.orel/wmworld_rl/v3a_patience/adapters_step1400.pt
arm 2 chain_v3b600 /l/users/daniil.orel/wmworld_rl/v3b_para/adapters_step600.pt
wait
echo "end=$(date)"
