#!/bin/bash
#SBATCH --job-name=ob_gen_chain_v4d100
#SBATCH --time=36:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# chain_v4d100 on the benchmark (amendment 19 evaluation, best monitor step): same flags as the other adapter arms, 6 shards on 2 GPUs.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
C="$PY scripts/steered_semstamp.py --run-name ob_bench --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapter-index mask --adapters /l/users/daniil.orel/wmworld_rl/v4d_bigjudge_klcap/adapters_step100.pt --arm-label chain_v4d100"
for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $C --shard $k --num-shards 6 > slurm_logs/ob_chain_v4d100_$k.log 2>&1 & done; done; wait
echo "end=$(date)"
