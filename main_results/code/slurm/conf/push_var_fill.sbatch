#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Refill of chain_protect shard 5/6 in runs/it_fix_llama3b (crashed 2026-10-08 on a variants.json write race, before
# generating anything).  Shards 5,11,...,35 of 36 = exactly the prompts of shard 5 of 6, with the same per-prompt seeds.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEC="--target llama3b --prompts it_bench --ob-domains it_gsm8k it_dolly --max-new 512 --min-new 0 --W 16 --join-leading-newline"
CHAIN="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --stop-margin 0.02 --eot-at-boundary"
P=(); i=0
for k in 5 11 17 23 29 35; do
  CUDA_VISIBLE_DEVICES=${G[$((i % 2))]} $PY scripts/push_variants.py --run-name it_fix_llama3b $DEC --arm-label chain_protect $CHAIN --beta 1 \
    --protect-tokens --per-domain 300 --shard $k --num-shards 36 > slurm_logs/it_fix_chain_protect_fill_$k.log 2>&1 & P+=($!); i=$((i + 1))
done
FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
[ $FAIL = 0 ] || { echo "A SHARD FAILED ($(date))"; exit 8; }
echo "chain_protect refill done ($(date))"
