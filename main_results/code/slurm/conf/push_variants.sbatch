#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# Push + margin variants (2026-10-08; scripts/push_variants.py, original code untouched): Llama-3B chat, first 300 GSM8K and
# 300 Dolly prompts (--per-domain 300), instruction-task protocol (512 cap, no minimum, end-of-turn and line-break rules).
# VARIANTS = space-separated labels among: chain_spread chain_protect chain_spread_protect chain_beta05.
# Plain / clean / SemStamp / original push texts for the same prompts were copied into runs/it_fix_llama3b.
VARIANTS=${VARIANTS:?}
echo "host=$(hostname) start=$(date) variants=$VARIANTS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEC="--target llama3b --prompts it_bench --ob-domains it_gsm8k it_dolly --max-new 512 --min-new 0 --W 16 --join-leading-newline"
CHAIN="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --stop-margin 0.02 --eot-at-boundary"
vargs() { case $1 in chain_spread) echo "$CHAIN --beta 1 --spread-gate";; chain_protect) echo "$CHAIN --beta 1 --protect-tokens";;
  chain_spread_protect) echo "$CHAIN --beta 1 --spread-gate --protect-tokens";; chain_beta05) echo "$CHAIN --beta 0.5";; esac; }
for L in $VARIANTS; do
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/push_variants.py --run-name it_fix_smoke $DEC --arm-label $L $(vargs $L) --per-domain 1 \
    > slurm_logs/it_fix_smoke_$L.log 2>&1 || { echo "SMOKE FAILED $L"; cat slurm_logs/it_fix_smoke_$L.log; exit 6; }
done
$PY scripts/smoke_check.py --leaks runs/it_fix_smoke || { echo "SMOKE: failed texts or leaks"; exit 7; }
echo "smoke ok ($(date))"
for L in $VARIANTS; do
  P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/push_variants.py --run-name it_fix_llama3b $DEC --arm-label $L $(vargs $L) --per-domain 300 --shard $k --num-shards 6 > slurm_logs/it_fix_${L}_$k.log 2>&1 & P+=($!); done; done
  FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
  [ $FAIL = 0 ] || { echo "$L: A SHARD FAILED ($(date))"; exit 8; }
  echo "$L done ($(date))"
done
echo "end=$(date)"
