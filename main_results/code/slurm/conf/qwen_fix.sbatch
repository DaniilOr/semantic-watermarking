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
# Qwen fix (2026-10-03, user go): Qwen's tokenizer writes a paragraph break as its own token, so without
# --join-leading-newline a unit could be only "\n\n"; push + margin dropped such text-free candidates and ended the text
# (chain texts 125 vs 199 tokens), SemStamp kept them out of step with the detector.  Both watermark arms regenerated with
# the flag, original protocol otherwise; plain / clean texts copied from the original runs.  Run dirs <original>_nl.
T=${T:?set T}; PLAIN=${PLAIN:-}
echo "host=$(hostname) start=$(date) T=$T plain=${PLAIN:-none} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for B in ob_bench sp_bench; do
  mkdir -p runs/${B}_${T}_nl
  for a in base clean_seed1; do for f in runs/${B}_${T}/generations.$a.shard*.jsonl; do cp -n $f runs/${B}_${T}_nl/; done; done
done
OB="--prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia"; SP="--prompts sp_bench --ob-domains sp_realnews sp_booksum sp_tifu"
COMMON="--target $T --max-new 200 --W 16 --join-leading-newline $PLAIN"
arm_args() { case $1 in chain_reference) echo "--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02";; semstamp) echo "--arm semstamp";; esac; }
for L in chain_reference semstamp; do
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/steered_semstamp.py --run-name qwen_fix_smoke_$T $OB $COMMON --arm-label $L $(arm_args $L) --per-domain 1 \
    > slurm_logs/qwen_fix_smoke_${T}_$L.log 2>&1 || { echo "SMOKE FAILED $L"; cat slurm_logs/qwen_fix_smoke_${T}_$L.log; exit 6; }
done
$PY scripts/smoke_check.py runs/qwen_fix_smoke_$T || { echo "SMOKE: failed texts"; exit 6; }
echo "smoke ok ($(date))"
for B in ob_bench sp_bench; do
  [ $B = ob_bench ] && P_=$OB || P_=$SP
  for L in chain_reference semstamp; do
    P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/steered_semstamp.py --run-name ${B}_${T}_nl $P_ $COMMON --arm-label $L $(arm_args $L) --shard $k --num-shards 6 > slurm_logs/${B}_${T}_nl_${L}_$k.log 2>&1 & P+=($!); done; done
    FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
    [ $FAIL = 0 ] || { echo "${B} $L: A SHARD FAILED ($(date))"; exit 8; }
    echo "${B}_${T}_nl $L done ($(date))"
  done
done
echo "end=$(date)"
