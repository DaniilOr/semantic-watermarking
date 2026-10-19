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
# Code protocol (user 2026-09-29: "let models generate as much as they want"): MBPP (974 problems, data/sp_bench) with no
# 200-token budget -- cap 1024 new tokens, no minimum length -- and the end-of-turn fix (--eot-at-boundary) in both
# sentence-chain arms.  Instruct models, arms base, clean_seed1, chain_reference (push + margin), semstamp; runs/code_mbpp_<T>.
TARGETS=${TARGETS:?set TARGETS}
echo "host=$(hostname) start=$(date) targets=$TARGETS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEC="--prompts sp_bench --ob-domains sp_mbpp --max-new 1024 --min-new 0 --W 16"
ARM_chain_reference="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02 --eot-at-boundary"
ARM_semstamp="--arm semstamp --eot-at-boundary"
# smoke: 4 problems per arm (first target); errors or any text continuing past the end of turn stop the job
T0=${TARGETS%% *}
for L in base chain_reference semstamp; do
  case $L in base) A="--arm base";; semstamp) A=$ARM_semstamp;; chain_reference) A=$ARM_chain_reference;; esac
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/steered_semstamp.py --run-name code_smoke_$T0 --target $T0 $DEC --arm-label $L $A --limit 4 \
    > slurm_logs/code_smoke_${T0}_$L.log 2>&1 || { echo "SMOKE FAILED $L"; cat slurm_logs/code_smoke_${T0}_$L.log; exit 6; }
done
$PY scripts/smoke_check.py --leaks runs/code_smoke_$T0 || { echo 'SMOKE: failed texts or end-of-turn leaks'; exit 7; }
echo "smoke ok ($(date))"
run() { L=$1; shift; for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/steered_semstamp.py --run-name code_mbpp_$T --target $T $DEC --arm-label $L "$@" --shard $k --num-shards 6 > slurm_logs/code_mbpp_${T}_${L}_$k.log 2>&1 & done; done; wait; echo "$T $L done ($(date))"; }
for T in $TARGETS; do
  run base --arm base
  run clean_seed1 --arm base --seed 1
  run chain_reference $ARM_chain_reference
  run semstamp $ARM_semstamp
done
echo "end=$(date)"
