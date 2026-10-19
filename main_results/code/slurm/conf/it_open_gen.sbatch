#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
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
# Open-ended instruction tasks (2026-10-17): ELI5, HC3 finance, PERSUADE essays (scripts/it_open_build.py); same protocol as
# it_gen.sbatch (512-token cap, end-of-turn and leading-newline rules); arm kgw = KGW reference (scripts/kgw_arm.py).
# SemStamp last (user rule).  Run dirs runs/it_open_<T>.
TARGETS=${TARGETS:?set TARGETS}; ARMS=${ARMS:-base clean_seed1 chain_reference kgw semstamp}
echo "host=$(hostname) start=$(date) targets=$TARGETS arms=$ARMS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEC="--prompts it_bench --ob-domains ${DOMAINS:-it_eli5 it_hc3_finance it_essays} --max-new 512 --min-new 0 --W 16 --join-leading-newline"
arm_args() { case $1 in base) echo "--arm base";; clean_seed1) echo "--arm base --seed 1";;
  chain_reference) echo "--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02 --eot-at-boundary";;
  kgw) echo "--arm base";; semstamp) echo "--arm semstamp --eot-at-boundary --n-cand 20 --id-n-cand 4";; esac; }
script() { [ "$1" = kgw ] && echo scripts/kgw_arm.py || echo scripts/steered_semstamp.py; }
T0=${TARGETS%% *}
for L in $ARMS; do
  CUDA_VISIBLE_DEVICES=${G[0]} $PY $(script $L) --run-name it_open_smoke_$T0 --target $T0 $DEC --arm-label $L $(arm_args $L) --per-domain 1 \
    > slurm_logs/it_open_smoke_${T0}_$L.log 2>&1 || { echo "SMOKE FAILED $L"; cat slurm_logs/it_open_smoke_${T0}_$L.log; exit 6; }
done
$PY scripts/smoke_check.py --leaks runs/it_open_smoke_$T0 || { echo "SMOKE: failed texts or end-of-turn leaks"; exit 7; }
echo "smoke ok ($(date))"
for T in $TARGETS; do
  for L in $ARMS; do
    P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY $(script $L) --run-name it_open_$T --target $T $DEC --arm-label $L $(arm_args $L) --shard $k --num-shards 6 > slurm_logs/it_open_${T}_${L}_$k.log 2>&1 & P+=($!); done; done
    FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
    [ $FAIL = 0 ] || { echo "$T $L: A SHARD FAILED ($(date))"; exit 8; }
    echo "$T $L done ($(date))"
  done
done
echo "end=$(date)"
