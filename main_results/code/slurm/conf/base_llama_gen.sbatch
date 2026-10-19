#!/bin/bash
#SBATCH --job-name=base_llama_gen
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
# Base (non-chat) Llama-3.2-1B / -3B and Qwen3.5-0.8B-Base on both benchmarks: runs/ob_bench_<T> (ArXiv, BookSum, Wikipedia) and runs/sp_bench_<T>
# (RealNews, BookSum, Reddit-TIFU; no MBPP -- its prompt is an instruction).  The model continues the benchmark's excerpt
# as plain text (--plain excerpt, no instruction wrapper, no chat template); decoding and arms as for the instruct models
# (T 0.8, top-p 0.95, 200 tokens; base, clean_seed1, chain_reference = push + margin, semstamp).  6 shards on 2 GPUs.
TARGETS=${TARGETS:-llama1b_base qwen08b_base llama3b_base}
echo "host=$(hostname) start=$(date) targets=$TARGETS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
ARM_chain_reference="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02"
OBP="--prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia"; SPP="--prompts sp_bench --ob-domains sp_realnews sp_booksum sp_tifu"
# smoke: 2 prompts per arm and model (first benchmark); any failure stops the job before the full runs
for T in $TARGETS; do for L in base semstamp chain_reference; do
  case $L in base) A="--arm base";; semstamp) A="--arm semstamp";; chain_reference) A=$ARM_chain_reference;; esac
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/steered_semstamp.py --run-name base_models_smoke_$T --target $T $OBP --plain excerpt --max-new 200 --W 16 --arm-label $L $A --limit 2 \
    > slurm_logs/base_models_smoke_${T}_$L.log 2>&1 || { echo "SMOKE FAILED $T $L"; cat slurm_logs/base_models_smoke_${T}_$L.log; exit 6; }
done; done
$PY scripts/smoke_check.py runs/base_models_smoke_* || { echo 'SMOKE: failed texts'; exit 6; }
echo "smoke ok ($(date))"
run() { L=$1; shift; for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/steered_semstamp.py --run-name $RUN --target $T $P --plain excerpt --max-new 200 --W 16 --arm-label $L "$@" --shard $k --num-shards 6 > slurm_logs/${RUN}_${L}_$k.log 2>&1 & done; done; wait; echo "$RUN $L done ($(date))"; }
for T in $TARGETS; do
  for B in ob sp; do
    if [ $B = ob ]; then RUN=ob_bench_$T; P=$OBP; else RUN=sp_bench_$T; P=$SPP; fi
    run base --arm base
    run clean_seed1 --arm base --seed 1
    run chain_reference $ARM_chain_reference
    run semstamp --arm semstamp
  done
done
echo "end=$(date)"
