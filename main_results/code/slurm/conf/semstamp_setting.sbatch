#!/bin/bash
#SBATCH --job-name=semstamp_setting
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
# SemStamp-setting check (separate from the main runs): does SemStamp reach its paper's detectability in its own setting?
# Base (non-chat) generators continuing SemStamp's prompt (first sentence of the document, --plain) with SemStamp's
# decoding (T 0.7, top-k 0, no top-p, repetition penalty 1.05, 195-205 new tokens); RealNews test prompts of data/sp_bench.
# Models: AbeHou/opt-1.3b-semstamp (SemStamp's released generator), meta-llama/Llama-3.2-3B (base).
# Arms: base (model null), semstamp, chain_reference (push + margin).  Run dirs runs/semstamp_setting_<model>.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
DEC="--prompts sp_bench --ob-domains sp_realnews --plain semstamp --temperature 0.7 --top-p 1.0 --top-k 0 --rep-penalty 1.05 --min-new 195 --max-new 205 --W 16"
ARM_semstamp="--arm semstamp"
ARM_chain_reference="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02"
# smoke: 2 prompts per arm and model, one process each; any failure stops the job before the full runs
for M in opt13b_ss llama3b_base; do
  for L in base semstamp chain_reference; do
    case $L in base) A="--arm base";; semstamp) A=$ARM_semstamp;; chain_reference) A=$ARM_chain_reference;; esac
    CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/steered_semstamp.py --run-name semstamp_setting_smoke_$M --target $M $DEC --arm-label $L $A --limit 2 \
      > slurm_logs/semstamp_setting_smoke_${M}_$L.log 2>&1 || { echo "SMOKE FAILED $M $L"; cat slurm_logs/semstamp_setting_smoke_${M}_$L.log; exit 6; }
  done
done
$PY scripts/smoke_check.py runs/semstamp_setting_smoke_* || { echo 'SMOKE: failed texts'; exit 6; }
echo "smoke ok ($(date))"
for M in opt13b_ss llama3b_base; do
  run() { L=$1; shift; for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/steered_semstamp.py --run-name semstamp_setting_$M --target $M $DEC --arm-label $L "$@" --shard $k --num-shards 6 > slurm_logs/semstamp_setting_${M}_${L}_$k.log 2>&1 & done; done; wait; echo "$M $L done ($(date))"; }
  run base --arm base
  run semstamp $ARM_semstamp
  run chain_reference $ARM_chain_reference
done
echo "end=$(date)"
