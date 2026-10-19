#!/bin/bash
#SBATCH --job-name=sp_gen_model
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
# Amendment 22: SemStamp's datasets and protocol (data/sp_bench: RealNews, BookSum, Reddit-TIFU, MBPP; 32-token prompts,
# 200 tokens, 1,000 samples; MBPP 974).  Step 0: regenerate this model's empty push + margin texts of the earlier
# benchmark (first-sentence fix).  Then the four arms, 6 shards on 2 GPUs each.
TARGET=${TARGET:?set TARGET}; OBRUN=${OBRUN:?set OBRUN}
echo "host=$(hostname) start=$(date) target=$TARGET CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
$PY scripts/drop_empty.py runs/$OBRUN chain_reference
O="$PY scripts/steered_semstamp.py --run-name $OBRUN --target $TARGET --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --arm-label chain_reference --arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02"
for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $O --shard $k --num-shards 6 > slurm_logs/refill_${OBRUN}_$k.log 2>&1 & done; done; wait
echo "empty texts refilled ($(date))"
C="$PY scripts/steered_semstamp.py --run-name sp_bench_$TARGET --target $TARGET --prompts sp_bench --ob-domains sp_realnews sp_booksum sp_tifu sp_mbpp --max-new 200 --W 16"
run() { L=$1; shift; for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $C --arm-label $L "$@" --shard $k --num-shards 6 > slurm_logs/sp_${TARGET}_${L}_${k}.log 2>&1 & done; done; wait; echo "$L done ($(date))"; }
run base --arm base
run clean_seed1 --arm base --seed 1
run chain_reference --arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02
run semstamp --arm semstamp
echo "end=$(date)"
