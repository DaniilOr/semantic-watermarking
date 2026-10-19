#!/bin/bash
#SBATCH --job-name=wm_trade_p0
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
CFG=configs/pilot.yaml
NULL=runs/null_calibration/eval/posstats_randomwords.npz
echo "host=$(hostname) start=$(date)"
# the generator refuses large stages without a budget manifest: write it first (idempotent, no model needed)
$PY -m wmworld.cli generate --config $CFG --stage ablation_tradeoff --run-name tradeoff --dry-run || exit 1
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"   # the GPUs Slurm allocated -- never assume 0 and 1
# trade-off sweep: shards 0 and 1 of 4, one per GPU, in parallel (60 dev prompts x 11 arms in total)
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -m wmworld.cli generate --config $CFG --stage ablation_tradeoff --run-name tradeoff --shard 0 --num-shards 4 > slurm_logs/tradeoff_shard0.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY -m wmworld.cli generate --config $CFG --stage ablation_tradeoff --run-name tradeoff --shard 1 --num-shards 4 > slurm_logs/tradeoff_shard1.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "shard 0 exit=$R1  shard 1 exit=$R2  end=$(date)"
[ $R1 -eq 0 ] && [ $R2 -eq 0 ]
