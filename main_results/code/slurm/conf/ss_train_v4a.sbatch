#!/bin/bash
#SBATCH --job-name=rl_v4a_bigjudge
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# v4a_bigjudge on one GPU (amendment 19): GRPO from the base model, reward judge gemma-3-27b-it via OpenRouter, 32 rollouts,
# patience 5 (stale-rule any); smoke first.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
RL="$PY scripts/ss_rl_mask.py --lr 2e-4 --group 32 --beta 0.2 --stop-patience 5 --stale-rule any --judge api --judge-model google/gemma-3-27b-it"
$RL --out runs/ss_rl/v4a_bigjudge_smoke --ckpt-dir runs/ss_rl/v4a_bigjudge_smoke --steps 2 --eval-every 2 --eval-n 2 --save-every 1000000 > slurm_logs/rl_v4a_bigjudge_smoke.log 2>&1 || { echo "SMOKE FAILED"; exit 7; }
echo "smoke passed ($(date))"
$RL --out runs/ss_rl/v4a_bigjudge --ckpt-dir /l/users/daniil.orel/wmworld_rl/v4a_bigjudge --steps 1500 --eval-every 100 --eval-n 48 --save-every 100 > slurm_logs/rl_v4a_bigjudge.log 2>&1
echo "exit=$? end=$(date)"
