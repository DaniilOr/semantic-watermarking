#!/bin/bash
#SBATCH --job-name=ss_retrain_home
#SBATCH --time=24:00:00
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
# /l hung (2026-09-25): re-run the GRPO v2 recipe to step 400 and the 3 RFT rounds with checkpoints on /home.
echo "host=$(hostname) start=$(date)"
$PY scripts/ss_rl_mask.py --out runs/ss_rl/v2_rerun --ckpt-dir /home/daniil.orel/wmworld_rl/v2_rerun --lr 2e-4 --group 16 --beta 0.2 --judge local \
    --steps 400 --eval-every 100 --eval-n 48 --save-every 100 --stop-patience 2 || exit 1
$PY scripts/ss_rft_lora.py --out runs/ss_rft/v1_rerun --ckpt-dir /home/daniil.orel/wmworld_rl/rft_v1_rerun || exit 1
echo "end=$(date)"
