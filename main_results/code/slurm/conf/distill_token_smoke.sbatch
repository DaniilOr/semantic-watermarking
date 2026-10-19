#!/bin/bash
#SBATCH --job-name=distok_smoke
#SBATCH --time=01:00:00
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
echo "host=$(hostname) start=$(date)"
$PY scripts/ss_distill_token.py teacher --limit 20 --out runs/ss_distill_token_smoke/teacher || { echo "TEACHER FAILED"; exit 7; }
$PY scripts/ss_distill_token.py train --teacher runs/ss_distill_token_smoke/teacher --out runs/ss_distill_token_smoke --ckpt-dir runs/ss_distill_token_smoke/ckpt --epochs 1 --accum 4 --group 4 --eval-n 2 || { echo "TRAIN FAILED"; exit 8; }
echo "smoke passed end=$(date)"
