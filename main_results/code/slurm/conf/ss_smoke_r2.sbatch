#!/bin/bash
#SBATCH --job-name=ss_smoke_r2
#SBATCH --time=03:00:00
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
S="$PY scripts/steered_semstamp.py --run-name ss_smoke_r2 --per-domain 2 --max-new 200"
$S --arm base || exit 1
$S --arm steered_seg --steer table --beta 2 --arm-label segA_b2 || exit 1
$S --arm steered_seg --steer table --beta 3 --arm-label segA_b3 || exit 1
$S --arm steered_seg --steer marginal --beta 2 --arm-label segB_b2 || exit 1
$S --arm steered_seg --best-of 4 --beta 0 --arm-label bestof4_seg || exit 1
$S --arm steered_seg --best-of 4 --steer marginal --beta 1 --arm-label bestof4B_b1 || exit 1
$PY scripts/ss_smoke_read.py runs/ss_smoke_r2
echo "end=$(date)"
