#!/bin/bash
#SBATCH --job-name=ss_train_adapters
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
# Amendment 11 step 4: train the 8 bin adapters on the full corpus (1 GPU), then the validation measurements (step 6).
echo "host=$(hostname) start=$(date)"
$PY scripts/ss_train_adapters.py --train runs/ss_corpus --val runs/ss_corpus_val --out runs/ss_adapters/v1 || exit 1
$PY scripts/ss_adapter_eval.py --adapters runs/ss_adapters/v1/adapters.pt --val runs/ss_corpus_val --contexts 200 --n 8 || exit 1
echo "end=$(date)"
