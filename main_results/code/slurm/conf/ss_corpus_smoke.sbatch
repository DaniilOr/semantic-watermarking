#!/bin/bash
#SBATCH --job-name=ss_corpus_smoke
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
# Amendment 11 step 2 smoke: candidate corpus (all 16 candidates per sentence logged) on 6 validation prompts.
echo "host=$(hostname) start=$(date)"
rm -rf runs/ss_corpus_smoke
$PY scripts/steered_semstamp.py --run-name ss_corpus_smoke --prompts ss_train_val --per-domain 2 --arm chain --unit sent --best-of 16 --W 16 \
    --log-cands runs/ss_corpus_smoke/cands || exit 1
echo "end=$(date)"
