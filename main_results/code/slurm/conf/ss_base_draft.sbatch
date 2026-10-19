#!/bin/bash
#SBATCH --job-name=ss_base_draft
#SBATCH --time=03:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
read -r BETA W EMB PFX < <($PY -c "import json; c=json.load(open(\"assets/ss_choice.json\")); print(c[\"beta\"], c[\"W\"], c[\"embedder\"], c.get(\"embedder_prefix\") or \"-\")")
[ "$PFX" = "-" ] && PFX=""
S="$PY scripts/steered_semstamp.py --run-name ss_llama3b --beta $BETA --W $W --embedder $EMB --embedder-prefix=$PFX"
echo "host=$(hostname) start=$(date) beta=$BETA W=$W embedder=$EMB"
$S --arm base --draft assist --limit 50 > slurm_logs/ss_base_draft_0.log 2>&1; echo "base_draft done ($(date))"
echo "end=$(date)"
