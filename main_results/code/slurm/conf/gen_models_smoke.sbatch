#!/bin/bash
#SBATCH --job-name=models_smoke
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
for T in llama1b qwen08b; do
  C="$PY scripts/steered_semstamp.py --run-name ob_smoke_$T --target $T --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --limit 2"
  $C --arm base || { echo "$T base FAILED"; exit 7; }
  $C --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --steer marginal --beta 1 --arm-label chain_reference || { echo "$T push FAILED"; exit 8; }
  echo "$T smoke passed ($(date))"
done
