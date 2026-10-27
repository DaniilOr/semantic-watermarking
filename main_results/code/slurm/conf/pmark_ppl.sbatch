#!/bin/bash
#SBATCH --job-name=pmark_ppl
#SBATCH --time=06:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# Perplexity of the PMark continuation texts: same scorer and definition as for plain / SemStamp / push (ppl.sbatch: OPT-2.7B,
# scripts/ppl_eval.py), for the comparison tables.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface HF_HUB_CACHE=/l/users/daniil.orel/hf_cache/hub TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
echo "host=$(hostname) start=$(date)"
$PY scripts/ppl_eval.py --runs runs/pm_sp_bench_llama3b runs/pm_ob_bench runs/pm_setting_opt13b_ss --arms pmark --out runs/ppl_pmark.json
echo "exit=$? end=$(date)"
