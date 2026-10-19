#!/bin/bash
#SBATCH --job-name=ss_draft_v2
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
[ -f /l/users/daniil.orel/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/config.json ] || { echo "3B NOT VISIBLE FROM THIS NODE"; exit 6; }
[ -f /l/users/daniil.orel/hf_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/config.json ] || { echo "1B NOT VISIBLE"; exit 6; }
echo "host=$(hostname) start=$(date)"
# 1B drafts, v2 (keep rule: pass AND in mask; token-substitution repair; tighter 1B sampler). Same 30 prompts as ss_chain_smoke.
S="$PY scripts/steered_semstamp.py --run-name ss_chain_smoke --per-domain 10 --arm chain --W 16 --draft nucleus --draft-v 2 --best-of 8 --verify-top 4 --max-repairs 4"
arm() { for I in 0 1 2; do $S "$@" --shard $I --num-shards 3 > slurm_logs/ss_draft_v2_$(echo "$@" | tr ' -' '__')_${I}.log 2>&1 & done; wait; echo "arm $* done ($(date))"; }
arm --unit seg --draft-temperature 0.7 --draft-top-p 0.9
arm --unit sent --draft-temperature 0.7 --draft-top-p 0.9
arm --unit seg
$PY scripts/ss_draft_diag.py
echo "end=$(date)"
