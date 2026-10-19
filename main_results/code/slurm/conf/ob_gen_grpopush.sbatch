#!/bin/bash
#SBATCH --job-name=ob_gen_grpopush
#SBATCH --time=36:00:00
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
[ -f /home/daniil.orel/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/config.json ] || { echo "3B NOT VISIBLE"; exit 6; }
# GRPO step 300 adapters WITH the push on top (2026-09-26): 2-prompt smoke first, then 3 shards on one GPU
F="--prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapter-index mask --adapters /l/users/daniil.orel/wmworld_rl/v2/adapters_step300.pt --steer marginal --beta 1 --arm-label chain_grpo300_push"
$PY scripts/steered_semstamp.py --run-name ob_smoke_grpopush $F --limit 2 > slurm_logs/ob_smoke_grpopush.log 2>&1 || { echo "SMOKE FAILED (see slurm_logs/ob_smoke_grpopush.log)"; exit 7; }
echo "smoke passed ($(date))"
for k in 0 1 2; do $PY scripts/steered_semstamp.py --run-name ob_bench $F --shard $k --num-shards 3 > slurm_logs/ob_chain_grpo300_push_$k.log 2>&1 & done; wait
echo "end=$(date)"
