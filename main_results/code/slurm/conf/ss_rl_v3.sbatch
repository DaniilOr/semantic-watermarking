#!/bin/bash
#SBATCH --job-name=ss_rl_v3
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
# GRPO run 3 (amendment 18): v2's recipe from the BASE model with 32 rollouts per context, stop after 5 monitors in a row
# without a new best clear-in-mask share (KL not part of the rule), checkpoints every 100 steps on /l.
#   TAG=v3a_patience                      -> v2 reward
#   TAG=v3b_para EXTRA="--para-model google/gemma-3-12b-it --w-para 1.0"  -> + depth of the gemma-3-12b paraphrase
# A 2-step smoke (memory at 32 rollouts, API) runs first; the real run starts only if it passes.
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
TAG=${TAG:?set TAG}; EXTRA=${EXTRA:-}
COMMON="--lr 2e-4 --group 32 --beta 0.2 --judge local --stop-patience 5 --stale-rule any"
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date) tag=$TAG extra=$EXTRA"
$PY scripts/ss_rl_mask.py --out runs/ss_rl/${TAG}_smoke --ckpt-dir runs/ss_rl/${TAG}_smoke $COMMON --steps 2 --eval-every 2 --eval-n 2 --save-every 1000000 $EXTRA \
  || { echo "SMOKE FAILED ($(date))"; exit 7; }
echo "smoke passed ($(date))"
$PY scripts/ss_rl_mask.py --out runs/ss_rl/$TAG --ckpt-dir /l/users/daniil.orel/wmworld_rl/$TAG $COMMON --steps 1500 --eval-every 100 --eval-n 48 --save-every 100 $EXTRA
echo "exit=$? end=$(date)"
