#!/bin/bash
#SBATCH --job-name=distill_token
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# Amendment 20: token-level push distillation.  Teacher: 3 shards on the GPU (all training contexts of runs/ss_corpus,
# one logged plain candidate each); then 2 epochs of KL(push || adapter) over the top-p nucleus at every token.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for k in 0 1 2; do $PY scripts/ss_distill_token.py teacher --shard $k --num-shards 3 --out runs/ss_distill_token/teacher > slurm_logs/distill_token_teacher_$k.log 2>&1 & done; wait
echo "teacher done ($(date))"
$PY scripts/ss_distill_token.py train --teacher runs/ss_distill_token/teacher --out runs/ss_distill_token --ckpt-dir /l/users/daniil.orel/wmworld_rl/distill_token --epochs 2 > slurm_logs/distill_token_train.log 2>&1
echo "train exit=$? end=$(date)"
