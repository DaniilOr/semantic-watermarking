#!/bin/bash
#SBATCH --job-name=worker_v4
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:3
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# 3 GPUs after the GRPO+push generation and the run-3 attacks: GPU 0 runs the run-3 benchmark detection (~10 min) and
# then 3 corpus shards; GPUs 1-2 run 6 corpus shards from the start (push-distillation corpus, 9 shards in total);
# then the distillation fine-tune and v4c_distill_bigjudge on GPU 1.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
P="$PY scripts/steered_semstamp.py --arm chain --unit sent --best-of 16 --W 16 --steer marginal --beta 1 --prompts ss_train_train --run-name ss_corpus_push --per-domain 200 --log-cands runs/ss_corpus_push/cands --num-shards 9"
for g in 1 2; do for l in 0 1 2; do k=$(((g - 1) * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $P --shard $k > slurm_logs/ss_corpus_push9_${k}.log 2>&1 & done; done
( CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/ob_detect.py --out runs/ob_detect_v3.json > slurm_logs/ob_detect_v3_worker.log 2>&1; echo "detection run-3 exit=$? ($(date))"
  for k in 6 7 8; do CUDA_VISIBLE_DEVICES=${G[0]} $P --shard $k > slurm_logs/ss_corpus_push9_${k}.log 2>&1 & done; wait ) &
wait
echo "push corpus done ($(date))"
CUDA_VISIBLE_DEVICES=${G[1]} $PY scripts/ss_rft_lora.py --train runs/ss_corpus_push --accept all --rounds 1 --out runs/ss_distill_push --ckpt-dir /l/users/daniil.orel/wmworld_rl/distill_push > slurm_logs/ss_distill_push_sft.log 2>&1 || { echo "distillation FAILED ($(date))"; exit 8; }
echo "distillation done ($(date))"
RL="$PY scripts/ss_rl_mask.py --lr 2e-4 --group 32 --beta 0.2 --stop-patience 5 --stale-rule any --judge api --judge-model google/gemma-3-27b-it"
CUDA_VISIBLE_DEVICES=${G[1]} $RL --out runs/ss_rl/v4c_distill_bigjudge --ckpt-dir /l/users/daniil.orel/wmworld_rl/v4c_distill_bigjudge --steps 1500 --eval-every 100 --eval-n 48 --save-every 100 --init-adapters /l/users/daniil.orel/wmworld_rl/distill_push/adapters_round1.pt > slurm_logs/rl_v4c_distill_bigjudge.log 2>&1
echo "v4c exit=$? end=$(date)"
