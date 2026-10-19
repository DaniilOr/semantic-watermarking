#!/bin/bash
#SBATCH --job-name=ss_distill_push
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
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
# Push distillation (amendment 19): (1) corpus of PUSHED samples -- the sentence chain with the true-shift push (beta 1),
# all 16 candidates per sentence position drawn in one batch and logged, 600 training prompts (200 per domain, the same
# prompts as the plain corpus); (2) mask-indexed LoRA adapters fitted by next-token likelihood on ALL logged candidates
# (up to 8 per position) = distillation of the pushed sampler; RL monitor after.  A 2-prompt smoke of both steps first.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
P="$PY scripts/steered_semstamp.py --arm chain --unit sent --best-of 16 --W 16 --steer marginal --beta 1 --prompts ss_train_train"
CUDA_VISIBLE_DEVICES=${G[0]} $P --run-name ss_corpus_push_smoke --limit 2 --max-new 160 --log-cands runs/ss_corpus_push_smoke/cands > slurm_logs/ss_distill_smoke_gen.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/ss_rft_lora.py --train runs/ss_corpus_push_smoke --accept all --rounds 1 --eval-n 2 --out runs/ss_distill_smoke --ckpt-dir runs/ss_distill_smoke > slurm_logs/ss_distill_smoke_sft.log 2>&1 \
  || { echo "SMOKE FAILED (slurm_logs/ss_distill_smoke_*.log)"; exit 7; }
echo "smoke passed ($(date))"
for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l))
  CUDA_VISIBLE_DEVICES=${G[$g]} $P --run-name ss_corpus_push --per-domain 200 --log-cands runs/ss_corpus_push/cands --shard $k --num-shards 6 > slurm_logs/ss_corpus_push_${k}.log 2>&1 &
done; done; wait
echo "corpus done ($(date))"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/ss_rft_lora.py --train runs/ss_corpus_push --accept all --rounds 1 --out runs/ss_distill_push --ckpt-dir /l/users/daniil.orel/wmworld_rl/distill_push
echo "exit=$? end=$(date)"
