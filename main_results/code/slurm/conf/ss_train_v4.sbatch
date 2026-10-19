#!/bin/bash
#SBATCH --job-name=ss_train_v4
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
# Amendment 19 on 2 GPUs.  Lane 0 (GPU 0): v4a_bigjudge = GRPO from the base model, reward judge gemma-3-27b-it (API),
# 32 rollouts, patience 5.  Lane 1 (GPU 1): push distillation (pushed-sample corpus, 600 training prompts, 3 shards;
# then fine-tuning on all candidates), then v4c_distill_bigjudge = GRPO from the distilled adapters, same judge.
# Every stage runs a short smoke first and stops its lane on failure.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
RL="$PY scripts/ss_rl_mask.py --lr 2e-4 --group 32 --beta 0.2 --stop-patience 5 --stale-rule any --judge api --judge-model google/gemma-3-27b-it"
grpo() {  # grpo <gpu> <tag> [extra...]
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $RL --out runs/ss_rl/${tag}_smoke --ckpt-dir runs/ss_rl/${tag}_smoke --steps 2 --eval-every 2 --eval-n 2 --save-every 1000000 "$@" > slurm_logs/rl_${tag}_smoke.log 2>&1 \
    || { echo "$tag SMOKE FAILED ($(date))"; return 7; }
  echo "$tag smoke passed ($(date))"
  CUDA_VISIBLE_DEVICES=$gpu $RL --out runs/ss_rl/$tag --ckpt-dir /l/users/daniil.orel/wmworld_rl/$tag --steps 1500 --eval-every 100 --eval-n 48 --save-every 100 "$@" > slurm_logs/rl_${tag}.log 2>&1
  echo "$tag exit=$? ($(date))"
}
lane1() {
  local gpu=${G[1]}
  P="$PY scripts/steered_semstamp.py --arm chain --unit sent --best-of 16 --W 16 --steer marginal --beta 1 --prompts ss_train_train"
  CUDA_VISIBLE_DEVICES=$gpu $P --run-name ss_corpus_push_smoke --limit 2 --max-new 160 --log-cands runs/ss_corpus_push_smoke/cands > slurm_logs/ss_distill_smoke_gen.log 2>&1 \
    && CUDA_VISIBLE_DEVICES=$gpu $PY scripts/ss_rft_lora.py --train runs/ss_corpus_push_smoke --accept all --rounds 1 --eval-n 2 --out runs/ss_distill_smoke --ckpt-dir runs/ss_distill_smoke > slurm_logs/ss_distill_smoke_sft.log 2>&1 \
    || { echo "distill SMOKE FAILED ($(date))"; return 7; }
  echo "distill smoke passed ($(date))"
  for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$gpu $P --run-name ss_corpus_push --per-domain 200 --log-cands runs/ss_corpus_push/cands --shard $k --num-shards 3 > slurm_logs/ss_corpus_push_${k}.log 2>&1 & done; wait
  echo "push corpus done ($(date))"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/ss_rft_lora.py --train runs/ss_corpus_push --accept all --rounds 1 --out runs/ss_distill_push --ckpt-dir /l/users/daniil.orel/wmworld_rl/distill_push > slurm_logs/ss_distill_push_sft.log 2>&1 \
    || { echo "distillation FAILED ($(date))"; return 8; }
  echo "distillation done ($(date))"
  grpo $gpu v4c_distill_bigjudge --init-adapters /l/users/daniil.orel/wmworld_rl/distill_push/adapters_round1.pt
}
grpo ${G[0]} v4a_bigjudge &
lane1 &
wait
echo "end=$(date)"
