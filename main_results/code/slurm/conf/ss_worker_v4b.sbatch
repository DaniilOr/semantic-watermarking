#!/bin/bash
#SBATCH --job-name=worker_v4b
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
# Replaces worker_v4 (279188) so no GPU idles after the corpus.  Phase 1: finish the push-distillation corpus on 3 GPUs
# (9 shards, resumable).  Phase 2, one lane per GPU (each GRPO run with a 2-step smoke first):
#   GPU 0: v4b_para_bigjudge    = v4a + paraphrase reward (gemma-3-12b-it paraphraser, w_para 1)
#   GPU 1: distillation fine-tune -> v4c_distill_bigjudge
#   GPU 2: v4d_bigjudge_klcap   = v4a + hard drift cap: sentence log(pi/base) > 3 nats -> reward - 2
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
P="$PY scripts/steered_semstamp.py --arm chain --unit sent --best-of 16 --W 16 --steer marginal --beta 1 --prompts ss_train_train --run-name ss_corpus_push --per-domain 200 --log-cands runs/ss_corpus_push/cands --num-shards 9"
for g in 0 1 2; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $P --shard $k > slurm_logs/ss_corpus_push9b_${k}.log 2>&1 & done; done; wait
echo "push corpus done ($(date))"
RL="$PY scripts/ss_rl_mask.py --lr 2e-4 --group 32 --beta 0.2 --stop-patience 5 --stale-rule any --judge api --judge-model google/gemma-3-27b-it"
grpo() {  # grpo <gpu> <tag> [extra...]
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $RL --out runs/ss_rl/${tag}_smoke --ckpt-dir runs/ss_rl/${tag}_smoke --steps 2 --eval-every 2 --eval-n 2 --save-every 1000000 "$@" > slurm_logs/rl_${tag}_smoke.log 2>&1 \
    || { echo "$tag SMOKE FAILED ($(date))"; return 7; }
  echo "$tag smoke passed ($(date))"
  CUDA_VISIBLE_DEVICES=$gpu $RL --out runs/ss_rl/$tag --ckpt-dir /l/users/daniil.orel/wmworld_rl/$tag --steps 1500 --eval-every 100 --eval-n 48 --save-every 100 "$@" > slurm_logs/rl_${tag}.log 2>&1
  echo "$tag exit=$? ($(date))"
}
lane_distill() {
  CUDA_VISIBLE_DEVICES=${G[1]} $PY scripts/ss_rft_lora.py --train runs/ss_corpus_push --accept all --rounds 1 --out runs/ss_distill_push --ckpt-dir /l/users/daniil.orel/wmworld_rl/distill_push > slurm_logs/ss_distill_push_sft.log 2>&1 \
    || { echo "distillation FAILED ($(date))"; return 8; }
  echo "distillation done ($(date))"
  grpo ${G[1]} v4c_distill_bigjudge --init-adapters /l/users/daniil.orel/wmworld_rl/distill_push/adapters_round1.pt
}
grpo ${G[0]} v4b_para_bigjudge --para-model google/gemma-3-12b-it --w-para 1.0 &
lane_distill &
grpo ${G[2]} v4d_bigjudge_klcap --kl-cap 3 --kl-cap-penalty 2 &
wait
echo "end=$(date)"
