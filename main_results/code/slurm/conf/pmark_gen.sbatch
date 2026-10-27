#!/bin/bash
#SBATCH --job-name=pmark_gen
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# PMark generation (user go 2026-10-20: continuation + instruct, online PMark; smoke 328510 ok, alignment 13/13).
# Same prompts / decoding as each setting's other arms; 6 shards = 3 per GPU; resumable.  Order: SemStamp's datasets,
# our benchmark, OPT in SemStamp's setting, instruct tasks.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
GEN="$PY scripts/pmark_arm.py gen --arm semstamp --arm-label pmark"
OPT="--plain semstamp --temperature 0.7 --top-p 1.0 --top-k 0 --rep-penalty 1.05"
lane() { NAME=$1; shift; P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $GEN --run-name $NAME "$@" --shard $k --num-shards 6 > slurm_logs/pmark_gen_${NAME}_$k.log 2>&1 & P+=($!); done; done
  for p in "${P[@]}"; do wait $p; done
  $PY -c "
import glob, json
n = bad = 0
for f in glob.glob('runs/$NAME/generations.pmark.shard*.jsonl'):
    for l in open(f):
        r = json.loads(l); n += 1; bad += r.get('status', 'ok') != 'ok'
print('$NAME: %d texts, %d failed' % (n, bad))"
  echo "$NAME done $(date)"; }
lane pm_sp_bench_llama3b --target llama3b --prompts sp_bench --ob-domains sp_realnews sp_booksum --max-new 200 --W 16
lane pm_ob_bench --target llama3b --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16
lane pm_setting_opt13b_ss --target opt13b_ss --prompts sp_bench --ob-domains sp_realnews $OPT --min-new 195 --max-new 205 --W 16
lane pm_it_bench_llama3b --target llama3b --prompts it_bench --ob-domains it_dolly it_writingprompts it_ifeval it_gsm8k --max-new 512 --min-new 0 --W 16 --eot-at-boundary
echo "end=$(date)"
