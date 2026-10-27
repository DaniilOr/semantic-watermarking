#!/bin/bash
#SBATCH --job-name=pmark_it
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
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# Instruct PMark (replaces 329853, then 330534).  328727 lost 1,383 of 3,860 texts to CUDA OOM (3 generators per GPU at 512 tokens;
# failed records moved to runs/pm_it_bench_llama3b/failed_oom_20261021).  Step 1 in parallel: GPU 0 regenerates them with
# 2 generators; GPU 1 scores the human null and the plain model's texts (2 lanes).  Step 2: the usual pmark_det.sbatch
# steps for "it" (attacks on the new texts, clean scoring, attacked copies, report); everything resumes.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
P=()
for k in 0 1; do CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/pmark_arm.py gen --arm semstamp --arm-label pmark --run-name pm_it_bench_llama3b --target llama3b --prompts it_bench \
  --ob-domains it_dolly it_writingprompts it_ifeval it_gsm8k --max-new 512 --min-new 0 --W 16 --eot-at-boundary --shard $k --num-shards 2 \
  > slurm_logs/pmark_refill_it_$k.log 2>&1 & P+=($!); done
for k in 0 1; do CUDA_VISIBLE_DEVICES=${G[1]} $PY scripts/pmark_arm.py score --target llama3b --prompts it_bench --texts runs/it_bench_llama3b:base --null runs/it_bench_null \
  --domains it_dolly it_writingprompts it_ifeval it_gsm8k --max-new 512 --max-rows 128 --min-new-tokens 1 --cache runs/pmark_scores/it --tag nullbase \
  --shard $k --num-shards 2 > slurm_logs/pmark_det_it_nullbase_$k.log 2>&1 & P+=($!); done
for p in "${P[@]:0:2}"; do wait $p; done
$PY -c "
import glob, json
n = bad = 0
for f in glob.glob('runs/pm_it_bench_llama3b/generations.pmark.shard*.jsonl'):
    for l in open(f):
        r = json.loads(l); n += 1; bad += r.get('status', 'ok') != 'ok'
print('pm_it_bench_llama3b after refill: %d texts, %d failed' % (n, bad))"
echo "refill done $(date)"
kill "${P[@]:2}" 2>/dev/null; for p in "${P[@]:2}"; do wait $p; done      # the 2 scoring lanes stop here; the 4-lane step resumes them
echo "2-lane null + plain scoring stopped $(date)"
SETTINGS=it bash slurm/conf/pmark_det.sbatch
echo "end=$(date)"
