#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# SemStamp-setting check: rerun of the OPT push + margin arm.  The first run failed on every text (CUDA index assert: the
# boundary mask had the tokenizer's 50,265 entries, OPT's logits 50,272; fixed in semwm.boundary_token_mask 2026-09-30;
# failed records in runs/semstamp_setting_opt13b_ss/failed_vocab_20260930).  Same prompts, decoding and arm as before.
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEC="--prompts sp_bench --ob-domains sp_realnews --plain semstamp --temperature 0.7 --top-p 1.0 --top-k 0 --rep-penalty 1.05 --min-new 195 --max-new 205 --W 16"
ARM="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/steered_semstamp.py --run-name semstamp_setting_smoke_opt13b_ss --target opt13b_ss $DEC --arm-label chain_reference $ARM --limit 2 \
  > slurm_logs/opt_push_fix_smoke.log 2>&1 || { echo "SMOKE FAILED"; cat slurm_logs/opt_push_fix_smoke.log; exit 6; }
$PY scripts/smoke_check.py runs/semstamp_setting_smoke_opt13b_ss || { echo "SMOKE: failed texts"; exit 6; }
P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/steered_semstamp.py --run-name semstamp_setting_opt13b_ss --target opt13b_ss $DEC --arm-label chain_reference $ARM --shard $k --num-shards 6 > slurm_logs/opt_push_fix_$k.log 2>&1 & P+=($!); done; done
FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
$PY -c "
import glob, json, sys
n = bad = 0
for f in glob.glob('runs/semstamp_setting_opt13b_ss/generations.chain_reference.shard*.jsonl'):
    for l in open(f):
        r = json.loads(l); n += 1; bad += r.get('status', 'ok') != 'ok'
print(f'{n} texts, {bad} failed'); sys.exit(1 if bad or n < 1000 else 0)" || FAIL=1
[ $FAIL = 0 ] || { echo "RERUN INCOMPLETE ($(date))"; exit 8; }
echo "end=$(date)"
