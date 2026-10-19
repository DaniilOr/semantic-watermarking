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
# SemStamp-setting check, OPT: both watermark arms regenerated with --join-leading-newline (OPT emits "\n" apart from the
# "." so every line break became a 1-token unit; generator/detector units matched in 2.9% of texts) and the vocab-size fix.
# Old records: runs/semstamp_setting_opt13b_ss/old_newline_units_20260930 (SemStamp), failed_vocab_20260930 (push).
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEC="--prompts sp_bench --ob-domains sp_realnews --plain semstamp --temperature 0.7 --top-p 1.0 --top-k 0 --rep-penalty 1.05 --min-new 195 --max-new 205 --W 16 --join-leading-newline"
ARM_semstamp="--arm semstamp"
ARM_chain_reference="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02"
for L in semstamp chain_reference; do
  V=ARM_$L
  CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/steered_semstamp.py --run-name semstamp_setting_smoke_nl_opt13b_ss --target opt13b_ss $DEC --arm-label $L ${!V} --limit 2 \
    > slurm_logs/opt_rerun_smoke_$L.log 2>&1 || { echo "SMOKE FAILED $L"; cat slurm_logs/opt_rerun_smoke_$L.log; exit 6; }
done
$PY scripts/smoke_check.py runs/semstamp_setting_smoke_nl_opt13b_ss || { echo "SMOKE: failed texts"; exit 6; }
$PY -c "
import glob, json, sys
sys.path.insert(0, '.')
from wmworld import semwm
for f in sorted(glob.glob('runs/semstamp_setting_smoke_nl_opt13b_ss/generations.*.jsonl')):
    for l in open(f):
        r = json.loads(l); c = r['cost']; n_gen = c.get('accepted') or c.get('n_units')
        print('alignment', r['arm'], 'generated units', n_gen, 'detector sentences', len(semwm.split_sentences(r['text'])))"
echo "smoke ok ($(date))"
for L in semstamp chain_reference; do
  V=ARM_$L
  P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/steered_semstamp.py --run-name semstamp_setting_opt13b_ss --target opt13b_ss $DEC --arm-label $L ${!V} --shard $k --num-shards 6 > slurm_logs/opt_rerun_${L}_$k.log 2>&1 & P+=($!); done; done
  FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
  $PY -c "
import glob, json, sys
n = bad = 0
for f in glob.glob('runs/semstamp_setting_opt13b_ss/generations.$L.shard*.jsonl'):
    for l in open(f):
        r = json.loads(l); n += 1; bad += r.get('status', 'ok') != 'ok'
print('$L', n, 'texts,', bad, 'failed'); sys.exit(1 if bad or n < 1000 else 0)" || FAIL=1
  [ $FAIL = 0 ] || { echo "$L INCOMPLETE ($(date))"; exit 8; }
  echo "$L done ($(date))"
done
echo "end=$(date)"
