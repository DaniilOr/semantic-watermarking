#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface HF_HUB_CACHE=/l/users/daniil.orel/hf_cache/hub TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# SemStamp's Parrot attack as their code runs it (scripts/sent_parrot_ss.py).  SPECS: space-separated "run|domain,domain"
# (pass through the environment: sbatch --export splits values on commas).
SPECS=${SPECS:?}
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
first=${SPECS%% *}; R0=${first%%|*}; D0=${first#*|}; D0=${D0//,/ }
CUDA_VISIBLE_DEVICES=${G[0]} $PY -c "
import sys; sys.argv=['x']; sys.path.insert(0,'scripts')
from sent_parrot_ss import SemStampParrot
P=SemStampParrot('cuda')
for s in ['The company said on Tuesday that its profits rose by 12 percent in the third quarter, beating analysts expectations.',
          'He was born in Ohio and moved to Chicago when he was ten.']:
    c=P.candidates(s); print('ORIG', s); print('  first ', c[0]); print('  bigram', P.bigram_pick(s, c)); print('  n_cand', len(c))
" || { echo "SMOKE FAILED"; exit 6; }
echo "smoke ok ($(date))"
for spec in $SPECS; do
  RUN=${spec%%|*}; DOMS=${spec#*|}; DOMS=${DOMS//,/ }
  P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/sent_parrot_ss.py --run $RUN --domains $DOMS --shard $k --num-shards 6 > slurm_logs/parrotss_$(basename $RUN)_$k.log 2>&1 & P+=($!); done; done
  FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
  [ $FAIL = 0 ] || { echo "$RUN: A SHARD FAILED ($(date))"; exit 8; }
  echo "$RUN parrotss done ($(date))"
done
echo "end=$(date)"
