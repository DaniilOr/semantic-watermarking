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
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# SemStamp's sentence-level paraphrase attacks (Pegasus, Parrot; plain + bigram variants; scripts/sent_paraphrase_attack.py)
# on existing watermarked texts (arms semstamp, chain_reference).  SPECS: space-separated "run|domain,domain".
SPECS=${SPECS:?}
echo "host=$(hostname) start=$(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for spec in $SPECS; do
  RUN=${spec%%|*}; DOMS=${spec#*|}; DOMS=${DOMS//,/ }
  for ATT in pegasus parrot; do
    P=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/sent_paraphrase_attack.py --run $RUN --attacker $ATT --domains $DOMS --shard $k --num-shards 6 > slurm_logs/sentpara_$(basename $RUN)_${ATT}_$k.log 2>&1 & P+=($!); done; done
    FAIL=0; for p in "${P[@]}"; do wait $p || FAIL=1; done
    [ $FAIL = 0 ] || { echo "$RUN $ATT: A SHARD FAILED ($(date))"; exit 8; }
    echo "$RUN $ATT done ($(date))"
  done
done
echo "end=$(date)"
