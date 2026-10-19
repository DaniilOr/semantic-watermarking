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
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
# k-SemStamp vs Push + Margin on k-SemStamp's k-means partition (scripts/ksemstamp.py; original code untouched).
# Env (pass through the environment, not --export=VAR=..., which splits on commas):
#   RUN     run name (runs/<RUN>, separate from the main runs)      TARGET  generator key
#   DEC     decoding / prompt flags of the matching main run       DOMS    "domain:partition ..." (partition c4 | booksum)
# Arms: ksemstamp     = SemStamp's rejection sampler on the k-means partition, margin 0.035 (k-SemStamp paper), 100 tries
#       chain_kmeans  = Push + Margin (blocks of 4, cap 16, push beta 1) on the same partition, stop margin 0.035
RUN=${RUN:?}; TARGET=${TARGET:?}; DEC=${DEC:?}; DOMS=${DOMS:?}
echo "host=$(hostname) start=$(date) RUN=$RUN TARGET=$TARGET DOMS=$DOMS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ENC_c4=/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282
ENC_booksum=/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-booksum-sbert/snapshots/e7569f31e4b249b18c043fdaeedbfdccd2028289
ARM_ksemstamp="--arm semstamp --margin 0.035"
ARM_chain_kmeans="--arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.035"
for spec in $DOMS; do
  D=${spec%%:*}; P=${spec#*:}; E=ENC_$P
  for L in ksemstamp chain_kmeans; do
    V=ARM_$L
    CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/ksemstamp.py gen --centroids third_party/semstamp_ref/centroids/$P-cluster_8_centers.pt \
      --run-name ks_smoke_$RUN --target $TARGET $DEC --ob-domains $D --embedder ${!E} --arm-label $L ${!V} --limit 2 \
      > slurm_logs/ks_smoke_${RUN}_${D}_$L.log 2>&1 || { echo "SMOKE FAILED $D $L"; cat slurm_logs/ks_smoke_${RUN}_${D}_$L.log; exit 6; }
  done
done
$PY scripts/smoke_check.py runs/ks_smoke_$RUN || { echo "SMOKE: failed texts"; exit 7; }
echo "smoke ok ($(date))"
for spec in $DOMS; do
  D=${spec%%:*}; P=${spec#*:}; E=ENC_$P
  for L in ksemstamp chain_kmeans; do
    V=ARM_$L
    PIDS=(); for g in 0 1; do for l in 0 1 2; do k=$((g * 3 + l)); CUDA_VISIBLE_DEVICES=${G[$g]} $PY scripts/ksemstamp.py gen \
      --centroids third_party/semstamp_ref/centroids/$P-cluster_8_centers.pt --run-name $RUN --target $TARGET $DEC --ob-domains $D \
      --embedder ${!E} --arm-label $L ${!V} --shard $k --num-shards 6 > slurm_logs/ks_${RUN}_${D}_${L}_$k.log 2>&1 & PIDS+=($!); done; done
    FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
    [ $FAIL = 0 ] || { echo "$D $L: A SHARD FAILED ($(date))"; exit 8; }
    echo "$D $L done ($(date))"
  done
done
echo "end=$(date)"
