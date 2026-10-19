#!/bin/bash
#SBATCH --job-name=w2_long
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f'free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB'); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY (foreign process on the allocated GPU)"; exit 5; }
CFG=configs/pilot.yaml
NULL=runs/null_calibration/eval/posstats_randomwords.npz
NULLC=runs/null_calibration/eval/posstats_confirmation.npz
ARM=$(cat runs/chosen_arm.txt 2>/dev/null || true)
echo "allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "host=$(hostname) start=$(date) chosen arm=${ARM:-<none yet>}"
[ -n "$ARM" ] && [ -f $NULLC ] || { echo "selection or confirmation null missing"; exit 1; }
# long texts with the chosen delivery: the same key throughout vs four keys rotating every 128 tokens
ARMS="base,$ARM,${ARM}_rot"
$PY -m wmworld.cli generate --config $CFG --stage long_c --run-name long_c --arms $ARMS --dry-run || exit 1
# 3 shards share the ONE GPU that Slurm allocated (CUDA_VISIBLE_DEVICES is inherited, never overridden):
# 3 shards share ONE GPU: single-sequence decoding of two ~1B models leaves an A100 mostly idle (~8 GB per process)
PIDS=""
$PY -m wmworld.cli generate --config $CFG --stage long_c --run-name long_c --arms $ARMS --shard 0 --num-shards 3 > slurm_logs/long_c_shard0.log 2>&1 &
PIDS="$PIDS $!"
$PY -m wmworld.cli generate --config $CFG --stage long_c --run-name long_c --arms $ARMS --shard 1 --num-shards 3 > slurm_logs/long_c_shard1.log 2>&1 &
PIDS="$PIDS $!"
$PY -m wmworld.cli generate --config $CFG --stage long_c --run-name long_c --arms $ARMS --shard 2 --num-shards 3 > slurm_logs/long_c_shard2.log 2>&1 &
PIDS="$PIDS $!"
FAIL=0; for P in $PIDS; do wait $P || FAIL=1; done
echo "long_c: all shards finished, fail=$FAIL ($(date))"
$PY scripts/posstats_dump.py --config $CFG --run runs/long_c --key-set randomwords --extra steering_decoys || exit 1
$PY scripts/detector_variants.py --null $NULL --run runs/long_c --prefixes 128,256,384,512,768 --config $CFG --rotation --out runs/detector_variants_long_c.json
echo "end=$(date)"
