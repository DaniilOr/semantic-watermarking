#!/bin/bash
#SBATCH --job-name=w2_ifeval
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
# instruction following with verifiable constraints: 200 prompts x (clean, chosen); official checker; then detection
# 3 shards share the ONE GPU that Slurm allocated (CUDA_VISIBLE_DEVICES is inherited, never overridden):
# 3 shards share ONE GPU: single-sequence decoding of two ~1B models leaves an A100 mostly idle (~8 GB per process)
PIDS=""
$PY -m wmworld.cli ifeval --config $CFG --arms base,$ARM --key-set confirmation --limit 200 --budget 1024 --run-name ifeval_cross --shard 0 --num-shards 3 > slurm_logs/ifeval_cross_shard0.log 2>&1 &
PIDS="$PIDS $!"
$PY -m wmworld.cli ifeval --config $CFG --arms base,$ARM --key-set confirmation --limit 200 --budget 1024 --run-name ifeval_cross --shard 1 --num-shards 3 > slurm_logs/ifeval_cross_shard1.log 2>&1 &
PIDS="$PIDS $!"
$PY -m wmworld.cli ifeval --config $CFG --arms base,$ARM --key-set confirmation --limit 200 --budget 1024 --run-name ifeval_cross --shard 2 --num-shards 3 > slurm_logs/ifeval_cross_shard2.log 2>&1 &
PIDS="$PIDS $!"
FAIL=0; for P in $PIDS; do wait $P || FAIL=1; done
echo "ifeval_cross: all shards finished, fail=$FAIL ($(date))"
$PY -m wmworld.cli ifeval --config $CFG --arms base,$ARM --key-set confirmation --limit 200 --budget 1024 --run-name ifeval_cross --evaluate-only
$PY scripts/posstats_dump.py --config $CFG --run runs/ifeval_cross --key-set confirmation || exit 1
$PY scripts/detector_variants.py --null $NULLC --key-set confirmation --run runs/ifeval_cross --prefixes 128,256 --out runs/detector_variants_ifeval.json
echo "end=$(date)"
