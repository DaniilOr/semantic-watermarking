#!/bin/bash
#SBATCH --job-name=ifeval_window
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f'free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB'); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY (foreign process on the allocated GPU)"; exit 5; }
CFG=configs/pilot.yaml
echo "allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname) start=$(date)"
# the windowed arm on the SAME 200 IFEval prompts, keys and budget as the earlier run; clean and whole-text texts are reused
PIDS=""
for I in 0 1 2; do $PY -m wmworld.cli ifeval --config $CFG --arms wm_w64_c3 --key-set confirmation --limit 200 --budget 1024 --run-name ifeval_cross --shard $I --num-shards 3 > slurm_logs/ifeval_window_shard$I.log 2>&1 & PIDS="$PIDS $!"; done
FAIL=0; for P in $PIDS; do wait $P || FAIL=1; done; echo "ifeval windowed arm: shards finished, fail=$FAIL ($(date))"
[ $FAIL -eq 0 ] || exit 1
$PY -m wmworld.cli ifeval --config $CFG --arms base,wm_c4,wm_w64_c3 --key-set confirmation --limit 200 --budget 1024 --run-name ifeval_cross --evaluate-only
$PY scripts/posstats_dump.py --config $CFG --run runs/ifeval_cross --key-set confirmation --arms base,wm_w64_c3 --window 64 --stride 32 || exit 1
echo "end=$(date)"
