#!/bin/bash
#SBATCH --job-name=window_gen
#SBATCH --time=08:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY (foreign process on the allocated GPU)"; exit 5; }
CFG=configs/pilot.yaml
echo "allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname) start=$(date)"
$PY scripts/window_selfcheck.py || { echo "WINDOW SELFCHECK FAILED - not generating"; exit 3; }
$PY -m wmworld.cli generate --config $CFG --stage window_dev --run-name window_dev --dry-run || exit 1
PIDS=""
for I in 0 1 2; do $PY -m wmworld.cli generate --config $CFG --stage window_dev --run-name window_dev --shard $I --num-shards 3 > slurm_logs/window_dev_shard$I.log 2>&1 & PIDS="$PIDS $!"; done
FAIL=0; for P in $PIDS; do wait $P || FAIL=1; done; echo "window_dev: shards finished, fail=$FAIL ($(date))"
[ $FAIL -eq 0 ] || exit 1
$PY - <<'PYEOF'
import json
from pathlib import Path
rows = [l for f in Path("runs/tradeoff").glob("generations*.jsonl") for l in open(f) if json.loads(l)["arm"] == "base"]
Path("runs/window_dev/generations.base_shared.jsonl").write_text("".join(rows)); print("linked", len(rows), "clean texts")
PYEOF
$PY scripts/posstats_dump.py --config $CFG --run runs/window_dev --key-set len10 --window 64 --stride 32 || exit 1
$PY scripts/posstats_dump.py --config $CFG --run runs/window_dev --key-set len10 || exit 1
$PY scripts/posstats_dump.py --config $CFG --run runs/strength_fill --run runs/keylen10 --key-set len10 --window 64 --stride 32 || exit 1
echo "end=$(date)"
