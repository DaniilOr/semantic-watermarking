#!/bin/bash
#SBATCH --job-name=keylen_b
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
echo "allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "host=$(hostname) start=$(date)"
# ---- keys of 8 words ----
$PY -m wmworld.cli generate --config $CFG --stage keylen8 --run-name keylen8 --dry-run || exit 1
PIDS=""
for I in 0 1 2; do $PY -m wmworld.cli generate --config $CFG --stage keylen8 --run-name keylen8 --shard $I --num-shards 3 > slurm_logs/keylen8_shard$I.log 2>&1 & PIDS="$PIDS $!"; done
FAIL=0; for P in $PIDS; do wait $P || FAIL=1; done; echo "keylen8: shards finished, fail=$FAIL ($(date))"
# the shared clean arm: the base texts of the sweep (same 60 prompts), so detection FPR and both judges have a clean twin
$PY - <<'EOF'
import json
from pathlib import Path
rows = [l for f in Path("runs/tradeoff").glob("generations*.jsonl") for l in open(f) if json.loads(l)["arm"] == "base"]
Path("runs/keylen8/generations.base_shared.jsonl").write_text("".join(rows)); print("linked", len(rows), "clean texts")
EOF
$PY scripts/posstats_dump.py --config $CFG --run runs/null_calibration --key-set len8 || exit 1
$PY scripts/posstats_dump.py --config $CFG --run runs/keylen8 --key-set len8 || exit 1
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_len8.npz --key-set len8 --run runs/keylen8 --variants T_clip05_gate,T_clip1,T --prefixes 128,256,384 --out runs/detector_variants_keylen8.json
# ---- keys of 10 words ----
$PY -m wmworld.cli generate --config $CFG --stage keylen10 --run-name keylen10 --dry-run || exit 1
PIDS=""
for I in 0 1 2; do $PY -m wmworld.cli generate --config $CFG --stage keylen10 --run-name keylen10 --shard $I --num-shards 3 > slurm_logs/keylen10_shard$I.log 2>&1 & PIDS="$PIDS $!"; done
FAIL=0; for P in $PIDS; do wait $P || FAIL=1; done; echo "keylen10: shards finished, fail=$FAIL ($(date))"
# the shared clean arm: the base texts of the sweep (same 60 prompts), so detection FPR and both judges have a clean twin
$PY - <<'EOF'
import json
from pathlib import Path
rows = [l for f in Path("runs/tradeoff").glob("generations*.jsonl") for l in open(f) if json.loads(l)["arm"] == "base"]
Path("runs/keylen10/generations.base_shared.jsonl").write_text("".join(rows)); print("linked", len(rows), "clean texts")
EOF
$PY scripts/posstats_dump.py --config $CFG --run runs/null_calibration --key-set len10 || exit 1
$PY scripts/posstats_dump.py --config $CFG --run runs/keylen10 --key-set len10 || exit 1
$PY scripts/detector_variants.py --null runs/null_calibration/eval/posstats_len10.npz --key-set len10 --run runs/keylen10 --variants T_clip05_gate,T_clip1,T --prefixes 128,256,384 --out runs/detector_variants_keylen10.json
echo "end=$(date)"
