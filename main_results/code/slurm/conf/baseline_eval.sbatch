#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
echo "host=$(hostname) start=$(date)"
# give every baseline run the SAME clean texts our arms were compared against, so the judge has its reference side
$PY - <<'PYEOF'
import json
from pathlib import Path
rows = [l for f in Path("runs/conf_cross").glob("generations*.jsonl") for l in open(f) if json.loads(l)["arm"] == "base"]
for r in ("baseline_kgw", "baseline_kgw_selfhash", "baseline_semstamp"):
    d = Path("runs") / r
    if d.exists():
        (d / "generations.base_shared.jsonl").write_text("".join(rows))
        print(f"{r}: linked {len(rows)} clean held-out texts")
PYEOF
echo "##### DETECTION (1% FPR on the same clean null) #####"
$PY scripts/baselines_detect.py --method kgw --seeding-scheme simple_1 --marked runs/baseline_kgw --out runs/detect_baseline_kgw.json
$PY scripts/baselines_detect.py --method kgw --seeding-scheme selfhash --marked runs/baseline_kgw_selfhash --out runs/detect_baseline_kgw_selfhash.json
$PY scripts/baselines_detect.py --method semstamp --marked runs/baseline_semstamp --out runs/detect_baseline_semstamp.json
echo "end=$(date)"
