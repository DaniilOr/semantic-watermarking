#!/bin/bash
#SBATCH --job-name=baseline_smoke
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
echo "host=$(hostname) start=$(date)"
rm -rf runs/smoke_kgw runs/smoke_kgw_sh runs/smoke_semstamp
echo "##### KGW #####"
$PY scripts/baselines.py --method kgw --run-name smoke_kgw --limit 2 2>&1
echo "##### KGW selfhash #####"
$PY scripts/baselines.py --method kgw --seeding-scheme selfhash --run-name smoke_kgw_sh --limit 2 2>&1
echo "##### SemStamp #####"
$PY scripts/baselines.py --method semstamp --run-name smoke_semstamp --limit 2 2>&1
$PY - <<'PYEOF'
import json
from pathlib import Path
for r in ("smoke_kgw", "smoke_kgw_sh", "smoke_semstamp"):
    for f in Path(f"runs/{r}").glob("generations*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            print(f"\n--- {r} | status={g['status']} tokens={g['visible_tokens']} cost={g.get('cost')}")
            if g.get("error"): print("ERROR:", g["error"][:400])
            print((g["text"] or "")[:320].replace("\n", " "))
PYEOF
echo "end=$(date)"
