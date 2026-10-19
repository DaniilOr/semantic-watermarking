#!/bin/bash
#SBATCH --job-name=long_judge_calib
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
CFG=configs/pilot.yaml
echo "host=$(hostname) start=$(date)"
# calibration of the long-story judge run: clean vs clean on the same prompts, full length, then flaws vs clean variation
$PY - <<'EOF'
import json
from pathlib import Path
rd = Path("runs/long_clean_pair"); rd.mkdir(exist_ok=True)
ref = [json.loads(l) for f in Path("runs/long_story").glob("generations*.jsonl") for l in open(f) if json.loads(l)["arm"] == "base"]
oth = []
for f in Path("runs/long_story_clean2").glob("generations*.jsonl"):
    for l in open(f):
        g = json.loads(l)
        if g["arm"] == "base" and g.get("status", "ok") == "ok":
            g = dict(g); g["arm"] = "clean_seed1"; g["id"] = g["id"] + ".as_arm"; oth.append(g)
(rd / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref + oth))
(rd / "manifest.json").write_text(Path("runs/long_story/manifest.json").read_text())
print("long_clean_pair:", len(ref), "reference clean +", len(oth), "second-draw clean")
EOF
$PY scripts/api_judge.py --mode preference --config $CFG --run runs/long_clean_pair --max-words 2200 --workers 8
$PY scripts/api_pointwise.py --config $CFG --run runs/long_story_clean2 --max-words 2200 --workers 8
$PY scripts/quality_vs_clean.py --clean runs/long_story:base --clean runs/long_story_clean2:base --marked runs/long_story --out runs/quality_vs_clean_long.json
echo "end=$(date)"
