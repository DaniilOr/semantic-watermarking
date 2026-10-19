#!/bin/bash
#SBATCH --job-name=ss_smoke_c
#SBATCH --time=03:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
[ -f /l/users/daniil.orel/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/config.json ] || { echo "3B NOT VISIBLE FROM THIS NODE"; exit 6; }
echo "host=$(hostname) start=$(date)"
rm -f runs/ss_smoke/generations.bestof4_seg.shard0.jsonl
$PY scripts/steered_semstamp.py --run-name ss_smoke --limit 3 --max-new 200 --arm steered_seg --beta 0 --best-of 4 || exit 1
$PY - <<'PYEOF2'
import json
for ln in open("runs/ss_smoke/generations.bestof4_seg.shard0.jsonl"):
    g = json.loads(ln); c = g["cost"]
    print(f"--- {g['arm']} [{g['domain']}] status={g['status']} tokens={g['visible_tokens']} chunks={c.get('chunks')} cand={c.get('candidates')} depth chosen/mean={c.get('depth_chosen')}/{c.get('depth_mean')} wall={c.get('wall_seconds'):.0f}s")
    if g.get("error"): print("ERROR", g["error"][:300])
    print("   " + g["text"][:700].replace("\n", " ") + "\n")
PYEOF2
echo "end=$(date)"
