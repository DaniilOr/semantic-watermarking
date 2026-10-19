#!/bin/bash
#SBATCH --job-name=ss_smoke2
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
S="$PY scripts/steered_semstamp.py --run-name ss_smoke --limit 2 --max-new 160"
echo "##### premise check #####"
$S --arm steered_sent --premise-check --premise-prompts 2 --premise-calls 8 --premise-betas 0 1 2 4 8 || exit 1
echo "##### gen/detect agreement (sentence arms) #####"
$PY - <<'PYEOF2'
import json, glob, sys
sys.path.insert(0, ".")
from wmworld import semwm
emb = semwm.Embedder("/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282", device="cuda")
key = semwm.Key.shipped(); pl = semwm.Hyperplanes.for_key(key, emb.dim)
for f in sorted(glob.glob("runs/ss_smoke/generations.*.jsonl")):
    for ln in open(f):
        g = json.loads(ln); tr = (g.get("cost") or {}).get("sentence_trace")
        if not tr: continue
        sents = semwm.split_sentences(g["text"]); bins = [pl.bin_of(p) for p in pl.projections(emb.encode(sents))]
        gen_bins = [t["bin"] for t in tr]
        n = min(len(bins), len(gen_bins)); agree = sum(1 for i in range(n) if bins[i] == gen_bins[i])
        print(f"{g['arm']:20s} gen sentences={len(gen_bins)} detector sentences={len(bins)} first-{n} bins agree={agree}/{n} in_mask(gen)={sum(t['in_mask'] for t in tr)}/{len(tr)}")
PYEOF2
echo "##### detection end to end #####"
$PY scripts/semstamp_detect.py --marked runs/ss_smoke --null runs/ss_smoke --device cuda --alt-keys 2 --out runs/ss_smoke/detect.json || exit 1
echo "end=$(date)"
