#!/bin/bash
#SBATCH --job-name=baseline_setup
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
export HF_HOME=/home/daniil.orel/.cache/huggingface
echo "host=$(hostname) start=$(date)"
$PY -m pip install --quiet "sentence-transformers>=3.0" 2>&1 | tail -5
$PY -c "import sentence_transformers as s; print('sentence-transformers', s.__version__)"
$PY - <<'PYEOF'
import nltk
for p in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{p}"); print(p, "present")
    except LookupError:
        nltk.download(p, quiet=True); print(p, "downloaded")
from huggingface_hub import snapshot_download
for repo in ("AbeHou/SemStamp-c4-sbert",):
    p = snapshot_download(repo, cache_dir="/l/users/daniil.orel/hf_cache/hub")
    print("downloaded", repo, "->", p)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(p, device="cpu")
v = m.encode(["The harbor was quiet that morning.", "That morning the port was calm."])
import numpy as np
print("embedder ok: dim", v.shape[1], "cos(sim pair) =", float(v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))))
PYEOF
echo "--- KGW import check"
$PY -c "
import sys; sys.path.insert(0, 'third_party/lm-watermarking')
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
print('KGW import ok:', WatermarkLogitsProcessor.__module__)
"
echo "end=$(date)"
