#!/bin/bash
#SBATCH --job-name=ss_fetch
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
export HF_HOME=/home/daniil.orel/.cache/huggingface
echo "host=$(hostname) start=$(date)"
$PY - <<'PYEOF'
import hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download
CACHE = "/l/users/daniil.orel/hf_cache/hub"          # user: keep the 3B (and the new embedders) on /l, not /nfs-stor
CACHE_OLD = "/l/users/daniil.orel/hf_cache/hub"     # where the 1B and SemStamp-sbert already live
Path(CACHE).mkdir(parents=True, exist_ok=True)
out = {"cache_dir": CACHE}
p3 = snapshot_download("meta-llama/Llama-3.2-3B-Instruct", cache_dir=CACHE,
                       allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
out["llama3b"] = {"path": p3, "revision": Path(p3).name}
p1 = snapshot_download("meta-llama/Llama-3.2-1B-Instruct", cache_dir=CACHE_OLD, revision="9213176726f574b556790deb65791e0c5aa438b6")
out["llama1b"] = {"path": p1, "revision": Path(p1).name}
sha = lambda p: hashlib.sha256(Path(p, "tokenizer.json").read_bytes()).hexdigest()[:16]
out["tokenizer_sha"] = {"1b": sha(p1), "3b": sha(p3)}
assert out["tokenizer_sha"]["1b"] == out["tokenizer_sha"]["3b"], "1B and 3B tokenizers differ"
cfg = json.load(open(Path(p3, "config.json")))
out["llama3b"].update({"vocab": cfg["vocab_size"], "hidden": cfg["hidden_size"], "layers": cfg["num_hidden_layers"], "dtype": cfg.get("torch_dtype")})
gen = json.load(open(Path(p3, "generation_config.json")))
out["llama3b"]["eos"] = gen.get("eos_token_id")
for repo in ("sentence-transformers/all-mpnet-base-v2", "BAAI/bge-base-en-v1.5", "thenlper/gte-base", "intfloat/e5-base-v2"):
    p = snapshot_download(repo, cache_dir=CACHE, allow_patterns=["*.json", "*.safetensors", "*.txt", "1_Pooling/*", "2_Normalize/*"])
    out[repo] = {"path": p, "revision": Path(p).name}
    print("downloaded", repo, "->", p, flush=True)
Path("assets/ss_models.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
PYEOF
$PY - <<'PYEOF'
import json
from sentence_transformers import SentenceTransformer
m = json.load(open("assets/ss_models.json"))
for repo in ("sentence-transformers/all-mpnet-base-v2", "BAAI/bge-base-en-v1.5", "thenlper/gte-base", "intfloat/e5-base-v2"):
    st = SentenceTransformer(m[repo]["path"], device="cpu")
    v = st.encode(["The harbor was quiet that morning."], normalize_embeddings=True)
    print(repo, "dim", v.shape[1], "modules", [type(x).__name__ for x in st], flush=True)
PYEOF
echo "end=$(date)"
