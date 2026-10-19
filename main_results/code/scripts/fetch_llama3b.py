"""Re-download Llama-3.2-3B-Instruct at the pinned revision into the home HF cache (/l is unavailable)."""
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
rev = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["revision"]
p = snapshot_download("meta-llama/Llama-3.2-3B-Instruct", revision=rev, cache_dir="/home/daniil.orel/.cache/huggingface/hub",
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
print("snapshot:", p)
print("files:", sorted(x.name for x in Path(p).iterdir()))
