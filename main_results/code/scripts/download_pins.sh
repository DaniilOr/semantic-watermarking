#!/bin/bash
# One-time pinned downloads (model snapshot into the HF cache, datasets into data/raw, IFEval evaluator into third_party).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODEL_SHA=2fc06364715b967f1860aea9cf38778875588b17
$PY - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
p = snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                      allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE", "README.md"])
print("model snapshot:", p)
R = "data/raw"
print(hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset",
                      revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", local_dir=R + "/dolly"))
print(hf_hub_download("allenai/c4", "realnewslike/c4-validation.00000-of-00001.json.gz", repo_type="dataset",
                      revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2", local_dir=R + "/c4_realnewslike"))
print(hf_hub_download("google/IFEval", "ifeval_input_data.jsonl", repo_type="dataset",
                      revision="966cd89545d6b6acfd7638bc708b98261ca58e84", local_dir=R + "/ifeval"))
EOF
mkdir -p data/raw/writingprompts && cd data/raw/writingprompts
if [ ! -f writingPrompts.tar.gz ]; then
  curl -sL -o writingPrompts.tar.gz https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz
fi
echo "4b8d0415c450fe1ceadec667bdf73c185268bc3d9e1f6ddf64edac0481cb71f4  writingPrompts.tar.gz" | sha256sum -c -
tar xzf writingPrompts.tar.gz
cd ../../..
mkdir -p third_party && cd third_party
if [ ! -d google_research_ifeval ]; then
  mkdir google_research_ifeval && cd google_research_ifeval && git init -q
  git remote add origin https://github.com/google-research/google-research.git
  git sparse-checkout init --cone && git sparse-checkout set instruction_following_eval
  git fetch --depth 1 --filter=blob:none origin e6890f85757dd84e27ca6df2dd30651dafad28e0 && git checkout -q FETCH_HEAD
fi
echo "downloads complete"
