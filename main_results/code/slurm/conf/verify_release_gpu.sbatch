#!/bin/bash
#SBATCH --job-name=verify_release_gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f'free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB'); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
echo "allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname) start=$(date)"
$PY scripts/verify_release_gpu.py || echo "EQUIVALENCE SCRIPT FAILED"

echo "=== end-to-end smoke of the exported scripts ==="
REL=/home/daniil.orel/wmark-release
W=/home/daniil.orel/wmworld/runs/_release_check/smoke
rm -rf $W; mkdir -p $W
$PY - <<'PYEOF'
import json
ps = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
      "Give me five practical tips for preparing for a job interview.",
      "Explain why the sky is blue to a curious teenager.",
      "Write a product description for a reusable water bottle.",
      "Describe a walk through an autumn forest.",
      "What are the advantages and disadvantages of remote work?"]
with open("/home/daniil.orel/wmworld/runs/_release_check/smoke/prompts.jsonl", "w") as f:
    for i, p in enumerate(ps):
        f.write(json.dumps({"prompt_id": f"p{i}", "prompt": p, "domain": "smoke"}) + "\n")
PYEOF
cd $REL
RC=/l/users/daniil.orel/hf_cache/hub
$PY scripts/make_keys.py --n-keys 3 --n-words 10 --seed 7 --out $W/keys.json || exit 1
$PY scripts/generate.py --prompts $W/prompts.jsonl --run $W/run --arm base --max-new-tokens 128 || exit 1
$PY scripts/generate.py --prompts $W/prompts.jsonl --run $W/null --arm base --seed 1 --max-new-tokens 128 || exit 1
$PY scripts/generate.py --prompts $W/prompts.jsonl --run $W/run --arm wm_b4 --beta 4 --keys $W/keys.json --max-new-tokens 128 --token-map $W/token_map.npz --ref-cache-dir $RC || exit 1
$PY scripts/dump_stats.py --run $W/run --run $W/null --keys $W/keys.json --ref-cache-dir $RC || exit 1
$PY scripts/evaluate_detection.py --null $W/null/posstats.npz --run $W/run --keys $W/keys.json --min-null 5 --prefixes 64 --out $W/detection.json || exit 1
echo "SMOKE PIPELINE: PASS"
echo "end=$(date)"
