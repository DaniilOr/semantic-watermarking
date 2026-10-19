#!/bin/bash
#SBATCH --job-name=ob_smoke
#SBATCH --time=01:00:00
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
# Amendment 17 smoke: benchmark prompts, 1 per dataset, 200 tokens; GRPO-step-400 adapter chain (mask-indexed) + reference chain.
echo "host=$(hostname) start=$(date)"
rm -rf runs/ob_smoke
C="$PY scripts/steered_semstamp.py --run-name ob_smoke --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --per-domain 1 --max-new 200 --W 16"
$C --arm base || exit 1
$C --arm chain --unit sent --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapter-index mask --adapters /l/users/daniil.orel/wmworld_rl/v2/adapters_step400.pt --arm-label chain_grpo400 || exit 1
$C --arm chain --unit sent --block 4 --best-of 16 --steer marginal --beta 1 --stop-margin 0.02 --arm-label chain_reference || exit 1
$PY - <<'PYEOF'
import json, glob, sys
sys.path.insert(0, "scripts")
from api_judge import load_prompts
from wmworld.config import load_config
from pathlib import Path
pr = load_prompts(load_config("configs/pilot.yaml"), Path("runs/ob_smoke"))
for f in sorted(glob.glob("runs/ob_smoke/generations.*.jsonl")):
    for ln in open(f):
        g = json.loads(ln); c = g["cost"]
        print(f"--- {g['arm']} [{g['domain']}] status={g['status']} err={g.get('error')} tokens={g['visible_tokens']} units={c.get('n_units')} in-mask={c.get('in_mask_rate')} wall={c['wall_seconds']:.0f}s judge-prompt-found={g['prompt_id'] in pr}")
        print("   " + g["text"][:300].replace("\n", " "))
PYEOF
echo "end=$(date)"
