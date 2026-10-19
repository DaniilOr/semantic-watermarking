#!/bin/bash
#SBATCH --job-name=ss_adapter_smoke
#SBATCH --time=01:30:00
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
# Amendment 11 code smoke: train adapters on the 6-text smoke corpus, then run both adapter arms on 2 validation prompts.
echo "host=$(hostname) start=$(date)"
rm -rf runs/ss_adapters/smoke runs/ss_adapter_smoke
$PY scripts/ss_train_adapters.py --train runs/ss_corpus_smoke --val runs/ss_corpus_smoke --out runs/ss_adapters/smoke --eval-every 50 --eval-n 20 --warmup 5 || exit 1
S="$PY scripts/steered_semstamp.py --run-name ss_adapter_smoke --prompts ss_train_val --per-domain 1 --limit 2 --arm chain --unit sent --W 16 --block 4 --best-of 16 --stop-margin 0.02 --proposal adapters --adapters runs/ss_adapters/smoke/adapters.pt --max-new 160"
$S || exit 1
$S --select sir || exit 1
$PY - <<'PYEOF'
import json, glob
for f in sorted(glob.glob("runs/ss_adapter_smoke/generations.*.jsonl")):
    for ln in open(f):
        g = json.loads(ln); c = g["cost"]; U = [u for u in c["units"] if u["prev_bin"] is not None]
        print(f"--- {g['arm']} status={g['status']} err={g.get('error')} tokens={g['visible_tokens']} units={c['n_units']} in-mask={c['in_mask_rate']} "
              f"cand/unit={sum(u['n_drawn'] for u in U)/max(1,len(U)):.2f} sir_ess={[round(u.get('sir_ess',0),2) for u in U if 'sir_ess' in u][:6]} wall={c['wall_seconds']:.0f}s")
        print("   " + g["text"][:500].replace("\n", " "))
PYEOF
echo "end=$(date)"
