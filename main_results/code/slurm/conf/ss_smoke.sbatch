#!/bin/bash
#SBATCH --job-name=ss_smoke
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
rm -rf runs/ss_smoke
S="$PY scripts/steered_semstamp.py --run-name ss_smoke --limit 2 --max-new 160"
echo "##### base #####";                 $S --arm base || exit 1
echo "##### base assist #####";          $S --arm base --draft assist || exit 1
echo "##### steered_seg #####";          $S --arm steered_seg || exit 1
echo "##### steered_seg assist #####";   $S --arm steered_seg --draft assist || exit 1
echo "##### semstamp #####";             $S --arm semstamp || exit 1
echo "##### steered_sent #####";         $S --arm steered_sent || exit 1
echo "##### semstamp draft-loop #####";  $S --arm semstamp --draft loop || exit 1
echo "##### steered_sent draft-loop #####"; $S --arm steered_sent --draft loop || exit 1
echo "##### resume check (expect 0 new) #####"
N0=$(cat runs/ss_smoke/generations.steered_seg.shard0.jsonl | wc -l); $S --arm steered_seg || exit 1; N1=$(cat runs/ss_smoke/generations.steered_seg.shard0.jsonl | wc -l); echo "records before/after rerun: $N0/$N1"
echo "##### premise check #####"
$S --arm steered_sent --premise-check --premise-prompts 2 --premise-calls 8 || exit 1
echo "##### detection end to end #####"
$PY scripts/semstamp_detect.py --marked runs/ss_smoke --null runs/ss_smoke --device cuda --alt-keys 2 --out runs/ss_smoke/detect.json || exit 1
$PY - <<'PYEOF'
import json, glob
for f in sorted(glob.glob("runs/ss_smoke/generations.*.jsonl")):
    for ln in open(f):
        g = json.loads(ln); c = g.get("cost", {})
        keep = {k: c.get(k) for k in ("n_tokens","trials","accepted","maxed_out","left_region_after_verify","n_capped","target_forward_calls","drafter_forward_calls","n_embed_calls","wall_seconds") if k in c}
        print(f"--- {g['arm']:22s} status={g['status']} tokens={g['visible_tokens']} {keep}")
        if g.get("error"): print("ERROR:", g["error"][:300])
        print("   ", (g["text"] or "")[:260].replace("\n", " "))
PYEOF
echo "end=$(date)"
