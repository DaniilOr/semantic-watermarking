#!/bin/bash
#SBATCH --job-name=pmark_smoke
#SBATCH --time=04:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-54,gpu-05,gpu-02,gpu-51,gpu-55
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/home/daniil.orel/wmworld/slurm_logs/%x_%j.out
set -uo pipefail
cd /home/daniil.orel/wmworld
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/daniil.orel/.cache/huggingface TOKENIZERS_PARALLELISM=false
PY=/home/daniil.orel/anaconda3/envs/wmworld/bin/python
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)" || { echo "CUDA PREFLIGHT FAILED"; exit 4; }
$PY -c "import torch,sys; f,t=torch.cuda.mem_get_info(); print(f\"free GPU memory {f/2**30:.1f} of {t/2**30:.1f} GiB\"); sys.exit(0 if f/2**30 > 30 else 5)" || { echo "GPU NOT EMPTY"; exit 5; }
# PMark smoke (user go 2026-10-20: continuation + instruct, online PMark): 1-2 prompts per dataset in each setting, then the
# online detector on them and on 3 human null texts per dataset; sentence alignment generator vs detector; seconds per text.
echo "host=$(hostname) start=$(date)"
G="$PY scripts/pmark_arm.py gen --arm semstamp --arm-label pmark"
S="$PY scripts/pmark_arm.py score"
$G --run-name pm_smoke_sp --target llama3b --prompts sp_bench --ob-domains sp_realnews sp_booksum --max-new 200 --W 16 --per-domain 2 || echo "GEN SP FAILED"
$G --run-name pm_smoke_ob --target llama3b --prompts ob_bench --ob-domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --W 16 --per-domain 1 || echo "GEN OB FAILED"
$G --run-name pm_smoke_it --target llama3b --prompts it_bench --ob-domains it_dolly it_writingprompts it_ifeval it_gsm8k --max-new 512 --min-new 0 --W 16 --eot-at-boundary --per-domain 1 || echo "GEN IT FAILED"
OPT="--plain semstamp --temperature 0.7 --top-p 1.0 --top-k 0 --rep-penalty 1.05"
$G --run-name pm_smoke_opt --target opt13b_ss --prompts sp_bench --ob-domains sp_realnews $OPT --min-new 195 --max-new 205 --W 16 --per-domain 2 || echo "GEN OPT FAILED"
echo "generation done $(date)"
$S --target llama3b --prompts sp_bench --texts runs/pm_smoke_sp:pmark --null runs/sp_bench_null --domains sp_realnews sp_booksum --max-new 200 --limit 3 --cache runs/pmark_scores/smoke_sp
$S --target llama3b --prompts ob_bench --texts runs/pm_smoke_ob:pmark --null runs/ob_bench_null --domains ob_arxiv ob_booksum ob_wikipedia --max-new 200 --limit 3 --cache runs/pmark_scores/smoke_ob
$S --target llama3b --prompts it_bench --texts runs/pm_smoke_it:pmark runs/it_bench_llama3b:base --null runs/it_bench_null --domains it_dolly it_writingprompts it_ifeval it_gsm8k --max-new 512 --min-new-tokens 1 --limit 2 --cache runs/pmark_scores/smoke_it
$S --target opt13b_ss --prompts sp_bench --texts runs/pm_smoke_opt:pmark --null runs/sp_bench_null --domains sp_realnews --max-new 205 $OPT --limit 3 --cache runs/pmark_scores/smoke_opt
echo "scoring done $(date)"
R="$PY scripts/pmark_arm.py report"
$R --cache runs/pmark_scores/smoke_sp --null sp_bench_null --align runs/pm_smoke_sp:pmark --out runs/pmark_detect_smoke_sp.json
$R --cache runs/pmark_scores/smoke_ob --null ob_bench_null --align runs/pm_smoke_ob:pmark --out runs/pmark_detect_smoke_ob.json
$R --cache runs/pmark_scores/smoke_it --null it_bench_null --model-null-run it_bench_llama3b --align runs/pm_smoke_it:pmark --out runs/pmark_detect_smoke_it.json
$R --cache runs/pmark_scores/smoke_opt --null sp_bench_null --align runs/pm_smoke_opt:pmark --out runs/pmark_detect_smoke_opt.json
$PY - <<'PYEOF'
import glob, json
for run in ("pm_smoke_sp", "pm_smoke_ob", "pm_smoke_it", "pm_smoke_opt"):
    for f in sorted(glob.glob(f"runs/{run}/generations.pmark.shard*.jsonl")):
        for l in open(f):
            r = json.loads(l); c = r.get("cost") or {}
            print(run, r["domain"], r["status"], r.get("error"), "tok", c.get("n_tokens"), "sent", c.get("accepted"), "cand/sent", round(c.get("candidates_per_sentence") or 0, 1),
                  "few", c.get("few_candidates"), "capped", c.get("n_capped"), "s", round(c.get("wall_seconds") or 0, 1))
            print("   ", repr(r["text"][:300]))
PYEOF
echo "end=$(date)"
