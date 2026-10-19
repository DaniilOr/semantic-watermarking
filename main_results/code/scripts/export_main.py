"""Self-contained snapshot of the main results: code, prompts and the generations behind wm_main_results (copies; the
working tree is not touched).  Re-runnable: overwrites the snapshot with the current files.
Usage: python scripts/export_main.py OUT_DIR
  OUT_DIR/RESULTS.md                    (scripts/main_results.py)
  OUT_DIR/code/                         wmworld package, scripts, slurm, configs, tests, docs, assets/*.json
  OUT_DIR/prompts/<bench>/prompts.jsonl
  OUT_DIR/generations/<run>/<arm>.jsonl  status-ok records of the main arms, one file per arm (shards merged, ids unique)
"""

import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"base", "clean_seed1", "semstamp", "chain_reference", "kgw", "ksemstamp", "chain_kmeans", "chain_spread", "chain_protect", "chain_spread_protect", "chain_beta05"}
ATTACKS = ["para_light", "para_strong", "pegasus", "pegasus_bigram", "parrot", "parrot_bigram", "parrotss", "parrotss_bigram"]
MAIN = ["sp_bench_llama3b", "sp_bench_llama1b", "sp_bench_qwen08b_nl", "sp_bench_llama3b_base", "sp_bench_llama1b_base",
        "sp_bench_qwen08b_base_nl", "ob_bench", "ob_bench_llama1b", "ob_bench_qwen08b_nl", "ob_bench_llama3b_base",
        "ob_bench_llama1b_base", "ob_bench_qwen08b_base_nl", "semstamp_setting_opt13b_ss", "semstamp_setting_llama3b_base"]
OTHER = ["it_bench_llama3b", "it_bench_llama1b", "it_bench_qwen08b", "it_fix_llama3b", "ks_setting_opt13b_ss", "ks_sp_bench_llama3b",
         "ks_sp_bench_llama3b_bs", "ks_sp_bench_llama1b", "ks_sp_bench_qwen08b_nl"]
HUMAN = ["sp_bench_null", "ob_bench_null", "it_bench_null"]          # human-written texts that set the detection thresholds


def export_run(run, out, keep_all=False):
    src = ROOT / "runs" / run
    files = sorted(src.glob("generations*.jsonl"))
    if not files:
        return 0
    by_arm = {}
    for f in files:
        for l in open(f, encoding="utf-8"):
            if not l.strip().endswith("}"):
                continue
            r = json.loads(l)
            arm = r.get("arm", "human")
            if (not keep_all and arm not in ARMS) or r.get("status", "ok") != "ok":
                continue
            by_arm.setdefault(arm, OrderedDict())[r.get("id") or len(by_arm.get(arm, ()))] = r
    d = out / "generations" / run
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    n = 0
    for arm, recs in sorted(by_arm.items()):
        with open(d / f"{arm}.jsonl", "w", encoding="utf-8") as g:
            for r in recs.values():
                g.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    code = out / "code"; code.mkdir(exist_ok=True)
    for d in ("wmworld", "scripts", "slurm", "configs", "tests", "docs"):
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", "--exclude", "*.bak*", str(ROOT / d), str(code)], check=True)
    (code / "assets").mkdir(exist_ok=True)
    for f in (ROOT / "assets").glob("*.json"):
        shutil.copy2(f, code / "assets" / f.name)
    for f in ("README.md",):
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, code / f)
    for b in ("sp_bench", "ob_bench", "it_bench"):
        (out / "prompts" / b).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data" / b / "prompts.jsonl", out / "prompts" / b / "prompts.jsonl")
    total = 0
    for run in MAIN + ["kgw_" + m for m in MAIN] + OTHER:
        for r in [run] + [f"{run}_{a}" for a in ATTACKS]:
            n = export_run(r, out)
            if n:
                print(f"{r:45s} {n:6d} records", flush=True); total += n
    for run in HUMAN:
        n = export_run(run, out, keep_all=True); total += n
        print(f"{run:45s} {n:6d} records (human)", flush=True)
    print(f"total {total} records -> {out}")


if __name__ == "__main__":
    main()
