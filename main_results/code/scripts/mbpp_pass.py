"""Functional correctness on MBPP (pass@1, one sample per problem) for the sp_bench runs.

Per run x arm (domain sp_mbpp): the code is the first ```python (or bare ```) block of the text; an unclosed block is
taken to the end of the text and counted as "unclosed"; a text without a fence is run whole.  The program
    test_setup_code + code + the problem's test_list asserts
runs in a fresh python subprocess (timeout 10 s, 2 GB address space, HumanEval-style reliability guard that disables
destructive os / shutil / subprocess calls, cwd = a scratch directory).  The human reference solutions
(runs/sp_bench_null, arm base, domain sp_mbpp) are scored the same way as a harness check.
Reported: pass rate with a Wilson interval, paired counts against the run's "base" arm (problems passed by only one of
the two), shares of texts with no fence / an unclosed fence / generation past the end of turn.
Usage: python scripts/mbpp_pass.py runs/sp_bench_llama1b [runs/sp_bench_llama3b ...]  -> runs/mbpp_pass.json (merged)
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.posstats import wilson  # noqa: E402

FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)(```|\Z)", re.S)
LEAK = re.compile(r"(^|[^\w])(assistant|user)\n\n")
GUARD = r'''
import resource, faulthandler, builtins, os, shutil, subprocess, sys
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
faulthandler.disable()
for _m, _names in ((os, ["kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
                         "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
                         "chroot", "lchown", "chdir"]),
                   (shutil, ["rmtree", "move", "chown"]), (subprocess, ["Popen", "run", "call", "check_output", "check_call"])):
    for _n in _names:
        if hasattr(_m, _n):
            setattr(_m, _n, None)
builtins.exit = builtins.quit = None
'''
OUT = Path(os.environ.get("MBPP_PASS_OUT", "runs/mbpp_pass.json"))


def extract(text: str) -> tuple[str, str]:
    m = FENCE.search(text)
    if not m:
        return text, "nofence"
    return m.group(1), ("unclosed" if m.group(2) != "```" else "ok")


def run_one(job):
    key, program = job
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(GUARD + "\n" + program)
        try:
            r = subprocess.run([sys.executable, str(f)], cwd=d, capture_output=True, timeout=10,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": d, "PYTHONHASHSEED": "0"})
            ok, why = r.returncode == 0, ("" if r.returncode == 0 else (r.stderr.decode(errors="replace").strip().splitlines() or ["?"])[-1][:120])
        except subprocess.TimeoutExpired:
            ok, why = False, "timeout"
    return key, ok, why


def main():
    runs = sys.argv[1:]
    mb = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/raw/mbpp/full-*.parquet"))]).drop_duplicates("task_id")
    probs = {int(r.task_id): r for r in mb.itertuples()}
    items = []                                                              # (run, arm, task_id, text)
    for ln in open("runs/sp_bench_null/generations.jsonl"):
        r = json.loads(ln)
        if r["domain"] == "sp_mbpp":
            items.append(("sp_bench_null", "human", int(r["prompt_id"].split(":")[-1]), r["text"]))
    for run in runs:
        for f in sorted(glob.glob(f"{run}/generations.*.jsonl")):
            for ln in open(f):
                r = json.loads(ln)
                if r.get("domain") == "sp_mbpp" and r.get("status", "ok") == "ok":
                    items.append((Path(run).name, r["arm"], int(r["prompt_id"].split(":")[-1]), r.get("text") or ""))
    seen, jobs, meta = set(), [], {}
    for run, arm, tid, text in items:
        k = (run, arm, tid)
        if k in seen:
            continue
        seen.add(k)
        code, fence = extract(text)
        p = probs[tid]
        prog = (p.test_setup_code or "") + "\n" + code + "\n\n" + "\n".join(p.test_list) + "\n"
        jobs.append(("|".join(map(str, k)), prog)); meta["|".join(map(str, k))] = (fence, bool(LEAK.search(text)))
    print(f"{len(jobs)} programs", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as pool:
        res = {k: (ok, why) for k, ok, why in pool.imap_unordered(run_one, jobs, chunksize=8)}
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    groups = {}
    for k, (ok, why) in res.items():
        run, arm, tid = k.split("|")
        groups.setdefault((run, arm), {})[int(tid)] = (ok, why) + meta[k]
    for (run, arm), g in sorted(groups.items()):
        n, k = len(g), sum(v[0] for v in g.values())
        lo, hi = wilson(k, n)
        base = groups.get((run, "base"), {})
        common = sorted(set(g) & set(base))
        only_arm = sum(g[t][0] and not base[t][0] for t in common); only_base = sum(base[t][0] and not g[t][0] for t in common)
        errs = {}
        for v in g.values():
            if not v[0]:
                e = "timeout" if v[1] == "timeout" else (v[1].split(":")[0] if v[1] else "?")
                errs[e] = errs.get(e, 0) + 1
        rec = {"n": n, "pass": k / n, "ci": [lo, hi], "paired_vs_base": {"n": len(common), "only_this": only_arm, "only_base": only_base},
               "nofence": sum(v[2] == "nofence" for v in g.values()) / n, "unclosed": sum(v[2] == "unclosed" for v in g.values()) / n,
               "past_end_of_turn": sum(v[3] for v in g.values()) / n,
               "pass_closed_fence": (lambda c: sum(g[t][0] for t in c) / len(c) if c else None)([t for t, v in g.items() if v[2] == "ok"]),
               "errors": dict(sorted(errs.items(), key=lambda x: -x[1])[:6])}
        report[f"{run}|{arm}"] = rec
        print(f"{run:22s} {arm:16s} n {n:4d} pass {100*k/n:5.1f}% [{100*lo:.1f},{100*hi:.1f}]  vs base +{only_arm}/-{only_base}  "
              f"closed-fence pass {100*(rec['pass_closed_fence'] or 0):5.1f}%  nofence {100*rec['nofence']:4.1f}%  unclosed {100*rec['unclosed']:4.1f}%  "
              f"past-EOT {100*rec['past_end_of_turn']:4.1f}%  errors {rec['errors']}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
