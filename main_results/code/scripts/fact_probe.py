"""Factual-consistency probe (the judge does not fact-check): on the Dolly prompts, how many numbers and named-entity
spans of a marked text are absent from the clean (base) answer to the same prompt, against the clean band (clean_seed1
vs base -- two clean draws also disagree).  Model-free, CPU.  Reported as an extra flaw measure, not a selection rule.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s?(?:%|percent|million|billion|thousand))?")
ENT_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|de|the|and)?\s*[A-Z][a-z]+)*")   # capitalised spans (sentence-initial function words removed via STOP)
STOP = {"The", "This", "These", "Those", "It", "In", "On", "At", "However", "Additionally", "Overall", "For", "As", "If", "When", "While"}


def numbers(t: str) -> set[str]:
    return {re.sub(r"[,\s]", "", m.group(0)).lower() for m in NUM_RE.finditer(t)}


def entities(t: str) -> set[str]:
    return {m.group(0) for m in ENT_RE.finditer(t) if m.group(0) not in STOP}


def novel(a: set[str], ref: set[str]) -> float | None:
    return (len(a - ref) / len(a)) if a else None


def boot(v: list[float], n: int = 2000) -> list[float]:
    v = np.asarray(v); rng = np.random.default_rng(0)
    m = [float(rng.choice(v, len(v)).mean()) for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ss_llama3b")
    ap.add_argument("--domain", default="dolly")
    ap.add_argument("--ref-arm", default="base")
    ap.add_argument("--clean-arm", default="clean_seed1")
    ap.add_argument("--out", default="runs/fact_probe_ss_llama3b.json")
    a = ap.parse_args()
    gens, _ = collect_run_texts(Path(a.run))
    by = defaultdict(dict)
    for g in gens:
        if g.get("domain") == a.domain and g.get("status", "ok") == "ok" and g.get("text"):
            by[g["arm"]][g["prompt_id"]] = g["text"]
    ref = by[a.ref_arm]
    out = {"run": a.run, "domain": a.domain, "ref_arm": a.ref_arm, "arms": {}}
    print(f"{'arm':20s} {'n':>3s} {'novel numbers':>22s} {'novel entities':>22s}")
    for arm in sorted(by):
        if arm == a.ref_arm:
            continue
        rn, re_ = [], []
        for pid, t in by[arm].items():
            if pid not in ref:
                continue
            x = novel(numbers(t), numbers(ref[pid])); y = novel(entities(t), entities(ref[pid]))
            if x is not None: rn.append(x)
            if y is not None: re_.append(y)
        rec = {"n_num": len(rn), "novel_numbers": float(np.mean(rn)) if rn else None, "ci_num": boot(rn) if rn else None,
               "n_ent": len(re_), "novel_entities": float(np.mean(re_)) if re_ else None, "ci_ent": boot(re_) if re_ else None}
        out["arms"][arm] = rec
        f = lambda m, ci: f"{m:.2f} [{ci[0]:.2f},{ci[1]:.2f}]" if m is not None else "-"
        print(f"{arm:20s} {len(by[arm]):3d} {f(rec['novel_numbers'], rec['ci_num']):>22s} {f(rec['novel_entities'], rec['ci_ent']):>22s}"
              + ("   <- clean band" if arm == a.clean_arm else ""))
    write_json(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
