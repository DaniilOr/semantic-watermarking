"""Summary of runs/lr_detect_*.json: TPR at the dual-null 1% threshold (stricter of human 1% and plain-model 1%, evaluation
half), hard count vs LR, per setting = mean over its datasets; clean, the plain model's 2nd sample (false positives) and
each attack.  Read-only.  Usage: python scripts/lr_summary.py ROOT"""
import glob, json, sys
import numpy as np
root = sys.argv[1]
ATT = ["", "_para_light", "_para_strong", "_pegasus", "_pegasus_bigram", "_parrot", "_parrot_bigram", "_parrotss", "_parrotss_bigram", "_reorder_sent", "_resegment"]
NAME = {"": "clean", "_para_light": "Gemma light", "_para_strong": "Gemma strong", "_pegasus": "Pegasus", "_pegasus_bigram": "Pegasus-bigram",
        "_parrot": "Parrot", "_parrot_bigram": "Parrot-bigram", "_parrotss": "Parrot (faithful)", "_parrotss_bigram": "Parrot-bigram (faithful)",
        "_reorder_sent": "reorder", "_resegment": "resegment"}
rows = {}
for f in sorted(glob.glob(f"{root}/lr_detect_*.json")):
    d = json.load(open(f)); run = d["run"]; R = d["results"]
    doms = sorted({k.split("|")[2] for k in R})
    def mean(r, arm, st):
        v = [R[f"{r}|{arm}|{x}"][f"{st}:dual_1%"]["rate"] for x in doms if R.get(f"{r}|{arm}|{x}", {}).get(f"{st}:dual_1%")]
        return (100 * float(np.mean(v)), len(v)) if v else (None, 0)
    for a in ATT:
        r = run + a
        if not any(k.startswith(r + "|") for k in R):
            continue
        rows[(d["setting"], NAME[a])] = {arm: {st: mean(r, arm, st) for st in ("hard", "lr")} for arm in ("semstamp", "chain_reference", "clean_seed1")}
f = lambda x: "  -  " if x[0] is None else f"{x[0]:5.1f}"
print(f"{'setting':22s} {'condition':24s} | SemStamp hard -> LR | Push hard -> LR | plain 2nd sample hard / LR (FPR)")
for (s, c), v in rows.items():
    print(f"{s:22s} {c:24s} | {f(v['semstamp']['hard'])} -> {f(v['semstamp']['lr'])} | {f(v['chain_reference']['hard'])} -> {f(v['chain_reference']['lr'])} | "
          f"{f(v['clean_seed1']['hard'])} / {f(v['clean_seed1']['lr'])}" if c == "clean" else
          f"{s:22s} {c:24s} | {f(v['semstamp']['hard'])} -> {f(v['semstamp']['lr'])} | {f(v['chain_reference']['hard'])} -> {f(v['chain_reference']['lr'])} |")
json.dump({f"{s}|{c}": v for (s, c), v in rows.items()}, open(f"{root}/lr_summary.json", "w"), indent=1)
