"""Robustness at the UNCHANGED clean-text detector and threshold (spec Section 10): for each attack, the fraction of
attacked watermarked texts still above the clean 1% threshold (TPR), the fraction of attacked CLEAN texts above it
(attacked-control FPR), the AUROC with its bootstrap CI, and how much the paraphrases actually changed the text.
Usage: python scripts/attack_proof.py runs/attacks_<name> runs/<name>"""
import json, sys
from collections import defaultdict
from pathlib import Path
att_dir, src_dir = Path(sys.argv[1]), Path(sys.argv[2])
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
tp_all = json.load(open(src_dir / "eval" / "t_null_pool.json"))["per_arm"]
wm_arms = [a for a in tp_all if a != "base"]
thr = tp_all[wm_arms[0]]["thr_base_alpha0.01"]          # same clean null for every arm of the run
res = json.load(open(att_dir / "eval" / "results.json"))
src_res = json.load(open(src_dir / "eval" / "results.json")) if (src_dir / "eval" / "results.json").exists() else {}
dets = {}
for r in load(att_dir / "detections_raw.jsonl"):
    if r.get("t_stat") is not None: dets[(r["text_hash"], r["key_id"])] = r["t_stat"]
src_dets = {}
if (src_dir / "detections_raw.jsonl").exists():
    for r in load(src_dir / "detections_raw.jsonl"):
        if r.get("t_stat") is not None: src_dets[(r["text_hash"], r["key_id"])] = r["t_stat"]
metrics = {a["id"]: a.get("metrics", {}) for a in load(att_dir / "attacks.jsonl")}
gens = load(att_dir / "generations.jsonl")
by = defaultdict(lambda: {"pos": [], "neg": [], "sim": []})
for g in gens:
    parent, attack = g["arm"].split("+", 1)
    t = dets.get((g["text_hash"], g["assigned_key_id"]))
    if t is None: continue
    if parent == "base":
        for wa in wm_arms:                                   # attacked clean texts are the control for every arm
            by[(wa, attack)]["neg"].append(t)
    else:
        by[(parent, attack)]["pos"].append(t)
        if metrics.get(g["id"], {}).get("word_similarity") is not None:
            by[(parent, attack)]["sim"].append(metrics[g["id"]]["word_similarity"])
# unattacked reference from the source run
sg = [g for f in sorted(src_dir.glob("generations*.jsonl")) for g in load(f)]
print(f"clean-text threshold (1% of pooled clean null, frozen from the unattacked run): t > {thr:.2f}")
print(f"{'arm / attack':44s} {'n':>3s} {'TPR@thr':>7s} {'ctrlFPR':>7s} {'AUROC':>6s} {'95% CI':>14s} {'words changed':>13s}")
u_neg = [src_dets[(g["text_hash"], g["assigned_key_id"])] for g in sg if g["arm"] == "base" and (g["text_hash"], g["assigned_key_id"]) in src_dets]
for wa in wm_arms:
    u_pos = [src_dets[(g["text_hash"], g["assigned_key_id"])] for g in sg if g["arm"] == wa and (g["text_hash"], g["assigned_key_id"]) in src_dets]
    au = (src_res.get("t_detector", {}).get("per_arm", {}).get(wa, {}).get("auroc_t_vs_base") or {}).get("auroc")
    if u_pos:
        print(f"{wa + ' (unattacked)':44s} {len(u_pos):3d} {sum(t > thr for t in u_pos)/len(u_pos):7.2f} {sum(t > thr for t in u_neg)/max(1,len(u_neg)):7.2f} "
              f"{au if au is not None else float('nan'):6.3f}")
    else:
        tp = tp_all[wa]
        print(f"{wa + ' (unattacked, pooled null)':44s} {tp.get('n_pos', 0):3d} {tp.get('tpr_base_alpha0.01', float('nan')):7.2f} {'0.01':>7s} {tp.get('auroc_vs_base_null', float('nan')):6.3f}")
for (wa, attack) in sorted(by):
    d = by[(wa, attack)]
    if not d["pos"] or not d["neg"]: continue
    a = res["t_detector"]["per_arm"].get(wa + "+" + attack, {}).get("auroc_t_vs_base") or {}
    ci = f"[{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]" if a.get("ci95") else ""
    chg = f"{100*(1-sum(d['sim'])/len(d['sim'])):.0f}%" if d["sim"] else ""
    print(f"{wa + ' + ' + attack:44s} {len(d['pos']):3d} {sum(t > thr for t in d['pos'])/len(d['pos']):7.2f} {sum(t > thr for t in d['neg'])/len(d['neg']):7.2f} "
          f"{a.get('auroc', float('nan')):6.3f} {ci:>14s} {chg:>13s}")
