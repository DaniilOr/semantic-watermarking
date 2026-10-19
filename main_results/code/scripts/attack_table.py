"""Per-attack robustness table from an attack run's evaluation (dev scale): for each attack, the watermarked arm's
paired score shift vs the attacked clean text, mean t, t-AUROC vs attacked clean texts, and the toy flag rate;
plus the clean parents' numbers from the source run for reference.  Usage: python scripts/attack_table.py runs/attacks_<name> runs/<name>"""
import json, sys
res = json.load(open(f"{sys.argv[1]}/eval/results.json"))
src = json.load(open(f"{sys.argv[2]}/eval/results.json")) if len(sys.argv) > 2 else None
def row(a, name):
    td = res["t_detector"]["per_arm"].get(name, {}) if "t_detector" in res else {}
    rd = res["rank_detector"]["per_arm"].get(name, {}) if "rank_detector" in res else {}
    return (a.get("n"), a.get("n_eligible"), (a.get("paired_shift_vs_base") or {}).get("mean"), (td.get("t_mean") or {}).get("mean"),
            (td.get("auroc_t_vs_base") or {}).get("auroc"), (rd.get("auroc_zdiff_vs_base") or {}).get("auroc"), a.get("visible_tokens_mean"))
print(f"{'arm':38s} {'n':>3s} {'elig':>4s} {'shift':>7s} {'t_mean':>7s} {'t_AUROC':>7s} {'z_AUROC':>7s} {'tokens':>6s}")
if src:
    for name in ("base", "poe_steer"):
        r = row(src["per_arm"][name], name)
        print(f"{'(unattacked) ' + name:38s} {r[0]:3d} {r[1]:4d} {r[2] if r[2] is not None else float('nan'):7.3f} {r[3] if r[3] is not None else float('nan'):7.2f} {r[4] if r[4] is not None else float('nan'):7.3f} {r[5] if r[5] is not None else float('nan'):7.3f} {r[6]:6.0f}")
for name in sorted(res["per_arm"]):
    if not name.startswith("poe_steer+"):
        continue
    r = row(res["per_arm"][name], name)
    f = lambda v, w: (f"{v:{w}}" if v is not None else " " * (int(w.split('.')[0]) - 0) if False else (f"{v:{w}}" if v is not None else "-".rjust(int(w.split('.')[0]))))
    print(f"{name:38s} {r[0]:3d} {r[1]:4d} {f(r[2], '7.3f')} {f(r[3], '7.2f')} {f(r[4], '7.3f')} {f(r[5], '7.3f')} {r[6]:6.0f}")
# attacked clean texts: null shift check (should be ~0 by construction) and how many attacks failed
att = [json.loads(l) for l in open(f"{sys.argv[1]}/attacks.jsonl") if l.strip()]
from collections import Counter
st = Counter((a["attack"], a["status"]) for a in att)
print("attack status counts:", dict(st))
