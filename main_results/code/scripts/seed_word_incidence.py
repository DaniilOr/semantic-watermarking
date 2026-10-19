"""Incidental occurrence of the seed-phrase vocabulary in ordinary text (no model): how often each SEED_WORD appears
as a whole word in the prepared human/web references (all partitions) and in the dev-stage base outputs.
Prints counts per word and the low-incidence subset (for a key vocabulary whose words rarely occur by chance)."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.config import load_config
from wmworld.data.prepare import load_prepared
from wmworld.key_catalog import SEED_WORDS

cfg = load_config("configs/pilot.yaml")
refs = [p.reference for p in load_prepared(cfg.path_of("data_prepared")) if p.reference]
base = []
for run in ("runs/dev", "runs/rw_base_seed1"):
    p = Path(run) / "generations.jsonl"
    if p.exists():
        base += [json.loads(l)["text"] for l in open(p) if l.strip() and '"arm": "base"' in l]
print(f"{len(refs)} references, {len(base)} base outputs")
rows = []
for w in SEED_WORDS:
    rx = re.compile(r"\b" + re.escape(w) + r"(s|es)?\b", re.I)
    nr = sum(1 for t in refs if rx.search(t)); nb = sum(1 for t in base if rx.search(t))
    rows.append((w, nr, nb, 1000.0 * nr / max(1, len(refs)), 1000.0 * nb / max(1, len(base))))
rows.sort(key=lambda r: (r[3] + r[4], r[0]))
for w, nr, nb, pr, pb in rows:
    print(f"{w:10s} refs={nr:4d} ({pr:5.1f}/1000)  base={nb:3d} ({pb:5.1f}/1000)")
low = [r[0] for r in rows if r[3] <= 1.0 and r[4] <= 2.0]
print(f"low-incidence (<=1/1000 refs and <=2/1000 base): {len(low)} words")
print("SEED_WORDS_RARE = " + json.dumps(low))
