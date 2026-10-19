"""Deterministic content-preserving edits: sentence reordering and resegmentation (no model, no API).

These separate two things the paraphrase attack confounds.  A paraphrase changes the words AND the order; these
edits keep every word and change only the arrangement.  They also discriminate sharply between the detector
families: a statistic that scores each segment on its own is EXACTLY invariant to a permutation of those
segments, while any statistic conditioned on the preceding text is not.  Reordering and resegmentation are
stress tests of the detector's dependence on arrangement, not claims of meaning preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.stages import collect_run_texts  # noqa: E402
from wmworld.textnorm import normalize_text, text_hash  # noqa: E402


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def reorder_sent(text: str, rng) -> str | None:
    s = sentences(text)
    if len(s) < 4:
        return None
    order = rng.permutation(len(s))
    while all(order[i] == i for i in range(len(s))):
        order = rng.permutation(len(s))
    return " ".join(s[i] for i in order)


def reorder_para(text: str, rng) -> str | None:
    parts = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) < 3:
        return None
    order = rng.permutation(len(parts))
    return "\n\n".join(parts[i] for i in order)


def resegment(text: str, rng) -> str | None:
    """Merge adjacent sentence pairs and split long ones at a comma: same words, different boundaries."""
    s = sentences(text)
    if len(s) < 4:
        return None
    out, i = [], 0
    while i < len(s):
        a = s[i]
        if i + 1 < len(s) and len(a.split()) + len(s[i + 1].split()) <= 45:
            b = s[i + 1]
            out.append(re.sub(r"[.!?]+$", "", a) + ", and " + (b[0].lower() + b[1:] if b[:1].isupper() else b))
            i += 2
            continue
        parts = a.split(", ")
        if len(parts) >= 2 and len(a.split()) > 18:
            head = parts[0]
            tail = ", ".join(parts[1:])
            out.append(head + ". " + (tail[0].upper() + tail[1:] if tail[:1].islower() else tail))
        else:
            out.append(a)
        i += 1
    return " ".join(out)


ATTACKS = {"reorder_sent": reorder_sent, "reorder_para": reorder_para, "resegment": resegment}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/conf_cross")
    ap.add_argument("--arms", default="base,wm_c4")
    ap.add_argument("--attacks", default="reorder_sent,resegment,reorder_para")
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    rd = Path(a.run)
    gens, _ = collect_run_texts(rd)
    keep = set(a.arms.split(","))
    gens = [g for g in gens if g.get("status", "ok") == "ok" and g["arm"] in keep and g.get("text")]
    print(f"{rd.name}: {len(gens)} source texts ({dict(Counter(g['arm'] for g in gens))})")
    for name in a.attacks.split(","):
        fn = ATTACKS[name]
        dest = rd.parent / f"{rd.name}_{name}"
        dest.mkdir(parents=True, exist_ok=True)
        n_ok, n_skip, ratios = 0, 0, []
        # keep the attacked copies of arms NOT in this call (the file used to be overwritten, dropping other arms)
        prev = [l for l in open(dest / "generations.jsonl", encoding="utf-8")] if (dest / "generations.jsonl").exists() else []
        kept = [l for l in prev if l.strip() and json.loads(l).get("arm") not in keep]
        with (dest / "generations.jsonl").open("w", encoding="utf-8") as f:
            for l in kept:
                f.write(l if l.endswith("\n") else l + "\n")
            for g in gens:
                rng = np.random.default_rng(int(hashlib.sha256(f"{a.seed}|{name}|{g['id']}".encode()).hexdigest()[:12], 16))
                out = fn(normalize_text(g["text"]), rng)
                if out is None:
                    n_skip += 1
                    continue
                out = normalize_text(out)
                rec = {k: g.get(k) for k in ("arm", "assigned_key_id", "key_id", "prompt_id", "group_id", "domain", "seed", "stage")}
                rec.update({"id": "att." + hashlib.sha256(f"{name}|{g['id']}".encode()).hexdigest()[:20], "parent_id": g["id"],
                            "text": out, "text_hash": text_hash(out), "status": "ok",
                            "visible_tokens": int(g.get("visible_tokens") or len(out.split()) * 1.3),
                            "attack": {"kind": name, "word_ratio": len(out.split()) / max(1, len(g["text"].split())),
                                       "same_multiset": sorted(out.split()) == sorted(g["text"].split())}})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                ratios.append(rec["attack"]["word_ratio"])
        if (rd / "manifest.json").exists():
            (dest / "manifest.json").write_text((rd / "manifest.json").read_text())
        same = sum(1 for l in open(dest / "generations.jsonl") if json.loads(l)["attack"]["same_multiset"])
        print(f"  {name:14s} -> {dest.name}: {n_ok} texts, {n_skip} inapplicable; "
              f"median word ratio {np.median(ratios):.3f}; identical word multiset in {same}/{n_ok}")


if __name__ == "__main__":
    main()
