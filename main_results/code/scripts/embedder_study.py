"""Which sentence encoder and which unit length keep a text's semantic signature under our attacks?  Zero generation.

For every encoder x unit (sentences, or W-word segments) x condition (light/strong paraphrase by gemma-4-31b,
sentence reordering, resegmentation) on the held-out texts and their attacked versions:
  consistency   share of aligned unit pairs (greedy cosine matching original <-> attacked) that keep their 3-bit LSH
                bin under the shipped key -- what SemStamp's chained detector needs;
  text_corr     correlation across texts of the mean unit projection onto 18 directions (shipped planes + 5 alt keys),
                original vs attacked -- alignment-free, what the fixed-direction segment statistic needs;
  margin_share  share of units at >= 0.02 from every plane (how much a margin constraint would reject);
  bin_balance   min/max bin frequency on clean units (a random-hyperplane partition is not uniform on real text).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

CONDITIONS = [("para_light", "runs/conf_cross_para_light"), ("para_strong", "runs/conf_cross_para_strong"),
              ("reorder_sent", "runs/conf_cross_reorder_sent"), ("resegment", "runs/conf_cross_resegment")]
UNITS = ["sent", "seg8", "seg16", "seg32"]
E5_PREFIX = "query: "


def units_of(text: str, unit: str) -> list[str]:
    return semwm.split_sentences(text) if unit == "sent" else semwm.split_segments(text, int(unit[3:]))


def embedders(a) -> list[tuple[str, str, str]]:
    """(name, path, prefix) for every encoder that is available locally."""
    out = [("semstamp_sbert", a.semstamp_sbert, "")]
    m = json.loads(Path(a.models_json).read_text()) if Path(a.models_json).exists() else {}
    for repo, name in (("sentence-transformers/all-mpnet-base-v2", "mpnet_v2"), ("BAAI/bge-base-en-v1.5", "bge_base"),
                       ("thenlper/gte-base", "gte_base"), ("intfloat/e5-base-v2", "e5_base")):
        if repo in m:
            out.append((name, m[repo]["path"], E5_PREFIX if name == "e5_base" else ""))
    return out


def greedy_align(a: np.ndarray, b: np.ndarray, min_cos: float = 0.5) -> list[tuple[int, int]]:
    """Pairs (i, j) of original unit i and attacked unit j, best cosine first, each used once, cosine >= min_cos."""
    if not len(a) or not len(b):
        return []
    sims = a @ b.T
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
    used_a, used_b, pairs = set(), set(), []
    for i, j in order:
        if sims[i, j] < min_cos:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); pairs.append((int(i), int(j)))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semstamp-sbert", default="/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282")
    ap.add_argument("--models-json", default="assets/ss_models.json")
    ap.add_argument("--source", default="runs/conf_cross")
    ap.add_argument("--arms", default="base,wm_c4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/embedder_study.json")
    a = ap.parse_args()

    arms = set(a.arms.split(","))
    src = {g["id"]: g for g, in [(g,) for g in collect_run_texts(Path(a.source))[0]] if g.get("status", "ok") == "ok" and g["arm"] in arms and g.get("text")}
    conds = {}
    for name, run in CONDITIONS:
        rows = [g for g in collect_run_texts(Path(run))[0] if g.get("status", "ok") == "ok" and g.get("text") and g.get("parent_id") in src]
        conds[name] = rows
        print(f"{name}: {len(rows)} attacked texts linked to originals", flush=True)

    keys = [semwm.Key.shipped()] + [semwm.Key.alt(i) for i in range(1, 6)]
    report = {"units": UNITS, "conditions": [c for c, _ in CONDITIONS], "embedders": {}}
    t0 = time.time()
    for name, path, prefix in embedders(a):
        emb = semwm.Embedder(path, device=a.device, batch_size=256, prefix=prefix)
        planes = [semwm.Hyperplanes.for_key(k, emb.dim) for k in keys]
        dirs = np.concatenate([p.unit for p in planes])                    # [18, dim]
        shipped = planes[0]
        rep = {"path": path, "dim": emb.dim, "units": {}}
        for unit in UNITS:
            cache: dict[str, np.ndarray] = {}

            def embed(g):
                if g["id"] not in cache:
                    cache[g["id"]] = emb.encode(units_of(g["text"], unit))
                return cache[g["id"]]

            orig = {gid: embed(g) for gid, g in src.items()}
            all_units = np.concatenate([v for v in orig.values() if len(v)])
            proj = shipped.projections(all_units)
            bins = np.array([shipped.bin_of(p) for p in proj])
            freq = np.bincount(bins, minlength=8) / len(bins)
            urec = {"n_units_clean": int(len(all_units)), "units_per_text": float(np.mean([len(v) for v in orig.values()])),
                    "bin_freq": [round(float(f), 4) for f in freq], "bin_balance": float(freq.min() / freq.max()),
                    "margin_share": float(np.mean(np.all(np.abs(proj) >= semwm.MARGIN, axis=1))),
                    "proj_sd": float(proj.std()), "conditions": {}}
            for cond, rows in conds.items():
                cons, n_pairs, mo, ma = [], 0, [], []
                for g in rows:
                    o, x = orig[g["parent_id"]], embed(g)
                    if not len(o) or not len(x):
                        continue
                    pairs = greedy_align(o, x)
                    if pairs:
                        po, px = shipped.projections(o[[i for i, _ in pairs]]), shipped.projections(x[[j for _, j in pairs]])
                        cons += [shipped.bin_of(p) == shipped.bin_of(q) for p, q in zip(po, px)]
                        n_pairs += len(pairs)
                    mo.append((o @ dirs.T).mean(axis=0)); ma.append((x @ dirs.T).mean(axis=0))
                mo, ma = np.array(mo), np.array(ma)
                corr = [float(np.corrcoef(mo[:, j], ma[:, j])[0, 1]) for j in range(dirs.shape[0]) if mo[:, j].std() > 0 and ma[:, j].std() > 0]
                urec["conditions"][cond] = {"n_texts": int(len(mo)), "n_aligned_pairs": n_pairs,
                                            "consistency": float(np.mean(cons)) if cons else None,
                                            "text_corr": float(np.mean(corr)) if corr else None,
                                            "aligned_share": float(n_pairs / max(1, sum(len(orig[g["parent_id"]]) for g in rows)))}
            rep["units"][unit] = urec
            print(f"[{name}/{unit}] units/text {urec['units_per_text']:.1f} balance {urec['bin_balance']:.2f} margin-share {urec['margin_share']:.2f} | "
                  + " ".join(f"{c}: cons {r['consistency'] if r['consistency'] is None else round(r['consistency'], 3)} corr {r['text_corr'] if r['text_corr'] is None else round(r['text_corr'], 3)}"
                             for c, r in urec["conditions"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        report["embedders"][name] = rep
        del emb
        import torch; torch.cuda.empty_cache()

    print("\n" + "=" * 118)
    print(f"{'embedder':15s} {'unit':6s} {'units/txt':>9s} {'balance':>7s} {'margin':>6s} | " + " ".join(f"{c[:12]:>12s}" for c, _ in CONDITIONS) + "   (bin consistency / text corr)")
    for name, rep in report["embedders"].items():
        for unit, u in rep["units"].items():
            row = f"{name:15s} {unit:6s} {u['units_per_text']:9.1f} {u['bin_balance']:7.2f} {u['margin_share']:6.2f} | "
            for c, _ in CONDITIONS:
                r = u["conditions"][c]
                row += f"{(r['consistency'] or 0):5.2f}/{(r['text_corr'] or 0):5.2f}  "
            print(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
