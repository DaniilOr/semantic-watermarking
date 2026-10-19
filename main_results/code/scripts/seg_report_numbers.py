"""Numbers for docs/segment_report.md, read from the records of runs/ss_llama3b (no model, no embedding).

Per arm: wall seconds per text, length, push mechanics; for chain arms the per-unit budget (candidates drawn, share of
units at the cap), the kept unit's position (in mask, clear of 0.02), per-candidate hit rates and duplicate candidates;
for shipped SemStamp the candidates per accepted sentence and the forced-through share.  Fact probe values from
runs/fact_probe_ss_llama3b_r6.json.  One example text of the segment push arm with its per-segment record.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.config import load_config  # noqa: E402
from wmworld.records import write_json  # noqa: E402

RUN = "runs/ss_llama3b"
EXAMPLE_ARM = "chain_seg_blk4x16_push1_m0.02"


def main() -> None:
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{RUN}/generations.*.jsonl")):
        for ln in open(f):
            g = json.loads(ln)
            if g.get("status", "ok") == "ok":
                by[g["arm"]].append(g)
    key = semwm.Key.shipped()
    signs = semwm.Hyperplanes(np.eye(3))                                  # bin_signs only depends on the number of planes
    out = {"run": RUN, "arms": {}}
    for arm, gs in sorted(by.items()):
        c = [g["cost"] for g in gs]
        wall = np.array([x.get("wall_seconds") or 0.0 for x in c])
        tok = np.array([g["visible_tokens"] for g in gs])
        rec = {"n": len(gs), "s_per_text_mean": float(wall.mean()), "s_per_text_median": float(np.median(wall)),
               "tokens_mean": float(tok.mean()), "tokens_median": float(np.median(tok)), "share_at_512_tokens": float(np.mean(tok >= 512)),
               "words_mean": float(np.mean([len(g["text"].split()) for g in gs]))}
        for k in ("steer_bias_abs_mean", "steer_bias_abs_max", "steer_p_boundary_before", "steer_p_boundary_after",
                  "steer_rep_guard_hits", "steer_strings", "steer_embed_calls"):
            v = [x[k] for x in c if k in x]
            if v:
                rec[k] = float(np.mean(v))
        U = [u for x in c for u in x.get("units", []) if u.get("prev_bin") is not None]
        if U and "n_drawn" in U[0]:
            nd = np.array([u["n_drawn"] for u in U])
            kept = [np.array(u["cands"][u["chosen"]][:3]) for u in U]
            masks = [semwm.valid_mask(key, u["prev_bin"]) for u in U]
            clear = [bool(u["in_mask"]) and float(np.min(np.abs(q))) >= semwm.MARGIN for u, q in zip(U, kept)]
            n_c = hit = hit_clear = dup = 0
            for u, m in zip(U, masks):
                P = [np.array(cd[:3]) for cd in u["cands"]]
                n_c += len(P)
                d = [semwm.signed_distance(q, m, signs) for q in P]
                hit += sum(x > 0 for x in d)
                hit_clear += sum(x > 0 and float(np.min(np.abs(q))) >= semwm.MARGIN for x, q in zip(d, P))
                dup += len(P) - len({tuple(q) for q in (tuple(cd[:3]) for cd in u["cands"])})
            rec.update({"units_scored": len(U), "units_per_text_incl_seed": float(np.mean([x["n_units"] for x in c])),
                        "cand_per_unit_mean": float(nd.mean()), "cand_per_unit_median": float(np.median(nd)),
                        "share_units_4": float(np.mean(nd == 4)), "share_units_8": float(np.mean(nd == 8)),
                        "share_units_12": float(np.mean(nd == 12)), "share_units_at_cap_16": float(np.mean(nd == 16)),
                        "kept_in_mask": float(np.mean([u["in_mask"] for u in U])), "kept_in_mask_clear_0.02": float(np.mean(clear)),
                        "kept_depth_mean": float(np.mean([u["depth"] for u in U])),
                        "candidates_total": n_c, "cand_in_mask_share": hit / n_c, "cand_in_mask_clear_share": hit_clear / n_c,
                        "cand_duplicate_share": dup / n_c})
        if arm == "semstamp" or arm.startswith("steered_sent"):
            tr, acc, mx = (sum(x.get(k, 0) for x in c) for k in ("trials", "accepted", "maxed_out"))
            rec.update({"trials": tr, "accepted": acc, "cand_per_accepted_sentence": tr / max(1, acc), "forced_through_share": mx / max(1, acc)})
        out["arms"][arm] = rec
    fp = json.loads(Path("runs/fact_probe_ss_llama3b_r6.json").read_text())
    out["fact_probe_r6"] = fp
    # one example: a Dolly prompt of the segment push arm, a text with 8-12 segments, not parked in bin 4 (<= 30% of its
    # units there; bin 4's mask contains bin 4) and with at least one unit outside its mask -- a typical text, not the best
    cfg = load_config("configs/pilot.yaml")
    rows, _ = semwm.held_out_prompts(cfg, "runs/conf_cross")
    prompt_of = {p.prompt_id: p.prompt for p in rows}
    typical = lambda us: np.mean([u["bin"] == 4 for u in us]) <= 0.3 and not all(u["in_mask"] for u in us if u.get("prev_bin") is not None)
    cands = sorted((g for g in by[EXAMPLE_ARM] if 8 <= g["cost"]["n_units"] <= 16 and "\n-" not in g["text"] and typical(g["cost"]["units"])),
                   key=lambda g: (g["domain"] != "writingprompts", g["prompt_id"]))
    g = cands[0]
    segs = []
    ids_words = g["text"].split()
    for i, u in enumerate(g["cost"]["units"]):
        segs.append({"index": i, "words": " ".join(ids_words[i * 16:(i + 1) * 16]), "prev_bin": u.get("prev_bin"), "bin": u["bin"],
                     "mask": list(semwm.valid_mask(key, u["prev_bin"])) if u.get("prev_bin") is not None else None,
                     "in_mask": u["in_mask"], "depth": u["depth"], "n_drawn": u["n_drawn"],
                     "kept_min_abs_proj": float(np.min(np.abs(np.array(u["cands"][u["chosen"]][:3]))))})
    out["example"] = {"arm": EXAMPLE_ARM, "prompt_id": g["prompt_id"], "prompt": prompt_of.get(g["prompt_id"]), "text": g["text"], "segments": segs}
    write_json("runs/seg_report_numbers.json", out)
    for arm, r in out["arms"].items():
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(out["example"], indent=1)[:6000])


if __name__ == "__main__":
    main()
