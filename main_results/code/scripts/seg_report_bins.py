"""Where the units of clean and watermarked texts land under the shipped key (docs/segment_report.md).

Shipped key: bin 4's mask is (4, 1), so a text that stays in bin 4 is in the mask at every transition; bin 4 is in the
masks of 5 of the 8 bins.  Per arm of runs/ss_llama3b: share of units per bin, share of scored transitions that are
4 -> 4, in-mask rate of the other transitions, and share of texts with >= 80% of their units in bin 4.  Units are the
detector's own (16-word segments or sentences), embedded with the SemStamp encoder on CPU.
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
from wmworld.records import write_json  # noqa: E402

RUN = "runs/ss_llama3b"
EMB = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
ARMS = {"seg": ["base", "clean_seed1", "chain_seg_blk4x16", "chain_seg_blk4x16_m0.02", "chain_seg_blk4x16_push1", "chain_seg_blk4x16_push1_m0.02"],
        "sent": ["base", "clean_seed1", "semstamp", "chain_sent_blk4x16", "chain_sent_blk4x16_push1_m0.02"]}


def main() -> None:
    texts = defaultdict(list)
    for f in sorted(glob.glob(f"{RUN}/generations.*.jsonl")):
        for ln in open(f):
            g = json.loads(ln)
            if g.get("status", "ok") == "ok" and g.get("text"):
                texts[g["arm"]].append(g["text"])
    emb = semwm.Embedder(EMB, device="cpu", batch_size=128)
    key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
    out = {"mask_table": {b: list(semwm.valid_mask(key, b)) for b in range(8)}, "units": {}}
    for unit, arms in ARMS.items():
        for arm in arms:
            split = (lambda t: semwm.split_segments(t, 16)) if unit == "seg" else semwm.split_sentences
            pieces = [split(t) for t in texts[arm]]
            flat = emb.encode([p for ps in pieces for p in ps])
            bins, k = [], 0
            for ps in pieces:
                bins.append([planes.bin_of(q) for q in planes.projections(flat[k:k + len(ps)])]); k += len(ps)
            allb = [b for bs in bins for b in bs]
            tr = [(bs[i - 1], bs[i]) for bs in bins for i in range(1, len(bs))]
            self44 = [a == 4 and b == 4 for a, b in tr]
            other = [b in semwm.valid_mask(key, a) for a, b in tr if not (a == 4 and b == 4)]
            rec = {"texts": len(bins), "units": len(allb), "transitions": len(tr),
                   "bin_share": {b: float(np.mean([x == b for x in allb])) for b in range(8)},
                   "in_mask_rate_all": float(np.mean([b in semwm.valid_mask(key, a) for a, b in tr])),
                   "share_transitions_4to4": float(np.mean(self44)),
                   "in_mask_rate_excl_4to4": float(np.mean(other)) if other else None,
                   "share_texts_80pct_bin4": float(np.mean([np.mean([x == 4 for x in bs]) >= 0.8 for bs in bins if bs]))}
            out["units"][f"{unit}|{arm}"] = rec
            print(unit, arm, {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in rec.items() if k2 != "bin_share"},
                  "bins", {b: round(v, 3) for b, v in rec["bin_share"].items()}, flush=True)
    write_json("runs/seg_report_bins.json", out)


if __name__ == "__main__":
    main()
