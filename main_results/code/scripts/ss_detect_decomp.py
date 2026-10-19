"""Why one sentence-chain arm is more detectable than another: decompose the detector's chain z per text.

z_chain = (k - 0.25 n) / sqrt(0.1875 n) over the n scored sentences (every sentence after the first; its mask comes from
the previous sentence's bin), k of them in their mask.  So TPR is set by n (sentences the detector sees) and k / n.
Per run x arm x domain:
  generation   sentences per text; SemStamp: share of sentences kept unfiltered after --max-trials (maxed out) and
               trials per sentence; chain arms: in-mask rate of the chosen units
  detection    (texts re-split and re-embedded exactly as ob_detect.py) scored sentences per text, in-mask share k / n
               (pooled), mean z_chain, share of texts whose detected sentence count equals the generated unit count
Usage: python scripts/ss_detect_decomp.py [--out FILE] RUN [RUN ...]   (arms semstamp and chain_reference; default out runs/ss_detect_decomp.json)
"""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld import semwm  # noqa: E402

SBERT = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
ARMS = ("semstamp", "chain_reference")


def main():
    emb = semwm.Embedder(SBERT, device="cpu", batch_size=256)
    key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
    args = sys.argv[1:]
    dest = "runs/ss_detect_decomp.json"
    if args[:1] == ["--out"]:
        dest, args = args[1], args[2:]
    out = {}
    for run in args:
        recs = defaultdict(list)
        for arm in ARMS:
            for f in sorted(glob.glob(f"{run}/generations.{arm}.shard*.jsonl")):
                for ln in open(f):
                    r = json.loads(ln)
                    if r.get("status", "ok") == "ok" and r.get("text"):
                        recs[(arm, r["domain"])].append(r)
        for (arm, dom), rs in sorted(recs.items()):
            sents = [semwm.split_sentences(r["text"]) for r in rs]
            flat = emb.encode([s for ss in sents for s in ss]) if any(sents) else np.zeros((0, emb.dim))
            k0, n_sc, k_in, zs, aligned, n_gen, fb, tr, gin = 0, [], 0, [], 0, [], [], [], []
            for r, ss in zip(rs, sents):
                e = flat[k0:k0 + len(ss)]; k0 += len(ss)
                bins = [planes.bin_of(q) for q in planes.projections(e)] if len(ss) else []      # as semwm.score_text
                sc = {"in_mask": [bins[i] in semwm.valid_mask(key, bins[i - 1], semwm.LMBD) for i in range(1, len(bins))],
                      "z_chain": semwm.chain_z(bins, key, semwm.LMBD) if len(bins) > 1 else None}
                n_sc.append(len(sc["in_mask"])); k_in += sum(sc["in_mask"])
                if sc["z_chain"] is not None:
                    zs.append(sc["z_chain"])
                c = r["cost"]
                if arm == "semstamp":
                    g = c.get("accepted", 0); fb.append((c.get("maxed_out", 0), g)); tr.append((c.get("trials", 0), g))
                else:
                    g = c.get("n_units", 0)
                    if c.get("in_mask_rate") is not None:
                        gin.append((c["in_mask_rate"], max(0, g - 1)))
                n_gen.append(g); aligned += (g == len(ss))
            n = len(rs); N = sum(n_sc)
            rec = {"n_texts": n, "gen_sentences_per_text": float(np.mean(n_gen)),
                   "det_scored_per_text": float(np.mean(n_sc)), "det_in_mask": k_in / N if N else None,
                   "det_mean_z": float(np.mean(zs)) if zs else None, "aligned_share": aligned / n}
            if fb:
                rec["gen_maxed_out_share"] = sum(a for a, _ in fb) / max(1, sum(b for _, b in fb))
                rec["gen_trials_per_sentence"] = sum(a for a, _ in tr) / max(1, sum(b for _, b in tr))
            if gin:
                rec["gen_in_mask"] = sum(x * w for x, w in gin) / max(1, sum(w for _, w in gin))
            out[f"{Path(run).name}|{arm}|{dom}"] = rec
            print(f"{Path(run).name:18s} {arm:16s} {dom:13s} n {n:4d} | gen sents {rec['gen_sentences_per_text']:5.1f}"
                  + (f" maxed-out {100 * rec['gen_maxed_out_share']:5.1f}% trials/sent {rec['gen_trials_per_sentence']:5.1f}" if fb else "")
                  + (f" gen in-mask {100 * rec['gen_in_mask']:5.1f}%" if gin else "")
                  + f" | det scored {rec['det_scored_per_text']:5.1f} in-mask {100 * (rec['det_in_mask'] or 0):5.1f}% mean z {rec['det_mean_z'] or 0:5.2f}"
                  f" aligned {100 * rec['aligned_share']:5.1f}%", flush=True)
    Path(dest).write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
