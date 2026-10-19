"""Integrity check of a bin-adapter candidate corpus (<run>/cands.shard*.jsonl + the run's generation records).

1. Rebuilt text: the chosen candidates' ids, concatenated, decode to the generation record's text.
2. Rebuilt context: prefix_len of unit i == len(chat prefix) + sum of chosen lengths before i.
3. Bins: a sample of candidate texts re-embedded gives the logged bins (and projections within 1e-4).
Also prints the corpus size and the per-bin label counts.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402
from wmworld.textnorm import normalize_text  # noqa: E402

SBERT = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
LLAMA3B = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"]


def load_corpus(run: str) -> dict[str, dict]:
    """Last line per prompt_id wins (a text regenerated after a crash replaces its earlier log)."""
    out = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            out[r["prompt_id"]] = r
    return out


def main() -> None:
    run = sys.argv[1]
    n_embed = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    from transformers import AutoTokenizer
    from wmworld.generation import GENERATOR_SYSTEM
    tok = AutoTokenizer.from_pretrained(LLAMA3B)
    corpus = load_corpus(run)
    recs = {g["prompt_id"]: g for g in collect_run_texts(Path(run))[0] if g.get("status", "ok") == "ok"}
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    text_ok = ctx_ok = n_units = n_cands = 0
    bins = Counter()
    sample = []
    for pid, c in corpus.items():
        chosen = [u["cands"][u["chosen"]]["ids"] for u in c["units"]]
        ids = [t for ch in chosen for t in ch]
        text_ok += normalize_text(tok.decode([t for t in ids if t < 128000], skip_special_tokens=True)) == recs[pid]["text"]
        chat = tok.apply_chat_template([{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": prompts[pid]}],
                                       tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024")
        L = len(tok(chat, add_special_tokens=False)["input_ids"])
        ok = True
        for i, u in enumerate(c["units"]):
            ok &= u["prefix_len"] == L + sum(len(x) for x in chosen[:i])
            n_units += 1; n_cands += len(u["cands"])
            for cd in u["cands"]:
                if u["mask"] is not None:
                    bins[cd["bin"]] += 1
                sample.append((cd["text"], cd["bin"], cd["proj"]))
        ctx_ok += ok
    print(f"{run}: {len(corpus)} texts ({len(recs)} records), {n_units} units, {n_cands} candidates")
    print(f"  rebuilt text == record text: {text_ok}/{len(corpus)}; prefix lengths consistent: {ctx_ok}/{len(corpus)}")
    print(f"  bin labels (units with a mask): " + " ".join(f"{b}:{bins[b]}" for b in range(8)))
    rng = np.random.default_rng(0)
    pick = [sample[i] for i in rng.choice(len(sample), size=min(n_embed, len(sample)), replace=False)]
    emb = semwm.Embedder(SBERT, device="cpu", batch_size=64)
    pl = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    P = pl.projections(emb.encode([t for t, _, _ in pick]))
    same_bin = sum(pl.bin_of(p) == b for p, (_, b, _) in zip(P, pick))
    max_dev = float(np.max(np.abs(P - np.array([q for _, _, q in pick]))))
    print(f"  re-embedded {len(pick)} candidates: same bin {same_bin}/{len(pick)}; max |projection difference| {max_dev:.2e}")


if __name__ == "__main__":
    main()
