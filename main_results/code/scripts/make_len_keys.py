"""Seed-phrase key sets of a fixed length, for the key-length experiment (labelled extension).

Each set holds N keys of exactly L random common words; keys of one set share NO word with each other, so the
"wrong key" comparison inside a set measures key specificity rather than accidental word overlap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.keys import SEED_WORDS, KeyRecord  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4,8,10,16")
    ap.add_argument("--per-set", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
    template = json.loads((Path("assets") / "keys_randomwords.json").read_text())
    for L in [int(x) for x in a.lengths.split(",")]:
        assert L * a.per_set <= len(SEED_WORDS), f"vocabulary too small for {a.per_set} disjoint keys of {L} words"
        rng = np.random.default_rng(a.seed + L)
        words = list(SEED_WORDS)
        rng.shuffle(words)
        name = f"len{L}"
        keys = []
        for i in range(a.per_set):
            phrase_words = words[i * L:(i + 1) * L]
            phrase = " ".join(phrase_words)
            keys.append(KeyRecord(key_id=f"{name}_{i + 1:03d}", key_set=name, family="random_words", prefix=phrase, template_ids=[],
                                  completions=list(phrase_words), statements=[], seed=a.seed + L,
                                  token_length=len(tok(phrase, add_special_tokens=False)["input_ids"]), rarity=[],
                                  length_stratum=f"len{L}", model_revision="2fc06364715b967f1860aea9cf38778875588b17"))
        for k in keys:
            k.wrong_key_ids = [o.key_id for o in keys if o.key_id != k.key_id]
        man = {f: template[f] for f in template if f not in ("keys", "key_set", "seed", "content_hash", "per_family")}
        man.update({"key_set": name, "seed": a.seed + L, "per_family": {"random_words": a.per_set},
                    "note": f"{a.per_set} keys of exactly {L} words, disjoint within the set (key-length experiment)",
                    "keys": [dataclasses.asdict(k) for k in keys]})
        dest = Path("assets") / f"keys_{name}.json"
        dest.write_text(json.dumps(man, indent=1))
        print(f"{dest}: " + " | ".join(k.prefix for k in keys[:2]) + " | ...")


if __name__ == "__main__":
    main()
