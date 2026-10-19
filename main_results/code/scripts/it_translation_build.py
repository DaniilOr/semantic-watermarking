"""Translation task for the instruction benchmark (2026-10-07): German -> English, passages of News Commentary.

Into English because the SemStamp sentence encoder reads English.  Source: Helsinki-NLP/news_commentary (de-en, the
OPUS release, sentence-aligned in document order; no document ids, so a passage can occasionally span two articles).
Passages = 6 consecutive sentence pairs (non-overlapping windows), each sentence 3-60 words, German passage 60-250 words.
Windows in a fixed hash order (salt "it_bench"): 1,000 test passages, then 2,000 null passages whose English side (human
translation) sets the false-alarm thresholds.  Appends domain it_translation to data/it_bench/prompts.jsonl and its
human texts to runs/it_bench_null/generations.jsonl (replacing earlier it_translation rows).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.textnorm import normalize_text, stable_hash, text_hash  # noqa: E402

PROMPT = "Translate the following German text into English. Reply with the English translation only.\n\n{de}"
WIN = 6


def main():
    df = pd.read_parquet(hf_hub_download("Helsinki-NLP/news_commentary", "de-en/train-00000-of-00001.parquet", repo_type="dataset",
                                         local_dir="data/raw/news_commentary"))
    pairs = [(r["de"].strip(), r["en"].strip()) for r in df["translation"]]
    ok = lambda s: 3 <= len(s.split()) <= 60
    wins = []
    for i in range(0, len(pairs) - WIN, WIN):
        w = pairs[i:i + WIN]
        if all(ok(d) and ok(e) for d, e in w):
            de, en = " ".join(d for d, _ in w), " ".join(e for _, e in w)
            if 60 <= len(de.split()) <= 250:
                wins.append((str(i), de, en))
    wins.sort(key=lambda x: stable_hash("it_bench", "translation", x[0]))
    test, null = wins[:1000], wins[1000:3000]
    tok = AutoTokenizer.from_pretrained(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"])
    rows = []
    for split, items in (("test", test), ("null", null)):
        for i, de, en in items:
            rows.append({"prompt_id": f"it:translation:{i}", "group_id": "it_" + stable_hash("translation", i, length=12),
                         "domain": "it_translation", "split": split, "prompt": PROMPT.format(de=de), "excerpt": de,
                         "human_continuation": en})
    p = Path("data/it_bench/prompts.jsonl")
    keep = [l for l in p.read_text().split("\n") if l if '"it_translation"' not in l]
    p.write_text("\n".join(keep + [json.dumps(r, ensure_ascii=False) for r in rows]) + "\n")
    n = Path("runs/it_bench_null/generations.jsonl")
    keepn = [l for l in n.read_text().split("\n") if l if '"it_translation"' not in l]
    new = []
    for r in rows:
        if r["split"] == "null":
            text = normalize_text(r["human_continuation"])
            new.append(json.dumps({"id": f"human.{r['prompt_id']}", "stage": "it_bench_null", "arm": "base", "group_id": r["group_id"],
                                   "prompt_id": r["prompt_id"], "domain": "it_translation", "key_id": None, "seed": 0, "text": text,
                                   "text_hash": text_hash(text), "visible_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                                   "status": "ok"}, ensure_ascii=False))
    n.write_text("\n".join(keepn + new) + "\n")
    words = [len(de.split()) for _, de, _ in test]
    print(json.dumps({"pairs": len(pairs), "windows": len(wins), "test": len(test), "null": len(null),
                      "test_german_words_mean": round(sum(words) / len(words), 1)}))


if __name__ == "__main__":
    main()
