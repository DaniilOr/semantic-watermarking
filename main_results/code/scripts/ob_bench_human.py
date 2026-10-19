"""Human-text negatives of the OpenStamp-style benchmark as generation records (so the detectors read them unchanged).

runs/ob_bench_null/generations.jsonl   arm "base": the next 200 tokens of the 2,000 NULL documents per dataset
                                        (threshold material for 1% and 0.1% FPR, per dataset)
runs/ob_bench/generations.human.jsonl  arm "human": the next 200 tokens of the 500 TEST documents per dataset
                                        (the paper's negatives, paired with the test prompts)
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.textnorm import normalize_text, text_hash  # noqa: E402

tok = AutoTokenizer.from_pretrained(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"])
rows = [json.loads(l) for l in open("data/ob_bench/prompts.jsonl")]
for split, run, arm in (("null", "runs/ob_bench_null", "base"), ("test", "runs/ob_bench", "human")):
    out = Path(run); out.mkdir(parents=True, exist_ok=True)
    path = out / ("generations.jsonl" if split == "null" else "generations.human.jsonl")
    with path.open("w") as f:
        for r in rows:
            if r["split"] != split:
                continue
            text = normalize_text(r["human_continuation"])
            f.write(json.dumps({"id": f"human.{r['prompt_id']}", "stage": out.name, "arm": arm, "group_id": r["group_id"], "prompt_id": r["prompt_id"],
                                "domain": r["domain"], "key_id": None, "seed": 0, "text": text, "text_hash": text_hash(text),
                                "visible_tokens": len(tok(text, add_special_tokens=False)["input_ids"]), "status": "ok"}, ensure_ascii=False) + "\n")
    print(path, sum(1 for _ in open(path)))
