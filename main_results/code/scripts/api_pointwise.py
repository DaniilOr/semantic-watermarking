"""Pointwise (absolute) quality judge through an external API model: ONE response at a time, no comparison.

The pairwise judge says which of two answers is better; it cannot say whether an answer is acceptable, and a forced
comparison amplifies small differences.  Here the judge sees the task and a single anonymous response and reports
whether it addresses the task, which flaws it has, how severe they are, and an overall 1-5 score.  Clean and
watermarked texts are judged by the identical procedure, so the clean generator's own flaw rate -- and its variation
from one clean draw to the next -- can be measured separately and used as the yardstick (scripts/quality_vs_clean.py).

Usage: python scripts/api_pointwise.py --config configs/pilot.yaml --run runs/<name> [--arms base,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_judge import FLAWS, URL, api_key, clip_words, load_prompts, parse_json  # noqa: E402
from wmworld.config import load_config  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

SYSTEM = ("You are a careful, impartial evaluator of a single response that an AI system wrote for a task. "
          "Be strict about real defects and do not reward length for its own sake.")
USER = """Task given to the writer:
<task>
{task}
</task>

<response>
{r}
</response>

Important: the response was produced under a length limit, so an abrupt ending at the very end is NOT a flaw.
Judge the content that is there.

Decide:
- addresses_task: "yes", "partly" or "no" -- does it actually attempt the task that was given (the same question, the same subject)?
- flaws: every flaw that applies, from this list only: {flaws}. Use an empty list if there is none.
  ("unrelated_words_or_insertions" means words, names or topics that do not belong in the text and appear out of nowhere.)
- severity: "none", "minor" or "major" -- how much the flaws damage the response.
- quality: an integer from 1 (unusable) to 5 (excellent) for the response overall.
- note: at most 15 words naming the main problem, or "ok".
Answer with JSON only."""
SCHEMA = {"name": "pointwise_judgment", "strict": True,
          "schema": {"type": "object", "additionalProperties": False,
                     "required": ["addresses_task", "flaws", "severity", "quality", "note"],
                     "properties": {"addresses_task": {"type": "string", "enum": ["yes", "partly", "no"]},
                                    "flaws": {"type": "array", "items": {"type": "string", "enum": FLAWS}},
                                    "severity": {"type": "string", "enum": ["none", "minor", "major"]},
                                    "quality": {"type": "integer", "minimum": 1, "maximum": 5},
                                    "note": {"type": "string"}}}}


def valid(j) -> bool:
    try:
        return (j["addresses_task"] in ("yes", "partly", "no") and j["severity"] in ("none", "minor", "major")
                and isinstance(j["flaws"], list) and 1 <= int(j["quality"]) <= 5)
    except (KeyError, TypeError, ValueError):
        return False


class Pointwise:
    def __init__(self, model: str, key: str):
        self.model, self.key, self.structured, self.no_reasoning, self.usage = model, key, True, True, Counter()

    def ask(self, task: str, response: str):
        body = {"model": self.model, "temperature": 0, "seed": 0, "max_tokens": 1200,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": USER.format(task=task, r=response, flaws=", ".join(FLAWS))}]}
        last = ""
        for attempt in range(6):
            req = dict(body)
            if self.structured:
                req["response_format"] = {"type": "json_schema", "json_schema": SCHEMA}
            if self.no_reasoning:
                req["reasoning"] = {"enabled": False}
            try:
                r = requests.post(URL, json=req, timeout=180, headers={"Authorization": f"Bearer {self.key}", "X-Title": "wmworld pointwise judge"})
            except requests.RequestException as e:
                last = type(e).__name__; time.sleep(min(60, 2 ** attempt)); continue
            if r.status_code == 200 and "error" not in r.json():
                d = r.json(); u = d.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                j = parse_json(d["choices"][0]["message"].get("content") or "")
                if valid(j):
                    j["quality"] = int(j["quality"])
                    return j
                last = "unparseable"; continue
            txt = r.text[:300].lower()
            if r.status_code in (400, 404, 422) and self.no_reasoning and "reasoning" in txt:
                self.no_reasoning = False; continue
            if r.status_code in (400, 404, 422) and self.structured and ("response_format" in txt or "schema" in txt or "structured" in txt):
                self.structured = False; continue
            if r.status_code in (401, 402, 403):
                raise SystemExit(f"API refused the request (HTTP {r.status_code}): check the key / credits")
            last = f"HTTP {r.status_code}"; time.sleep(min(60, 2 ** attempt))
        print(f"  [pointwise] gave up on one text ({last})", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--arms", default="")
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    judge = Pointwise(a.model, api_key())
    slug = a.model.replace("/", "_").replace(":", "_")
    for run in a.run:
        rd = Path(run)
        prompts = load_prompts(cfg, rd)
        gens, _ = collect_run_texts(rd)
        if os.environ.get("JUDGE_DOMAINS"):                         # e.g. only the finished datasets of a run
            gens = [g for g in gens if g.get("domain") in os.environ["JUDGE_DOMAINS"].split()]
        gens = [g for g in gens if g.get("status", "ok") == "ok" and g.get("text") and g["prompt_id"] in prompts
                and (not a.arms or g["arm"] in set(a.arms.split(",")))]
        (rd / "eval").mkdir(exist_ok=True)
        log = rd / "eval" / f"api_pointwise_{slug}.jsonl"
        done = {json.loads(l)["id"] for l in open(log)} if log.exists() else set()
        jid = lambda g: hashlib.sha256(f"{g['text_hash']}|{g['prompt_id']}".encode()).hexdigest()[:20]
        todo = [g for g in gens if jid(g) not in done]
        print(f"[{rd.name}] pointwise judge={a.model}: {len(todo)} texts to rate ({len(done)} done)", flush=True)
        t0 = time.time()

        def work(g):
            j = judge.ask(clip_words(prompts[g["prompt_id"]], 400), clip_words(g["text"], a.max_words))
            return None if j is None else {"id": jid(g), "gen_id": g["id"], "arm": g["arm"], "seed": g.get("seed"), "prompt_id": g["prompt_id"],
                                           "domain": g.get("domain"), "visible_tokens": g.get("visible_tokens"), "judgment": j}
        with ThreadPoolExecutor(max_workers=a.workers) as ex, open(log, "a") as fh:
            for n, rec in enumerate(ex.map(work, todo), 1):
                if rec is not None:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                if n % 100 == 0:
                    print(f"  [{rd.name}] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{rd.name}] tokens used: {dict(judge.usage)}", flush=True)


if __name__ == "__main__":
    main()
