import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _path
from wmark.io import append_jsonl, load_generations, load_prompts, read_jsonl, short_hash
from wmark.openrouter import DEFAULT_MODEL, Client, clip_words, slug

FLAWS = ["repetition_or_loops", "incoherent_or_nonsensical", "drifts_off_topic", "unrelated_words_or_insertions",
         "factual_or_logical_error", "ignores_instructions", "garbled_or_broken_text"]
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


def accept(j):
    try:
        ok = (j["addresses_task"] in ("yes", "partly", "no") and j["severity"] in ("none", "minor", "major")
              and isinstance(j["flaws"], list) and 1 <= int(j["quality"]) <= 5)
    except (KeyError, TypeError, ValueError):
        return None
    if not ok:
        return None
    j["quality"] = int(j["quality"])
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--arms", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    prompts = load_prompts(a.prompts)
    client = Client(a.model)
    for run in a.run:
        rd = Path(run)
        gens = [g for g in load_generations(rd) if g["prompt_id"] in prompts and (not a.arms or g["arm"] in set(a.arms.split(",")))]
        log = rd / f"flaws_{slug(a.model)}.jsonl"
        done = {r["id"] for r in read_jsonl(log)}
        jid = lambda g: short_hash(g["id"], a.max_words)
        todo = [g for g in gens if jid(g) not in done]
        print(f"[{rd.name}] judge={a.model}: {len(todo)} texts to rate ({len(done)} done)", flush=True)
        t0 = time.time()

        def work(g):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(task=clip_words(prompts[g["prompt_id"]]["prompt"], 400),
                                                            r=clip_words(g["text"], a.max_words), flaws=", ".join(FLAWS))}]
            j = client.chat(msgs, schema=SCHEMA, accept=accept, max_tokens=1200)
            return None if j is None else {"id": jid(g), "gen_id": g["id"], "arm": g["arm"], "seed": g.get("seed"),
                                           "prompt_id": g["prompt_id"], "domain": g.get("domain"), "judgment": j}

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for n, rec in enumerate(ex.map(work, todo), 1):
                if rec is not None:
                    append_jsonl(log, rec)
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{rd.name}] tokens used: {dict(client.usage)}", flush=True)


if __name__ == "__main__":
    main()
