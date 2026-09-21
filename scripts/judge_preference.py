import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import _path
from wmark.io import append_jsonl, load_generations, load_prompts, read_jsonl, short_hash, write_json
from wmark.openrouter import DEFAULT_MODEL, Client, clip_words, slug

SYSTEM = ("You are a careful, impartial evaluator. You will see a task and two responses (A and B) that AI systems wrote "
          "for that task. You do not know which system wrote which response.")
USER = """Task given to the writer:
<task>
{task}
</task>

<response_a>
{a}
</response_a>

<response_b>
{b}
</response_b>

Both responses were produced under a length limit, so an abrupt ending at the very end is NOT a flaw.

Which response is better overall? Answer with JSON only: {{"better": "A"}}, {{"better": "B"}} or {{"better": "tie"}}."""
SCHEMA = {"name": "preference", "strict": True,
          "schema": {"type": "object", "additionalProperties": False, "required": ["better"],
                     "properties": {"better": {"type": "string", "enum": ["A", "B", "tie"]}}}}


def accept(j):
    return {"better": j["better"]} if isinstance(j, dict) and j.get("better") in ("A", "B", "tie") else None


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        b = r["judgment"]["better"]
        by[r["prompt_id"]].append(0.5 if b == "tie" else (1.0 if b == r["candidate_is"] else 0.0))
    score = np.array([np.mean(v) for v in by.values()])
    cons = [1.0 if v[0] == v[1] else 0.0 for v in by.values() if len(v) == 2]
    ties = np.mean([1.0 if r["judgment"]["better"] == "tie" else 0.0 for r in rows])
    rng = np.random.default_rng(0)
    boot = [float(rng.choice(score, size=score.size, replace=True).mean()) for _ in range(2000)]
    return {"n_prompts": int(score.size), "win_rate": float(score.mean()),
            "win_rate_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "tie_share": float(ties), "order_consistency": float(np.mean(cons)) if cons else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="", help="run:arm of the reference side; default <run>:base")
    ap.add_argument("--arms", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rd = Path(a.run)
    prompts = load_prompts(a.prompts)
    base_run, base_arm = (a.base.split(":") if a.base else (a.run, "base"))
    base = {g["prompt_id"]: g for g in load_generations(base_run) if g["arm"] == base_arm}
    cand = defaultdict(dict)
    for g in load_generations(rd):
        if not (Path(base_run) == rd and g["arm"] == base_arm):
            cand[g["arm"]][g["prompt_id"]] = g
    arms = [x for x in sorted(cand) if not a.arms or x in set(a.arms.split(","))]
    client = Client(a.model)
    log = rd / f"preference_{slug(a.model)}.jsonl"
    done = {r["id"]: r for r in read_jsonl(log)}
    todo = []
    for arm in arms:
        pids = sorted((p for p in cand[arm] if p in base and p in prompts), key=lambda p: short_hash("judge", p))
        if a.limit:
            pids = pids[:a.limit]
        for pid in pids:
            for side in ("A", "B"):
                jid = short_hash(arm, pid, side, cand[arm][pid]["id"], base[pid]["id"], a.max_words)
                if jid not in done:
                    todo.append((jid, arm, pid, side))
    print(f"[{rd.name}] judge={a.model} arms={arms} comparisons: {len(todo)} (done: {len(done)})", flush=True)
    t0 = time.time()

    def work(item):
        jid, arm, pid, side = item
        c, b = clip_words(cand[arm][pid]["text"], a.max_words), clip_words(base[pid]["text"], a.max_words)
        x, y = (c, b) if side == "A" else (b, c)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(task=clip_words(prompts[pid]["prompt"], 400), a=x, b=y)}]
        j = client.chat(msgs, schema=SCHEMA, accept=accept, max_tokens=200)
        return None if j is None else {"id": jid, "arm": arm, "prompt_id": pid, "candidate_is": side,
                                       "domain": prompts[pid].get("domain"), "judgment": j}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, rec in enumerate(ex.map(work, todo), 1):
            if rec is not None:
                done[rec["id"]] = rec
                append_jsonl(log, rec)
            if n % 40 == 0:
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"judge": a.model, "max_words": a.max_words, "base": f"{base_run}:{base_arm}", "usage": dict(client.usage), "per_arm": {}}
    for arm in arms:
        rows = [r for r in done.values() if r["arm"] == arm]
        if rows:
            s = out["per_arm"][arm] = summarize(rows)
            print(f"{rd.name} {arm:14s} n={s['n_prompts']:3d} win={s['win_rate']:.3f} [{s['win_rate_ci95'][0]:.3f},{s['win_rate_ci95'][1]:.3f}] "
                  f"ties={s['tie_share']:.2f} order-consistent={s['order_consistency']}", flush=True)
    write_json(rd / f"preference_{slug(a.model)}.json", out)


if __name__ == "__main__":
    main()
