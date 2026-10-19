"""Blind pairwise judge through an external API model (OpenRouter).  No local judge model is loaded.

For every prompt the judge sees the task and two anonymous responses -- the clean one and the watermarked one --
in BOTH presentation orders (position bias cancels).  It never sees the key, the arm, or any watermark score.
For each response it reports whether the response actually addresses the task and which flaws it has; then whether
both responses are attempts at the SAME question, and which one is better.

Reported per arm (paired with the clean response to the same prompt):
  win rate            P(watermarked better) + 0.5 P(tie); 0.50 = the judge cannot tell which is better
  on-task             does the response address the task (yes=1, partly=0.5, no=0): watermarked vs clean
  same question       share of pairs where both responses answer the same question
  flaws               any / major flaw rate, watermarked vs clean, and the per-category counts
  order consistency   how often the verdict survives swapping A and B (the judge's own reliability)

The key is read from $OPENROUTER_API_KEY or ~/.config/wmworld/openrouter.key and is never printed or stored.
Judgments are appended to runs/<run>/eval/api_judge_<model>.jsonl, so an interrupted run resumes for free.

Usage: python scripts/api_judge.py --config configs/pilot.yaml --run runs/<name> [--model google/gemma-4-31b-it]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
FLAWS = ["repetition_or_loops", "incoherent_or_nonsensical", "drifts_off_topic", "unrelated_words_or_insertions",
         "factual_or_logical_error", "ignores_instructions", "garbled_or_broken_text"]
SYSTEM = ("You are a careful, impartial evaluator. You will see a task and two responses (A and B) that AI systems wrote "
          "for that task. You do not know which system wrote which response. Assess each response on its own, then "
          "compare them. Be strict about real defects and do not reward length for its own sake.")
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

Important: both responses were produced under a length limit, so an abrupt ending at the very end is NOT a flaw.
Judge the content that is there.

For EACH response decide:
- addresses_task: "yes", "partly" or "no" -- does it actually attempt the task that was given (the same question, the same subject)?
- flaws: every flaw that applies, from this list only: {flaws}. Use an empty list if there is none.
  ("unrelated_words_or_insertions" means words, names or topics that do not belong in the text and appear out of nowhere.)
- severity: "none", "minor" or "major" -- how much the flaws damage the response.
- note: at most 15 words naming the main problem, or "ok".
Then decide:
- same_question: true if both responses are attempts at the same task/question (even if their quality differs), false if one of them is answering something else.
- better: "A", "B" or "tie" -- which response is better overall (relevance to the task, coherence, fluency, usefulness).
- margin: "slight" or "clear".
Answer with JSON only."""

PREF_SYSTEM = ("You are a careful, impartial evaluator. You will see a task and two responses (A and B) that AI systems wrote "
               "for that task. You do not know which system wrote which response.")
PREF_USER = """Task given to the writer:
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
PREF_SCHEMA = {"name": "preference", "strict": True,
               "schema": {"type": "object", "additionalProperties": False, "required": ["better"],
                          "properties": {"better": {"type": "string", "enum": ["A", "B", "tie"]}}}}


_RESP = {"type": "object", "additionalProperties": False, "required": ["addresses_task", "flaws", "severity", "note"],
         "properties": {"addresses_task": {"type": "string", "enum": ["yes", "partly", "no"]},
                        "flaws": {"type": "array", "items": {"type": "string", "enum": FLAWS}},
                        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
                        "note": {"type": "string"}}}
SCHEMA = {"name": "pairwise_judgment", "strict": True,
          "schema": {"type": "object", "additionalProperties": False,
                     "required": ["a", "b", "same_question", "better", "margin"],
                     "properties": {"a": _RESP, "b": _RESP, "same_question": {"type": "boolean"},
                                    "better": {"type": "string", "enum": ["A", "B", "tie"]},
                                    "margin": {"type": "string", "enum": ["slight", "clear"]}}}}


def api_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        p = Path.home() / ".config" / "wmworld" / "openrouter.key"
        k = p.read_text().strip() if p.exists() else ""
    if not k:
        raise SystemExit("no API key: set OPENROUTER_API_KEY or create ~/.config/wmworld/openrouter.key (mode 600)")
    return k


def clip_words(text: str, n: int) -> str:
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."


def parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def valid(j: dict | None) -> bool:
    try:
        return (j["better"] in ("A", "B", "tie") and isinstance(j["same_question"], bool)
                and all(j[s]["addresses_task"] in ("yes", "partly", "no") and j[s]["severity"] in ("none", "minor", "major")
                        and isinstance(j[s]["flaws"], list) for s in ("a", "b")))
    except (KeyError, TypeError):
        return False


class Judge:
    def __init__(self, model: str, key: str, mode: str = "full"):
        self.model, self.key, self.mode = model, key, mode       # mode "preference": ONLY "which is better" (flaws are judged separately)
        self.structured, self.no_reasoning = True, True
        self.usage = Counter()

    def ask(self, task: str, a: str, b: str) -> dict | None:
        pref = self.mode == "preference"
        body = {"model": self.model, "temperature": 0, "seed": 0, "max_tokens": 200 if pref else 1500,
                "messages": [{"role": "system", "content": PREF_SYSTEM if pref else SYSTEM},
                             {"role": "user", "content": (PREF_USER.format(task=task, a=a, b=b) if pref
                                                          else USER.format(task=task, a=a, b=b, flaws=", ".join(FLAWS)))}]}
        last = ""
        for attempt in range(6):
            req = dict(body)
            if self.structured:
                req["response_format"] = {"type": "json_schema", "json_schema": PREF_SCHEMA if pref else SCHEMA}
            if self.no_reasoning:
                req["reasoning"] = {"enabled": False}
            try:
                r = requests.post(URL, json=req, timeout=180, headers={"Authorization": f"Bearer {self.key}",
                                                                       "X-Title": "wmworld quality judge"})
            except requests.RequestException as e:
                last = f"network: {type(e).__name__}"
                time.sleep(min(60, 2 ** attempt)); continue
            if r.status_code == 200:
                d = r.json()
                if "error" in d:
                    last = f"provider error: {str(d['error'])[:200]}"
                    time.sleep(min(60, 2 ** attempt)); continue
                u = d.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                j = parse_json(d["choices"][0]["message"].get("content") or "")
                if pref and isinstance(j, dict) and j.get("better") in ("A", "B", "tie"):
                    return {"better": j["better"]}
                if not pref and valid(j):
                    return j
                last = "unparseable judgment"
                continue
            txt = r.text[:300].lower()
            if r.status_code in (400, 404, 422) and self.no_reasoning and "reasoning" in txt:
                self.no_reasoning = False; continue           # this model cannot switch reasoning off: let it think
            if r.status_code in (400, 404, 422) and self.structured and ("response_format" in txt or "schema" in txt or "structured" in txt):
                self.structured = False; continue             # no schema support on this route: ask for plain JSON
            if r.status_code in (401, 402, 403):
                raise SystemExit(f"API refused the request (HTTP {r.status_code}): check the key / credits")
            last = f"HTTP {r.status_code}"
            time.sleep(min(60, 2 ** attempt))
        print(f"  [judge] gave up on one comparison ({last})", flush=True)
        return None


def load_prompts(cfg, run_dir: Path) -> dict[str, str]:
    """prompt_id -> task text, for ordinary stages (prepared data) and for the IFEval run (its own prompt file)."""
    mf = run_dir / "manifest.json"
    if mf.exists():
        from wmworld.data.prepare import load_prepared
        part = json.loads(mf.read_text()).get("partition")
        for prefix, path in (("ob_bench", "data/ob_bench/prompts.jsonl"), ("sp_bench", "data/sp_bench/prompts.jsonl"), ("it_bench", "data/it_bench/prompts.jsonl"), ("ss_train", "data/ss_train/prompts.jsonl")):
            if str(part).startswith(prefix):                  # benchmark / training-prompt runs keep their own prompt files
                return {r["prompt_id"]: r["prompt"] for r in map(json.loads, open(path))}
        return {p.prompt_id: p.prompt for p in load_prepared(cfg.path_of("data_prepared"), partition=part)}
    from wmworld.ifeval import load_ifeval_prompts
    return {f"ifeval.{p['key']}": p["prompt"] for p in load_ifeval_prompts(cfg)}


def summarize(rows: list[dict]) -> dict:
    """rows: one per (prompt, order); 'wm_is' says whether the watermarked text was shown as A or B."""
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["prompt_id"]].append(r)
    score, on_wm, on_base, same, any_wm, any_base, maj_wm, maj_base, consistent = [], [], [], [], [], [], [], [], []
    cat_wm, cat_base = Counter(), Counter()
    onv = {"yes": 1.0, "partly": 0.5, "no": 0.0}
    for pid, rs in by_pair.items():
        s, ow, ob, sq, aw, ab, mw, mb, verdicts = [], [], [], [], [], [], [], [], []
        for r in rs:
            j, wm, base = r["judgment"], r["wm_is"].lower(), ("b" if r["wm_is"] == "A" else "a")
            v = 0.5 if j["better"] == "tie" else (1.0 if j["better"] == r["wm_is"] else 0.0)
            s.append(v); verdicts.append(v)
            ow.append(onv[j[wm]["addresses_task"]]); ob.append(onv[j[base]["addresses_task"]])
            sq.append(1.0 if j["same_question"] else 0.0)
            aw.append(1.0 if j[wm]["flaws"] else 0.0); ab.append(1.0 if j[base]["flaws"] else 0.0)
            mw.append(1.0 if j[wm]["severity"] == "major" else 0.0); mb.append(1.0 if j[base]["severity"] == "major" else 0.0)
            for f in set(j[wm]["flaws"]): cat_wm[f] += 1.0 / len(rs)
            for f in set(j[base]["flaws"]): cat_base[f] += 1.0 / len(rs)
        score.append(np.mean(s)); on_wm.append(np.mean(ow)); on_base.append(np.mean(ob)); same.append(np.mean(sq))
        any_wm.append(np.mean(aw)); any_base.append(np.mean(ab)); maj_wm.append(np.mean(mw)); maj_base.append(np.mean(mb))
        if len(verdicts) == 2:
            consistent.append(1.0 if verdicts[0] == verdicts[1] else 0.0)
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, dtype=np.float64)
        b = [float(rng.choice(x, size=x.size, replace=True).mean()) for _ in range(2000)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    n = len(score)
    d_on = np.asarray(on_wm) - np.asarray(on_base); d_maj = np.asarray(maj_wm) - np.asarray(maj_base)
    d_any = np.asarray(any_wm) - np.asarray(any_base)
    return {"n_prompts": n, "win_rate": float(np.mean(score)), "win_rate_ci95": ci(score),
            "on_task_wm": float(np.mean(on_wm)), "on_task_clean": float(np.mean(on_base)),
            "on_task_delta": float(d_on.mean()), "on_task_delta_ci95": ci(d_on),
            "same_question": float(np.mean(same)),
            "any_flaw_wm": float(np.mean(any_wm)), "any_flaw_clean": float(np.mean(any_base)),
            "any_flaw_delta": float(d_any.mean()), "any_flaw_delta_ci95": ci(d_any),
            "major_flaw_wm": float(np.mean(maj_wm)), "major_flaw_clean": float(np.mean(maj_base)),
            "major_flaw_delta": float(d_maj.mean()), "major_flaw_delta_ci95": ci(d_maj),
            "flaws_wm_per_100": {k: round(100 * v / n, 1) for k, v in sorted(cat_wm.items())},
            "flaws_clean_per_100": {k: round(100 * v / n, 1) for k, v in sorted(cat_base.items())},
            "order_consistency": float(np.mean(consistent)) if consistent else None}


def summarize_preference(rows: list[dict]) -> dict:
    """Pure preference: per prompt the mean over the two orders of (1 watermarked better, 0.5 tie, 0 clean better)."""
    by = defaultdict(list)
    for r in rows:
        b = r["judgment"]["better"]
        by[r["prompt_id"]].append(0.5 if b == "tie" else (1.0 if b == r["wm_is"] else 0.0))
    score = np.array([np.mean(v) for v in by.values()])
    cons = [1.0 if v[0] == v[1] else 0.0 for v in by.values() if len(v) == 2]
    ties = np.mean([1.0 if r["judgment"]["better"] == "tie" else 0.0 for r in rows])
    rng = np.random.default_rng(0)
    boot = [float(rng.choice(score, size=score.size, replace=True).mean()) for _ in range(2000)]
    return {"n_prompts": int(score.size), "win_rate": float(score.mean()), "win_rate_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "tie_share": float(ties), "order_consistency": float(np.mean(cons)) if cons else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--mode", choices=["full", "preference"], default="full",
                    help="preference: ask ONLY which response is better (flaws are extracted separately by api_pointwise.py)")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--arms", default="")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts per arm (stable hash order)")
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    judge = Judge(args.model, api_key(), mode=args.mode)
    slug = args.model.replace("/", "_").replace(":", "_")
    stem = "api_pref" if args.mode == "preference" else "api_judge"
    for run in args.run:
        rd = Path(run)
        prompts = load_prompts(cfg, rd)
        gens, _ = collect_run_texts(rd)
        if os.environ.get("JUDGE_DOMAINS"):                         # e.g. only the finished datasets of a run
            gens = [g for g in gens if g.get("domain") in os.environ["JUDGE_DOMAINS"].split()]
        by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
        for g in gens:
            if g.get("status", "ok") == "ok" and g.get("text"):
                by_prompt[g["prompt_id"]][g["arm"]] = g
        arms = sorted({a for v in by_prompt.values() for a in v if a != "base"})
        if args.arms:
            arms = [a for a in arms if a in set(args.arms.split(","))]
        (rd / "eval").mkdir(exist_ok=True)
        log_path = rd / "eval" / f"{stem}_{slug}.jsonl"
        done = {}
        if log_path.exists():
            for ln in open(log_path):
                r = json.loads(ln); done[r["id"]] = r
        todo = []
        for arm in arms:
            pids = sorted((p for p, v in by_prompt.items() if "base" in v and arm in v and p in prompts),
                          key=lambda p: hashlib.sha256(p.encode()).hexdigest())
            if args.limit:
                pids = pids[: args.limit]
            for pid in pids:
                for wm_is in ("A", "B"):
                    jid = hashlib.sha256(f"{arm}|{pid}|{wm_is}|{by_prompt[pid][arm]['text_hash']}|{by_prompt[pid]['base']['text_hash']}".encode()).hexdigest()[:20]
                    if jid not in done:
                        todo.append((jid, arm, pid, wm_is))
        print(f"[{rd.name}] judge={args.model} arms={arms} comparisons to run: {len(todo)} (already done: {len(done)})", flush=True)
        t0 = time.time()

        def work(item):
            jid, arm, pid, wm_is = item
            wm = clip_words(by_prompt[pid][arm]["text"], args.max_words)
            base = clip_words(by_prompt[pid]["base"]["text"], args.max_words)
            a, b = (wm, base) if wm_is == "A" else (base, wm)
            j = judge.ask(clip_words(prompts[pid], 400), a, b)
            return None if j is None else {"id": jid, "arm": arm, "prompt_id": pid, "wm_is": wm_is,
                                           "domain": by_prompt[pid][arm].get("domain"), "judgment": j}
        with ThreadPoolExecutor(max_workers=args.workers) as ex, open(log_path, "a") as fh:
            for n, rec in enumerate(ex.map(work, todo), 1):
                if rec is not None:
                    done[rec["id"]] = rec
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                if n % 40 == 0:
                    print(f"  [{rd.name}] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        out = {"judge": args.model, "mode": args.mode, "max_words": args.max_words, "usage": dict(judge.usage), "per_arm": {}, "per_arm_domain": {}}
        for arm in arms:
            rows = [r for r in done.values() if r["arm"] == arm]
            if not rows:
                continue
            if args.mode == "preference":
                s = out["per_arm"][arm] = summarize_preference(rows)
                for dom in sorted({r.get("domain") or "" for r in rows}):
                    out["per_arm_domain"][f"{arm}|{dom}"] = summarize_preference([r for r in rows if (r.get("domain") or "") == dom])
                print(f"{rd.name:12s} {arm:12s} n={s['n_prompts']:3d} PREFERENCE-ONLY win={s['win_rate']:.3f} [{s['win_rate_ci95'][0]:.3f},{s['win_rate_ci95'][1]:.3f}] "
                      f"ties={s['tie_share']:.2f} order-consistent={s['order_consistency']}", flush=True)
                continue
            s = out["per_arm"][arm] = summarize(rows)
            for dom in sorted({r.get("domain") or "" for r in rows}):
                out["per_arm_domain"][f"{arm}|{dom}"] = summarize([r for r in rows if (r.get("domain") or "") == dom])
            print(f"{rd.name:12s} {arm:12s} n={s['n_prompts']:3d} win={s['win_rate']:.3f} [{s['win_rate_ci95'][0]:.3f},{s['win_rate_ci95'][1]:.3f}] "
                  f"on-task wm/clean={s['on_task_wm']:.2f}/{s['on_task_clean']:.2f} same-q={s['same_question']:.2f} "
                  f"major-flaw wm/clean={s['major_flaw_wm']:.2f}/{s['major_flaw_clean']:.2f} order-consistent={s['order_consistency']}", flush=True)
        write_json(rd / "eval" / f"{stem}_{slug}.json", out)
        print(f"[{rd.name}] tokens used: {dict(judge.usage)}", flush=True)


if __name__ == "__main__":
    main()
