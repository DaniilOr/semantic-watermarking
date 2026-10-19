"""Quality evaluation (spec Section 11): blinded paired comparison forms, an
optional Qwen3.5-0.8B blind judge with a fixed prompt, topic-drift annotation,
and explicit "pending" markers for human review.  The judge sees the task and
the response only; never the key, arm, score or candidate index.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .backend import Backend, SamplingParams
from .config import Config
from .data.prepare import load_prepared
from .protocol import JUDGE_INSTRUCTION, QUALITY_TOLERANCES
from .records import JsonlWriter, load_jsonl, read_json, write_json
from .stages import build_backend, collect_run_texts
from .textnorm import seed_from, stable_hash


def build_forms(cfg: Config, run_dir: Path, compare_arms: list[str], min_groups: int = 100, seed: int | None = None) -> Path:
    """Randomly ordered A/B forms per group (base vs each compared arm), balanced across domains.

    The public form contains prompt + two texts; the private key file maps A/B to arms.
    """
    gens, _ = collect_run_texts(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    seed0 = manifest["seeds"][0]
    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    for g in gens:
        if g["seed"] == seed0 and g["status"] == "ok":
            by_group[g["group_id"]][g["arm"]] = g
    prompts = {p.prompt_id: p for p in load_prepared(cfg.path_of("data_prepared"), partition=manifest["partition"])}
    rng = np.random.default_rng(seed_from("forms", seed if seed is not None else cfg.seed))
    domains = cfg["data"]["domains"]
    per_dom = max(1, min_groups // len(domains))
    chosen = []
    for dom in domains:
        gids = sorted([gid for gid, arms in by_group.items() if "base" in arms and all(a in arms for a in compare_arms)
                       and by_group[gid]["base"]["domain"] == dom], key=lambda x: stable_hash("form", x))
        chosen += gids[:per_dom]
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    forms = JsonlWriter(out_dir / "forms.jsonl")
    keyfile = JsonlWriter(out_dir / "forms_private_key.jsonl")
    n = 0
    for gid in chosen:
        arms = by_group[gid]
        for arm in compare_arms:
            fid = f"form.{gid}.{arm}"
            if fid in forms:
                continue
            a_first = bool(rng.random() < 0.5)
            first, second = (arms["base"], arms[arm]) if a_first else (arms[arm], arms["base"])
            p = prompts.get(first["prompt_id"])
            forms.write({"id": fid, "domain": first["domain"], "task": p.prompt if p else None, "text_A": first["text"], "text_B": second["text"],
                         "instructions": "Rate each text 1-5 for task relevance, coherence, absence of repetition, factual/constraint "
                                         "preservation and overall usefulness; then state which you prefer (A, B or tie).",
                         "human_rating": None, "human_preference": None, "status": "pending_human_review"})
            keyfile.write({"id": fid, "A": "base" if a_first else arm, "B": arm if a_first else "base", "output_A": first["id"], "output_B": second["id"]})
            n += 1
    forms.close(); keyfile.close()
    write_json(out_dir / "forms_manifest.json", {"n_forms": n, "groups": len(chosen), "compare_arms": compare_arms,
                                                  "human_review": "pending", "tolerances": QUALITY_TOLERANCES})
    return out_dir


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def model_judge(cfg: Config, run_dir: Path, device: str | None = None, max_outputs: int | None = None, log=print) -> Path:
    """Exploratory blind self-evaluation with the shared checkpoint (fixed prompt; JSON ratings)."""
    gens, _ = collect_run_texts(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    prompts = {p.prompt_id: p for p in load_prepared(cfg.path_of("data_prepared"), partition=manifest["partition"])}
    backend: Backend = build_backend(cfg, device)
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    w = JsonlWriter(out_dir / "model_judge.jsonl")
    n = 0
    for g in gens:
        if g["status"] != "ok" or g["prompt_id"] not in prompts:
            continue
        jid = f"judge.{g['id']}"
        if jid in w:
            continue
        instr = JUDGE_INSTRUCTION.format(task=prompts[g["prompt_id"]].prompt, response=g["text"])
        ctx = backend.chat_prefix_ids("You are a careful, impartial grader.", instr)
        if len(ctx) + 96 > backend.max_input_tokens:
            w.write({"id": jid, "output_id": g["id"], "status": "too_long", "ratings": None, "raw": None})
            continue
        c = backend.sample_units(ctx, 1, 96, SamplingParams(temperature=0.0, top_p=1.0, top_k=0), seed_from("judge", g["id"]), eos_allowed_from=0)[0]
        raw = backend.decode(c.token_ids)
        d = _parse_json(raw)
        ok = d is not None and all(k in d for k in ("relevance", "coherence", "repetition", "constraints", "overall"))
        w.write({"id": jid, "output_id": g["id"], "arm": g["arm"], "domain": g["domain"], "status": "ok" if ok else "unparsed",
                 "ratings": {k: d.get(k) for k in ("relevance", "coherence", "repetition", "constraints", "overall")} if ok else None,
                 "off_topic_subject": (d or {}).get("off_topic_subject"), "raw": raw})
        n += 1
        if max_outputs and n >= max_outputs:
            break
        if n % 25 == 0:
            log(f"[judge] {n} rated")
    w.close()
    return out_dir / "model_judge.jsonl"


def summarize_judge(run_dir: Path) -> dict:
    rows = load_jsonl(run_dir / "quality" / "model_judge.jsonl")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["ratings"]:
            try:
                by_arm[r["arm"]].append({k: float(v) for k, v in r["ratings"].items()})
            except (TypeError, ValueError):
                continue
    out = {}
    for arm, rs in by_arm.items():
        out[arm] = {k: float(np.mean([r[k] for r in rs])) for k in ("relevance", "coherence", "repetition", "constraints", "overall")}
        out[arm]["n"] = len(rs)
        out[arm]["off_topic_frac"] = float(np.mean([bool((r0.get("off_topic_subject") or "").strip()) for r0 in rows if r0.get("arm") == arm and r0["status"] == "ok"]))
    return {"model_judge": out, "note": "exploratory self-evaluation: generator, paraphraser and judge share weights", "human_review": "pending"}
