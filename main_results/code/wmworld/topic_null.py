"""Topic-matched null challenge (spec Section 11).

For each key of the confirmation set (or development set when no protocol is
given), generate five ordinary ``base`` responses to frozen task templates
about the key's subject, with no watermark instruction and no steering.  The
outputs are then scored under that key by ``evaluate`` and reported as a
separate domain-shift false-positive diagnostic; thresholds are never retuned.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .generation import CandidateSelectionGenerator, arm_config
from .key_catalog import TEMPLATES, TOPIC_NULL_TASKS
from .keys import KeyRecord
from .records import GenerationRecord, JsonlWriter, content_hash, output_id, write_json
from .stages import build_backend, load_key_records
from .textnorm import text_hash


def topic_tasks(key: KeyRecord) -> list[tuple[str, str]]:
    """Five (task_text, subject) pairs for a key; multi-statement keys cycle through their statements."""
    if key.family == "random_digits":
        return [(t.format(x=key.prefix), key.prefix) for t in TOPIC_NULL_TASKS["digits"]]
    subjects = []
    for tid, comp in zip(key.template_ids, key.completions):
        subjects.append((TEMPLATES[tid]["category"], comp))
    out = []
    for i in range(5):
        cat, comp = subjects[i % len(subjects)]
        out.append((TOPIC_NULL_TASKS[cat][i].format(x=comp), comp))
    return out


def run_topic_null(cfg: Config, device: str | None = None, log=print) -> Path:
    key_set = "confirmation" if cfg.raw.get("_protocol") else "development"
    keys = load_key_records(cfg, key_set)
    run_dir = cfg.path_of("runs") / "topic_null"
    run_dir.mkdir(parents=True, exist_ok=True)
    gcfg = arm_config("base", cfg.generation_config("positive_test" if key_set == "confirmation" else "dev"))
    ch = content_hash({"arm": "base", "gen": gcfg.to_dict(), "topic_null": True}, length=12)
    backend = build_backend(cfg, device)
    gen = CandidateSelectionGenerator(backend, None, None)
    w = JsonlWriter(run_dir / "generations.jsonl")
    tasks_manifest = {}
    t0 = time.time()
    n = 0
    for k in keys:
        tasks = topic_tasks(k)
        tasks_manifest[k.key_id] = [{"task": t, "subject": s} for t, s in tasks]
        for i, (task, subject) in enumerate(tasks):
            gid = f"topic.{k.key_id}.{i}"
            oid = output_id("topic_null", "base", gid, None, 0, ch)
            if oid in w:
                continue
            res = gen.generate(task, None, gcfg, 0, oid)
            w.write(GenerationRecord(id=oid, stage="topic_null", arm="base", group_id=gid, prompt_id=gid, domain="topic_null",
                                     key_id=None, seed=0, text=res.text, text_hash=res.text_hash, accepted_tokens=len(res.accepted_tokens),
                                     visible_tokens=res.visible_tokens, stop_reason=res.stop_reason, ends_mid_sentence=res.ends_mid_sentence,
                                     config_hash=ch, status=res.status, error=res.error, cost=asdict(res.cost), assigned_key_id=k.key_id,
                                     arm_config={"topic_task": task, "subject": subject}))
            n += 1
            if n % 10 == 0:
                log(f"[topic-null] {n} outputs ({time.time() - t0:.0f}s)")
    w.close()
    write_json(run_dir / "manifest.json", {"stage": "topic_null", "partition": "topic_null", "arms": ["base"], "seeds": [0], "key_set": key_set,
                                           "n_prompts": 5 * len(keys), "assignment": {f"topic.{k}.{i}": k for k in tasks_manifest for i in range(5)},
                                           "tasks": tasks_manifest, "config_hash": cfg.content_hash(),
                                           "note": "ordinary generator responses about each key's subject; a domain-shift false-positive diagnostic"})
    log(f"[topic-null] wrote {run_dir} ({n} new outputs); evaluate with `evaluate --run {run_dir}`")
    return run_dir
