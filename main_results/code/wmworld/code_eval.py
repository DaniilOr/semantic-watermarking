"""Execution-based quality for the code domain (MBPP): run each generated solution against the task's test
assertions in a subprocess with a timeout and resource limits, and report pass rates per arm, paired against base.

Generated code is untrusted: it runs as the current user in an isolated temp directory with `python -I`, a CPU-time
and memory limit, and a wall-clock timeout.  No network isolation beyond that; run it on a compute node, not a login node.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from .config import Config
from .data.prepare import load_prepared
from .records import load_jsonl, read_json, write_json

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """The first fenced code block if any, else the whole text (spec: keep failures, do not curate)."""
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip("\n")
    return text.strip()


def _limits(cpu_seconds: int, mem_bytes: int):
    def pre():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        os.setsid()
    return pre


def run_tests(code: str, setup: str, tests: list[str], timeout: float = 10.0, python: str | None = None) -> dict:
    script = (setup or "") + "\n\n" + code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="wm_code_") as d:
        path = Path(d) / "prog.py"
        path.write_text(script, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": d, "LANG": "C.UTF-8"}
        t0 = time.time()
        try:
            r = subprocess.run([python or sys.executable, "-I", str(path)], cwd=d, env=env, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits(int(timeout) + 1, 2 * 1024 ** 3))
            status = "pass" if r.returncode == 0 else "fail"
            err = (r.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            status, err = "timeout", ""
        except Exception as e:  # noqa: BLE001
            status, err = "error", repr(e)[:400]
    return {"status": status, "error": err, "seconds": time.time() - t0}


def evaluate_code_run(cfg: Config, run_dir: Path, timeout: float = 10.0, log=print, python: str | None = None) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    prompts = {p.prompt_id: p for p in load_prepared(cfg.path_of("data_prepared"), partition=manifest["partition"], domains=["mbpp"])}
    from .stages import collect_run_texts
    gens = [g for g in collect_run_texts(run_dir)[0] if g["prompt_id"] in prompts]        # sharded files included
    out_dir = run_dir / "eval"; out_dir.mkdir(exist_ok=True)
    rows_path = out_dir / "code_eval.jsonl"
    done = {}
    if rows_path.exists():
        for r in load_jsonl(rows_path):
            done[r["id"]] = r
    n = 0
    with rows_path.open("a", encoding="utf-8") as f:
        # reference solutions first (sanity: they should pass)
        for pid, p in sorted(prompts.items()):
            rid = f"ref:{pid}"
            if rid in done or not p.reference:
                continue
            res = run_tests(p.reference, p.meta.get("test_setup_code", ""), p.meta.get("test_list", []), timeout, python)
            rec = {"id": rid, "prompt_id": pid, "arm": "reference", **res}
            f.write(json.dumps(rec) + "\n"); done[rid] = rec; n += 1
        for g in gens:
            if g["id"] in done or g.get("status", "ok") != "ok":
                continue
            p = prompts[g["prompt_id"]]
            code = extract_code(g["text"])
            res = run_tests(code, p.meta.get("test_setup_code", ""), p.meta.get("test_list", []), timeout, python)
            rec = {"id": g["id"], "prompt_id": g["prompt_id"], "arm": g["arm"], "fenced": bool(FENCE_RE.search(g["text"])),
                   "code_chars": len(code), **res}
            f.write(json.dumps(rec) + "\n"); done[g["id"]] = rec; n += 1
            if n % 100 == 0:
                log(f"[code-eval] {n} programs executed")
    rows = list(done.values())
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    base = {r["prompt_id"]: r for r in by_arm.get("base", [])}
    summary = {"run": str(run_dir), "timeout": timeout, "n_prompts": len(prompts), "per_arm": {}}
    for arm, rs in by_arm.items():
        passed = sum(r["status"] == "pass" for r in rs)
        paired = [(r["status"] == "pass", base[r["prompt_id"]]["status"] == "pass") for r in rs if r["prompt_id"] in base]
        summary["per_arm"][arm] = {
            "n": len(rs), "pass_rate": passed / max(1, len(rs)),
            "status_counts": {s: sum(r["status"] == s for r in rs) for s in ("pass", "fail", "timeout", "error")},
            "fenced_frac": (sum(r.get("fenced", False) for r in rs) / len(rs)) if arm != "reference" else None,
            "paired_vs_base": ({"n": len(paired), "both_pass": sum(a and b for a, b in paired) / len(paired),
                                "arm_only": sum(a and not b for a, b in paired) / len(paired),
                                "base_only": sum(b and not a for a, b in paired) / len(paired),
                                "pass_rate_delta": (sum(a for a, _ in paired) - sum(b for _, b in paired)) / len(paired)}
                               if paired and arm != "base" else None),
        }
    write_json(out_dir / "code_eval.json", summary)
    for arm, s in summary["per_arm"].items():
        log(f"[code-eval] {arm:14s} n={s['n']:4d} pass={s['pass_rate']:.3f} {s['status_counts']}"
            + (f" paired delta vs base {s['paired_vs_base']['pass_rate_delta']:+.3f}" if s.get("paired_vs_base") else ""))
    return summary
