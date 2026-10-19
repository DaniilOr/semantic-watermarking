"""IFEval adapter (spec Section 7): original prompts, 2,048-token budget, no
minimum length, official evaluator pinned to a commit, strict/loose prompt-
and instruction-level pass rates with counts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .generation import CandidateSelectionGenerator, arm_config
from .keys import assign_keys
from .records import GenerationRecord, JsonlWriter, content_hash, load_jsonl, output_id, read_json, write_json
from .scoring import LikelihoodScorer
from .stages import build_backend, load_key_records, load_scales
from .textnorm import text_hash

IFEVAL_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
IFEVAL_DATASET_SHA = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_BUDGET = 2048


def load_ifeval_prompts(cfg: Config) -> list[dict]:
    p = cfg.path_of("data_raw") / "ifeval" / "ifeval_input_data.jsonl"
    return load_jsonl(p)


def run_ifeval(cfg: Config, arms: list[str] | None, device: str | None, evaluate_only: bool, shard: int = 0, num_shards: int = 1, log=print,
               key_set: str | None = None, limit: int = 0, budget: int = IFEVAL_BUDGET, run_name: str = "ifeval") -> Path:
    """``key_set``/``limit``/``budget``/``run_name`` are labelled extensions: a seed-phrase key set, a fixed hash-ordered
    subset of the prompts (never chosen by content or results), a smaller token budget, and a separate run directory."""
    run_dir = cfg.path_of("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_ifeval_prompts(cfg)
    if limit:
        prompts = sorted(prompts, key=lambda p: content_hash({"ifeval_subset": str(p["key"])}, length=16))[:limit]
    proto = cfg.raw.get("_protocol")
    if arms is None:
        if proto:
            sel = read_json(proto["path"])["selected_configurations"]
            arms = ["base"] + [s["arm"] for s in sel]
        else:
            arms = ["base", "llm_global"]
    key_set = key_set or ("confirmation" if proto else "development")
    if not evaluate_only:
        keys = load_key_records(cfg, key_set)
        assignment = assign_keys([str(p["key"]) for p in prompts], [k.key_id for k in keys], cfg.seed, "ifeval")
        kmap = {k.key_id: k for k in keys}
        backend = build_backend(cfg, device)
        scorer = LikelihoodScorer(backend, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
        arm_cfgs = {a: arm_config(a, cfg.generation_config(None, {"max_new_tokens": budget, "min_new_tokens": 0}), cfg.arm_overrides(a)) for a in arms}
        scales = load_scales(cfg, "dev") if any(g.objective in ("global", "local", "specific") for g in arm_cfgs.values()) else None
        reference = ref_map = None
        if any(g.poe_reference for g in arm_cfgs.values()):        # cross-model steering: a second checkpoint is the world
            from .reference import build_reference
            reference = build_reference(cfg)
            ref_map = reference.token_map(backend, cache_path=cfg.path_of("assets") / f"token_map_{Path(reference.cfg.model_id).name}.npz", log=log)
            log(f"[ifeval] cross-model world: {reference.info['model_id']}")
        gen = CandidateSelectionGenerator(backend, scorer, scales, reference=reference, ref_token_map=ref_map)
        base_gen = cfg.generation_config(None, {"max_new_tokens": budget, "min_new_tokens": 0})
        mine = [p for i, p in enumerate(prompts) if i % num_shards == shard]
        for arm in arms:
            gcfg = arm_config(arm, base_gen, cfg.arm_overrides(arm))
            ch = content_hash({"arm": arm, "gen": gcfg.to_dict(), "ifeval": IFEVAL_DATASET_SHA}, length=12)
            suffix = f".shard{shard}" if num_shards > 1 else ""
            w = JsonlWriter(run_dir / f"generations_{arm}{suffix}.jsonl")
            t0 = time.time(); n = 0
            for p in mine:
                pid = str(p["key"])
                gid = f"ifeval.{pid}"
                key = kmap[assignment[pid]] if (gcfg.objective in ("global", "local", "specific") or gcfg.prefix_in_generator or gcfg.steer_alpha > 0 or gcfg.poe_beta > 0) else None
                oid = output_id("ifeval", arm, gid, key.key_id if key else None, 0, ch)
                if oid in w:
                    continue
                res = gen.generate(p["prompt"], key.to_keyspec() if key else None, gcfg, 0, oid)
                w.write(GenerationRecord(id=oid, stage="ifeval", arm=arm, group_id=gid, prompt_id=gid, domain="ifeval", key_id=key.key_id if key else None,
                                         seed=0, text=res.text, text_hash=res.text_hash, accepted_tokens=len(res.accepted_tokens), visible_tokens=res.visible_tokens,
                                         stop_reason=res.stop_reason, ends_mid_sentence=res.ends_mid_sentence, config_hash=ch, status=res.status,
                                         error=res.error, cost=asdict(res.cost), assigned_key_id=assignment[pid], arm_config=gcfg.to_dict()))
                n += 1
                if n % 10 == 0:
                    log(f"[ifeval] {arm}: {n}/{len(mine)} ({time.time() - t0:.0f}s)")
            w.close()
        return run_dir
    # ---- evaluation with the pinned official evaluator ------------------------------------ #
    results = {"evaluator_commit": IFEVAL_COMMIT, "dataset_sha": IFEVAL_DATASET_SHA, "budget_tokens": budget,
               "n_prompts": len(prompts), "subset": bool(limit), "key_set": key_set, "arms": {}}
    evaluator_root = cfg.root / "third_party" / "google_research_ifeval"
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)
    for arm in arms:
        gens = []
        for p in sorted(run_dir.glob(f"generations_{arm}*.jsonl")):
            gens.extend(load_jsonl(p))
        by_pid = {g["prompt_id"]: g for g in gens}
        resp_path = run_dir / f"responses_{arm}.jsonl"
        n_written = 0
        inp_path = run_dir / f"input_{arm}.jsonl"        # the evaluator needs a response for EVERY input row: give it the answered subset
        with resp_path.open("w", encoding="utf-8") as f, inp_path.open("w", encoding="utf-8") as fi:
            for p in prompts:
                g = by_pid.get(f"ifeval.{p['key']}")
                if g is None:
                    continue
                f.write(json.dumps({"prompt": p["prompt"], "response": g["text"]}, ensure_ascii=False) + "\n")
                fi.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_written += 1
        out_dir = run_dir / f"eval_{arm}"
        out_dir.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "instruction_following_eval.evaluation_main",
               f"--input_data={inp_path}",
               f"--input_response_data={resp_path}", f"--output_dir={out_dir}"]
        proc = subprocess.run(cmd, cwd=str(evaluator_root), capture_output=True, text=True)
        (out_dir / "evaluator_stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
        arm_res = {"n_prompts_with_responses": n_written, "n_prompts_total": len(prompts), "returncode": proc.returncode}
        for mode in ("strict", "loose"):
            rows = load_jsonl(out_dir / f"eval_results_{mode}.jsonl") if (out_dir / f"eval_results_{mode}.jsonl").exists() else []
            if rows:
                pl = sum(1 for r in rows if r["follow_all_instructions"]) / len(rows)
                inst = [x for r in rows for x in r["follow_instruction_list"]]
                arm_res[mode] = {"prompt_level": pl, "n_prompts": len(rows), "instruction_level": sum(inst) / len(inst), "n_instructions": len(inst)}
            else:
                arm_res[mode] = None
        results["arms"][arm] = arm_res
        log(f"[ifeval] {arm}: {json.dumps(arm_res)}")
    write_json(run_dir / "results.json", results)
    return run_dir
