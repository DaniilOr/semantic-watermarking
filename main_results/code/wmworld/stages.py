"""Stage orchestration (spec Sections 8 and 12): work-item enumeration,
``--dry-run`` budget manifests, resumable sharded generation, the
``fit-scorer`` stage and scoring of a finished run.

Every stage is processed separately; no command launches other stages.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .backend import Backend
from .config import Config
from .data.prepare import PreparedExample, load_prepared
from .generation import ARM_PRESETS, CandidateSelectionGenerator, GenerationConfig, arm_config
from .keys import KeyRecord, assign_keys, load_keys
from .records import (CostRecord, GenerationRecord, JsonlWriter, ReferenceRecord, content_hash, load_jsonl, output_id,
                      read_json, write_json)
from .scales import (BIN_NAMES, ScaleTable, entry_key, fit_scale_table, length_bin, unit_setting_name, within_pool_sd)
from .scoring import KeySpec, LikelihoodScorer
from .textnorm import seed_from, stable_hash, text_hash

SCORING_WRAPPER_TOKENS = 40   # approximate keyed/neutral context length (measured exactly at run time)


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #

def build_backend(cfg: Config, device: str | None = None) -> Backend:
    from .runtime import QwenRuntime
    rc = cfg.runtime_config()
    if device:
        rc.device = device
    return QwenRuntime(rc)


def keys_path(cfg: Config, key_set: str) -> Path:
    return cfg.path_of("assets") / f"keys_{key_set}.json"


def load_key_records(cfg: Config, key_set: str) -> list[KeyRecord]:
    p = keys_path(cfg, key_set)
    if not p.exists():
        raise FileNotFoundError(f"key manifest {p} missing; run make-keys --set {key_set}")
    return load_keys(read_json(p))


STEERING_DECOYS_PER_FAMILY = 4
RW_STEERING_DECOY_POOL = 12       # seed-phrase decoy pool; per key, phrases sharing a word with the key are skipped


def steering_decoys(cfg: Config) -> dict[str, list[dict]]:
    """Fixed steering decoys per family for the 'specific' objective (amendment): the first
    STEERING_DECOYS_PER_FAMILY decoys of each family in the decoy bank, by key id.  Written to
    assets/steering_decoys.json on first use so the assignment is frozen and auditable."""
    p = cfg.path_of("assets") / "steering_decoys.json"
    by_family: dict[str, list[dict]] = read_json(p)["by_family"] if p.exists() else {}
    changed = False
    bank = keys_path(cfg, "decoy")
    if not by_family and bank.exists():
        for k in sorted(read_json(bank)["keys"], key=lambda k: k["key_id"]):
            if k.get("guard_status", "ok") != "ok":
                continue
            fam = by_family.setdefault(k["family"], [])
            if len(fam) < STEERING_DECOYS_PER_FAMILY:
                fam.append({"key_id": k["key_id"], "prefix": k["prefix"], "family": k["family"], "length_stratum": k.get("length_stratum")})
        changed = bool(by_family)
    if len(by_family.get("random_words", [])) < RW_STEERING_DECOY_POOL:
        # seed-phrase decoy POOL (v2): 12 PRNG phrases from the full seed vocabulary; keyspec_with_decoys picks, per target
        # key, the first phrases that share no word with that key.  (v1 drew 4 phrases excluding the words of the key sets
        # that existed at the time; the runs before 2026-09-15 used that pool, kept in git history.)
        from .key_catalog import SEED_WORDS
        rng = np.random.default_rng([cfg.seed, 5, 2])
        phrases: list[str] = []
        while len(phrases) < RW_STEERING_DECOY_POOL:
            idx = rng.choice(len(SEED_WORDS), size=4, replace=False)
            ph = " ".join(SEED_WORDS[int(i)] for i in idx)
            if ph not in phrases:
                phrases.append(ph)
        by_family["random_words"] = [{"key_id": f"rwdecoy_{i + 1:03d}", "prefix": ph, "family": "random_words", "length_stratum": "short"}
                                     for i, ph in enumerate(phrases)]
        changed = True
    if changed:
        write_json(p, {"by_family": by_family, "per_family": STEERING_DECOYS_PER_FAMILY, "source": str(bank),
                       "note": "steering decoys for objective 'specific' and poe_contrast 'decoys'; excluded from the detection decoy bank"})
    return by_family


def keyspec_with_decoys(cfg: Config, key: KeyRecord) -> KeySpec:
    ks = key.to_keyspec()
    dec = steering_decoys(cfg).get(key.family, [])
    if key.family == "random_words":
        kw = set(key.prefix.split())
        dec = [d for d in dec if not (set(d["prefix"].split()) & kw)]
    ks.meta["steering_decoys"] = [d["prefix"] for d in dec if d["prefix"] != key.prefix]
    ks.meta["rotation_keys"] = [d["prefix"] for d in dec if d["prefix"] != key.prefix]   # partners for span key rotation
    ks.meta["steering_decoy_ids"] = [d["key_id"] for d in dec if d["prefix"] != key.prefix]
    return ks


def load_scales(cfg: Config, stage: str) -> ScaleTable | None:
    rel = cfg.stage(stage).get("scales")
    if not rel:
        return None
    p = cfg.resolve_path(rel)
    if not p.exists():
        raise FileNotFoundError(f"scale table {p} missing; run fit-scorer first")
    return ScaleTable.from_dict(read_json(p))


def stage_prompts(cfg: Config, stage: str) -> list[PreparedExample]:
    st = cfg.stage(stage)
    rows = load_prepared(cfg.path_of("data_prepared"), partition=st["partition"], domains=list(st.get("domains") or cfg["data"]["domains"]))
    if st.get("ablation_subset_only"):
        rows = [r for r in rows if r.ablation_subset]
    if st.get("limit_per_domain"):
        # deterministic per-domain subset by hash order (never by content or results)
        n = int(st["limit_per_domain"])
        by_dom: dict[str, list] = {}
        for r in sorted(rows, key=lambda r: stable_hash("limit", cfg.seed, stage, r.prompt_id)):
            by_dom.setdefault(r.domain, []).append(r)
        rows = [r for d in by_dom.values() for r in d[:n]]
    rows.sort(key=lambda r: (r.domain, r.prompt_id))
    if not rows:
        raise RuntimeError(f"no prepared prompts for partition {st['partition']}; run prepare-data")
    return rows


# --------------------------------------------------------------------------- #
# work items
# --------------------------------------------------------------------------- #

@dataclass
class WorkItem:
    output_id: str
    stage: str
    arm: str
    seed: int
    prompt: PreparedExample
    key: KeyRecord | None            # key used by generation (None for no-prefix arms)
    assigned_key_id: str | None
    gen_cfg: GenerationConfig
    config_hash: str


def enumerate_stage(cfg: Config, stage: str, arms: list[str] | None = None, seeds: list[int] | None = None,
                    config_overrides: dict | None = None, label: str | None = None) -> tuple[list[WorkItem], dict]:
    st = cfg.stage(stage)
    prompts = stage_prompts(cfg, stage)
    arms = arms or cfg.stage_arms(stage)
    seeds = seeds or list(st.get("seeds", [0]))
    key_set = st.get("keys")
    keys = load_key_records(cfg, key_set) if key_set else []
    # the key-to-prompt schedule depends on the stage only, so ablation/variant runs share their baseline's assignment
    assignment = assign_keys([p.prompt_id for p in prompts], [k.key_id for k in keys], cfg.seed, stage)
    kmap = {k.key_id: k for k in keys}
    base_gen = cfg.generation_config(stage, config_overrides)
    base_gen_control = cfg.generation_config(stage)          # the `base` control never sees --override values
    # explicit overrides (--override / ablations) apply LAST, after the arm preset and per-arm config overrides
    late = {("clip" if k == "standardized_gain_clip" else k): v for k, v in (config_overrides or {}).items()}
    items: list[WorkItem] = []
    arm_hashes = {}
    for arm in arms:
        # the `base` control never receives explicit overrides (it must stay the ordinary generator)
        gcfg = (arm_config("base", base_gen_control, cfg.arm_overrides("base")) if arm == "base"
                else arm_config(arm, base_gen, {**cfg.arm_overrides(arm), **late}))
        ch = content_hash({"arm": arm, "gen": gcfg.to_dict(), "scorer": cfg["scoring"], "model": cfg["models"]["shared"],
                           "textnorm": "v1"}, length=12)
        arm_hashes[arm] = ch
        needs_key = gcfg.objective in ("global", "local", "specific") or gcfg.prefix_in_generator or gcfg.matched_scoring or gcfg.steer_alpha > 0 or gcfg.poe_beta > 0
        for p in prompts:
            akid = assignment.get(p.prompt_id)
            for seed in seeds:
                key = kmap.get(akid) if (needs_key and akid) else None
                if needs_key and key is None:
                    raise RuntimeError(f"arm {arm} needs a key but none assigned to {p.prompt_id}")
                items.append(WorkItem(output_id=output_id(stage, arm, p.group_id, key.key_id if key else None, seed, ch),
                                      stage=stage, arm=arm, seed=seed, prompt=p, key=key, assigned_key_id=akid,
                                      gen_cfg=gcfg, config_hash=ch))
    manifest = {"stage": stage, "partition": st["partition"], "arms": arms, "seeds": seeds, "key_set": key_set,
                "n_prompts": len(prompts), "n_keys": len(keys), "assignment": assignment, "arm_config_hashes": arm_hashes,
                "arm_configs": {a: (arm_config("base", base_gen_control, cfg.arm_overrides("base")) if a == "base"
                                    else arm_config(a, base_gen, {**cfg.arm_overrides(a), **late})).to_dict() for a in arms},
                "config_hash": cfg.content_hash(), "n_items": len(items)}
    return items, manifest


def estimate_costs(items: list[WorkItem], wrapper_tokens: int = SCORING_WRAPPER_TOKENS, prompt_tokens: int = 160) -> dict:
    """Analytic upper bounds (spec Section 12): proposal tokens ~ N * max_new; full rescoring ~ quadratic."""
    tot = {"outputs": 0, "max_proposal_tokens": 0, "max_scoring_tokens": 0, "max_scoring_sequences": 0,
           "max_proposal_forward_calls": 0}
    per_arm: dict[str, dict] = {}
    for it in items:
        g = it.gen_cfg
        unit = g.unit_tokens if g.unit == "tokens" else g.sentence_cap_tokens
        D = math.ceil(g.max_new_tokens / unit)
        prop = g.candidates * g.max_new_tokens
        scoring_tok = 0
        scoring_seq = 0
        if g.objective in ("global", "local", "specific") or g.matched_scoring:
            active = g.active_fraction if g.objective in ("global", "local", "specific") else 1.0
            # scoring contexts per candidate: keyed + neutral, or keyed + steering decoys for the 'specific' objective
            n_ctx = (1 + STEERING_DECOYS_PER_FAMILY) if g.objective == "specific" else 2
            # cumulative texts: sum_d N*n_ctx*(wrapper + d*unit) for d=1..D
            scoring_tok = int(active * (g.candidates * n_ctx * (D * wrapper_tokens + unit * D * (D + 1) / 2)))
            if g.objective == "local":
                scoring_tok = int(active * g.candidates * n_ctx * D * (wrapper_tokens + unit))
            scoring_seq = int(active * g.candidates * n_ctx * D)
        if g.poe_beta > 0:
            # token-level product-of-experts: two world contexts advance with the generator (2 forwards per token + 2 prefills)
            # two world contexts per token plus the final whole-text score (keyed + neutral) at the end
            scoring_tok = 4 * wrapper_tokens + 4 * g.max_new_tokens
            scoring_seq = 2 * (1 + g.max_new_tokens) + 2
            prop = g.max_new_tokens
        fw = D * (1 + unit)  # prefill + one call per generated token in the unit
        a = per_arm.setdefault(it.arm, {"outputs": 0, "max_proposal_tokens": 0, "max_scoring_tokens": 0,
                                        "max_scoring_sequences": 0, "max_proposal_forward_calls": 0})
        for k, v in (("outputs", 1), ("max_proposal_tokens", prop), ("max_scoring_tokens", scoring_tok),
                     ("max_scoring_sequences", scoring_seq), ("max_proposal_forward_calls", fw)):
            a[k] += v
            tot[k] += v
    return {"total": tot, "per_arm": per_arm, "assumptions": {"wrapper_tokens": wrapper_tokens, "prompt_tokens": prompt_tokens,
                                                               "note": "upper bounds; outputs may stop early at EOS"}}


def measure_wrapper_tokens(cfg: Config, encode, chat_prefix_ids) -> int:
    """Longest scoring-context length (keyed or neutral) under the pinned tokenizer, for budget estimates."""
    from .scoring import SCORING_USER, KeySpec
    st = cfg["scoring"]
    lens = [len(chat_prefix_ids(st["system_template"].format(prefix=cfg["keys"]["neutral_prefix"]), st.get("user_message", SCORING_USER)))]
    for key_set in ("development", "confirmation"):
        p = keys_path(cfg, key_set)
        if p.exists():
            for k in read_json(p)["keys"]:
                lens.append(len(chat_prefix_ids(st["system_template"].format(prefix=k["prefix"]), st.get("user_message", SCORING_USER))))
    return max(lens)


def budget_manifest(cfg: Config, stage: str, items: list[WorkItem], manifest: dict, wall_hours: float | None,
                    smoke_benchmark: dict | None, wrapper_tokens: int = SCORING_WRAPPER_TOKENS) -> dict:
    est = estimate_costs(items, wrapper_tokens=wrapper_tokens)
    margin = 1.5
    limits = {"max_outputs": len(items), "max_proposal_tokens": int(est["total"]["max_proposal_tokens"] * margin),
              "max_scoring_tokens": int(est["total"]["max_scoring_tokens"] * margin) + 1,
              "max_wall_seconds": int((wall_hours or cfg["budgets"].get(stage, {}).get("wall_hours", 24)) * 3600)}
    est_wall = None
    if smoke_benchmark:
        pt = smoke_benchmark.get("proposal_tokens_per_second") or 0
        stt = smoke_benchmark.get("scoring_tokens_per_second") or 0
        if pt and stt:
            est_wall = est["total"]["max_proposal_tokens"] / pt + est["total"]["max_scoring_tokens"] / stt
    b = {"stage": stage, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": cfg.content_hash(),
         "enumeration": {k: v for k, v in manifest.items() if k not in ("assignment", "arm_configs")},
         "estimates": est, "estimated_wall_seconds_1gpu": est_wall, "smoke_benchmark": smoke_benchmark,
         "limits": limits, "model": cfg["models"]["shared"]}
    b["content_hash"] = content_hash(b)
    return b


# --------------------------------------------------------------------------- #
# generation stage
# --------------------------------------------------------------------------- #

class BudgetExceeded(RuntimeError):
    pass


def run_generate_stage(cfg: Config, stage: str, backend: Backend, run_dir: Path, shard: int = 0, num_shards: int = 1,
                       arms: list[str] | None = None, seeds: list[int] | None = None, config_overrides: dict | None = None,
                       label: str | None = None, max_items: int | None = None, log=print) -> dict:
    items, manifest = enumerate_stage(cfg, stage, arms, seeds, config_overrides, label)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "manifest.json", manifest)
    budget_path = run_dir / "budget.json"
    if not budget_path.exists():
        if cfg["runtime"].get("require_budget_manifest_for_large_runs", True) and len(items) > 100:
            raise RuntimeError(f"stage {stage} has {len(items)} items; run --dry-run first to create {budget_path}")
        wt = measure_wrapper_tokens(cfg, backend.encode_text, backend.chat_prefix_ids)
        write_json(budget_path, budget_manifest(cfg, stage, items, manifest, None, None, wrapper_tokens=wt))
    budget = read_json(budget_path)
    limits = budget["limits"]

    # shard by prompt so that all arms/seeds of one prompt stay together
    prompt_ids = sorted({it.prompt.prompt_id for it in items})
    mine = {pid for i, pid in enumerate(prompt_ids) if i % num_shards == shard}
    items = [it for it in items if it.prompt.prompt_id in mine]
    if max_items:
        items = items[:max_items]
    # order: prompt-major so partial runs cover all arms on a prefix of prompts
    order = {pid: i for i, pid in enumerate(prompt_ids)}
    arm_order = {a: i for i, a in enumerate(manifest["arms"])}
    items.sort(key=lambda it: (order[it.prompt.prompt_id], arm_order[it.arm], it.seed))

    suffix = f".shard{shard}" if num_shards > 1 else ""
    gen_w = JsonlWriter(run_dir / f"generations{suffix}.jsonl")
    traj_w = JsonlWriter(run_dir / f"trajectories{suffix}.jsonl")
    ref_w = JsonlWriter(run_dir / f"references{suffix}.jsonl")
    scorer = LikelihoodScorer(backend, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
    scales = load_scales(cfg, stage) if any(it.gen_cfg.objective in ("global", "local", "specific") for it in items) else None
    reference = ref_map = None
    ref_names = {it.gen_cfg.poe_reference for it in items if it.gen_cfg.poe_reference}
    if ref_names:
        # cross-model steering: a second checkpoint is the keyed/neutral world (labelled extension)
        from .reference import build_reference
        reference = build_reference(cfg)
        mp = cfg.path_of("assets") / f"token_map_{Path(reference.cfg.model_id).name}.npz"
        ref_map = reference.token_map(backend, cache_path=mp, log=log)
        log(f"[generate] cross-model world: {reference.info['model_id']} "
            f"(vocab {reference.info['vocab_size']}, {int((ref_map >= 0).sum())}/{len(ref_map)} generator tokens mapped)")
    gen = CandidateSelectionGenerator(backend, scorer, scales, reference=reference, ref_token_map=ref_map)
    totals = CostRecord()
    t_start = time.time()
    done = skipped = failed = 0
    status = "complete"
    try:
        for it in items:
            if it.output_id in gen_w:
                skipped += 1
                continue
            # reference records for eligible human/web texts (once per prompt/stage)
            rid = f"{stage}.ref.{it.prompt.group_id}"
            if it.prompt.reference_eligible and rid not in ref_w:
                ref_w.write(ReferenceRecord(id=rid, stage=stage, group_id=it.prompt.group_id, prompt_id=it.prompt.prompt_id,
                                            domain=it.prompt.domain, text=it.prompt.reference, text_hash=it.prompt.reference_hash,
                                            visible_tokens=it.prompt.reference_tokens, source=it.prompt.source_split))
            if hasattr(backend, "reset_peak_memory"):
                backend.reset_peak_memory()
            key = keyspec_with_decoys(cfg, it.key) if it.key else None
            res = gen.generate(it.prompt.prompt, key, it.gen_cfg, it.seed, it.output_id)
            rec = GenerationRecord(
                id=it.output_id, stage=stage, arm=it.arm, group_id=it.prompt.group_id, prompt_id=it.prompt.prompt_id,
                domain=it.prompt.domain, key_id=key.key_id if key else None, seed=it.seed, text=res.text,
                text_hash=res.text_hash, accepted_tokens=len(res.accepted_tokens), visible_tokens=res.visible_tokens,
                stop_reason=res.stop_reason, ends_mid_sentence=res.ends_mid_sentence, config_hash=it.config_hash,
                status=res.status, error=res.error, cost=asdict(res.cost), n_decisions=res.n_decisions,
                n_active_decisions=res.n_active_decisions, n_inactive_no_signal=res.n_inactive_no_signal, shard=shard,
                assigned_key_id=it.assigned_key_id, final_score_trace=res.final_score, arm_config=it.gen_cfg.to_dict(),
            )
            for tr in res.trajectory:
                traj_w.write(tr)
            gen_w.write(rec)
            totals.add(res.cost)
            done += 1
            failed += int(res.status != "ok")
            if done % 5 == 0 or done == 1:
                log(f"[{stage} shard {shard}] {done} done, {skipped} skipped, {failed} failed; "
                    f"last {it.arm} {res.stop_reason} {len(res.accepted_tokens)} tok in {res.cost.wall_seconds:.1f}s")
            if totals.proposal_tokens > limits["max_proposal_tokens"] or totals.scoring_tokens > limits["max_scoring_tokens"] \
                    or (time.time() - t_start) > limits["max_wall_seconds"]:
                status = "budget_exceeded"
                log(f"budget limit reached after {done} outputs; checkpointed and stopping")
                break
    finally:
        gen_w.close(); traj_w.close(); ref_w.close()
    steer_cache = getattr(gen, "_steer_cache", None)
    if steer_cache:
        write_json(run_dir / f"steering_vectors{suffix}.json",
                   {k: {"norms": v.get("norms"), "residual_norms": v.get("residual_norms"), "n_layers": v.get("n_layers")}
                    for k, v in steer_cache.items()})
    summary = {"stage": stage, "shard": shard, "num_shards": num_shards, "done": done, "skipped_existing": skipped,
               "failed": failed, "status": status, "totals": asdict(totals), "wall_seconds": time.time() - t_start,
               "backend": asdict(backend.info) if hasattr(backend.info, "__dataclass_fields__") else str(backend.info),
               "remaining_items": len(items) - done - skipped}
    write_json(run_dir / f"summary{suffix}.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# fit-scorer stage (spec Sections 5 and 8, stage 2)
# --------------------------------------------------------------------------- #

def run_fit_scorer(cfg: Config, partition: str, backend: Backend, run_dir: Path, out_path: Path, toy: bool,
                   pool_lengths: list[int], unit_settings: list[dict], objectives: list[str], keys: list[KeyRecord],
                   n_candidates: int = 8, min_pools: int | None = None, repeat_subset: int = 12, log=print) -> dict:
    """Generate ordinary continuations, fork unwatermarked candidate pools, fit sigma per bin, measure numerics."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prepared(cfg.path_of("data_prepared"), partition=partition, domains=cfg["data"]["domains"])
    rows.sort(key=lambda r: (r.domain, r.prompt_id))
    if not rows:
        raise RuntimeError(f"no prompts in partition {partition}")
    scorer = LikelihoodScorer(backend, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
    base_gen = cfg.generation_config(None)
    gcfg = arm_config("base", base_gen)
    gen = CandidateSelectionGenerator(backend, None, None)
    # 1. ordinary continuations (resumable)
    base_w = JsonlWriter(run_dir / "base_generations.jsonl")
    base_records: dict[str, dict] = {r["id"]: r for r in load_jsonl(run_dir / "base_generations.jsonl")}
    for p in rows:
        oid = output_id("fit_scorer", "base", p.group_id, None, 0, content_hash(gcfg.to_dict(), length=8))
        if oid in base_w:
            continue
        res = gen.generate(p.prompt, None, gcfg, 0, oid)
        rec = {"id": oid, "prompt_id": p.prompt_id, "group_id": p.group_id, "domain": p.domain, "text": res.text,
               "accepted_tokens": res.accepted_tokens, "stop_reason": res.stop_reason, "status": res.status,
               "cost": asdict(res.cost)}
        base_w.write(rec)
        base_records[oid] = rec
        log(f"[fit-scorer] base {len(base_records)}/{len(rows)} ({p.domain}, {len(res.accepted_tokens)} tok)")
    base_w.close()

    # 2. candidate pools (resumable) : one record per (prompt, length, unit setting)
    pools_w = JsonlWriter(run_dir / "pools.jsonl")
    pools: dict[str, dict] = {r["id"]: r for r in load_jsonl(run_dir / "pools.jsonl")}
    gsys = gcfg.generator_system
    for p in rows:
        oid = output_id("fit_scorer", "base", p.group_id, None, 0, content_hash(gcfg.to_dict(), length=8))
        rec = base_records.get(oid)
        if not rec or rec["status"] != "ok":
            continue
        acc = rec["accepted_tokens"]
        ctx = backend.chat_prefix_ids(gsys, p.prompt)
        for L in pool_lengths:
            if len(acc) < L:
                continue
            for us in unit_settings:
                unit_max = us["unit_tokens"] if us["unit"] == "tokens" else us["sentence_cap_tokens"]
                uname = unit_setting_name(us["unit"], us.get("unit_tokens", 16), us.get("sentence_cap_tokens", 64))
                pid = f"pool.{p.group_id}.L{L}.{uname}"
                if pid in pools_w:
                    continue
                prefix = acc[:L]
                cands = backend.sample_units(ctx + prefix, n_candidates, unit_max, gcfg.sampling(),
                                             seed_from("pool", cfg.seed, pid), eos_allowed_from=max(0, gcfg.min_new_tokens - L))
                if us["unit"] == "sentence":
                    from .generation import sentence_truncate
                    cands = [type(c)(token_ids=sentence_truncate(backend, c.token_ids)[0], ended=c.ended, n_generated=c.n_generated)
                             for c in cands]
                prec = {"id": pid, "group_id": p.group_id, "domain": p.domain, "length": L, "unit_setting": uname,
                        "prefix_text": backend.decode(prefix), "candidate_texts": [backend.decode(c.token_ids) for c in cands],
                        "cumulative_texts": [backend.decode(prefix + c.token_ids) for c in cands]}
                pools_w.write(prec)
                pools[pid] = prec
        log(f"[fit-scorer] pools for {p.prompt_id}: {sum(1 for k in pools if p.group_id in k)}")
    pools_w.close()

    # 3. score pools under every key (global and local objectives) -> within-pool SDs
    pool_sds: dict[str, list[tuple[str, float]]] = {}
    scores_log = JsonlWriter(run_dir / "pool_scores.jsonl")
    existing_scores = {r["id"]: r for r in load_jsonl(run_dir / "pool_scores.jsonl")}
    fam_of = {k.key_id: k.family for k in keys}
    prefix_sensitivity: dict[str, list[float]] = {}
    for pid, prec in pools.items():
        for k in keys:
            sid = f"{pid}.{k.key_id}"
            if sid in existing_scores:
                r = existing_scores[sid]
            else:
                ks = k.to_keyspec()
                pre = scorer.score(prec["prefix_text"], ks)
                cum = scorer.score_texts(prec["cumulative_texts"], ks)
                loc = scorer.score_texts(prec["candidate_texts"], ks)
                r = {"id": sid, "pool_id": pid, "key_id": k.key_id, "family": k.family, "length": prec["length"],
                     "unit_setting": prec["unit_setting"], "group_id": prec["group_id"],
                     "prefix_score": pre.score, "prefix_status": pre.status,
                     "global_gains": [(c.score - pre.score) if (c.status == "ok" and pre.status in ("ok", "empty")) else None for c in cum],
                     "local_gains": [(c.score if c.status in ("ok", "empty") else None) for c in loc],
                     "cumulative_scores": [c.score for c in cum], "cumulative_contrast_sums": [c.contrast_sum for c in cum]}
                scores_log.write(r)
                existing_scores[sid] = r
            b = length_bin(r["length"])
            for objective in objectives:
                gains = r["global_gains"] if objective == "global" else r["local_gains"]
                sd = within_pool_sd([g for g in gains if g is not None])
                if math.isfinite(sd):
                    pool_sds.setdefault(entry_key(fam_of[k.key_id], r["unit_setting"], objective, b), []).append((r["group_id"], sd))
            if r["prefix_score"] is not None:
                prefix_sensitivity.setdefault(k.key_id, []).append(r["prefix_score"])
    scores_log.close()

    # 4. numerical repeatability: rescore a fixed subset twice (batched and singly)
    disc = 0.0
    subset = sorted(pools)[:repeat_subset]
    for pid in subset:
        prec = pools[pid]
        k = keys[0].to_keyspec()
        a = scorer.score_texts(prec["cumulative_texts"], k)
        b = [scorer.score(t, k) for t in prec["cumulative_texts"]]
        c = scorer.score_texts(prec["cumulative_texts"], k)
        for x, y, z in zip(a, b, c):
            if x.score is not None and y.score is not None and z.score is not None:
                disc = max(disc, abs(x.score - y.score), abs(x.score - z.score))
    table = fit_scale_table(pool_sds, scorer.scorer_id, partition, disc, toy,
                            min_pools=(min_pools if min_pools is not None else (1 if toy else 30)),
                            meta={"pool_lengths": pool_lengths, "unit_settings": unit_settings, "objectives": objectives,
                                  "n_candidates": n_candidates, "n_keys": len(keys), "key_set": keys[0].key_set if keys else None,
                                  "n_prompts": len(rows), "backend": asdict(backend.info)})
    write_json(out_path, table.to_dict())
    # diagnostics: prefix sensitivity per key, wrong-key correlation on prefix scores
    diag = {"numerical_discrepancy": disc, "floor": table.floor, "entries": table.entries,
            "prefix_score_by_key": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
                                    for k, v in prefix_sensitivity.items()},
            "pool_sd_counts": {k: len(v) for k, v in pool_sds.items()}}
    # correlation between keys of the same pools
    kid = sorted(prefix_sensitivity)
    mat = {}
    pool_ids = sorted(pools)
    vecs = {}
    for k in kid:
        vecs[k] = [existing_scores.get(f"{pid}.{k}", {}).get("prefix_score") for pid in pool_ids]
    for i, a in enumerate(kid):
        for b2 in kid[i + 1:]:
            xs = [(x, y) for x, y in zip(vecs[a], vecs[b2]) if x is not None and y is not None]
            if len(xs) > 3:
                x, y = zip(*xs)
                mat[f"{a}|{b2}"] = float(np.corrcoef(x, y)[0, 1])
    diag["wrong_key_prefix_score_correlation"] = mat
    write_json(run_dir / "diagnostics.json", diag)
    return {"scales": str(out_path), "entries": len(table.entries), "numerical_discrepancy": disc, "floor": table.floor,
            "n_pools": len(pools)}


# --------------------------------------------------------------------------- #
# 'specific' objective scales (amendment): reuse the unwatermarked pools, add decoy scorings
# --------------------------------------------------------------------------- #

def run_fit_specific(cfg: Config, partition: str, backend: Backend, run_dir: Path, scales_path: Path,
                     keys: list[KeyRecord], log=print) -> dict:
    """Add ``<family>|<unit>|specific|<bin>`` entries to an existing scale table.

    S_k(text) = (ell_k - mean_d ell_d) / n.  ell_k is recovered from the stored keyed contrast sums plus the
    neutral sum (ell_k = contrast_k + ell_0); ell_0 and the decoy sums ell_d are scored here (neutral once per
    text, decoys once per family per text), so the new cost is ~21 scorings per pool text instead of 100.
    """
    from .scoring import SCORING_USER
    pools = {r["id"]: r for r in load_jsonl(run_dir / "pools.jsonl")}
    pscores = {r["id"]: r for r in load_jsonl(run_dir / "pool_scores.jsonl")}
    if not pools or not pscores:
        raise RuntimeError(f"{run_dir}: run the primary fit-scorer first (pools.jsonl / pool_scores.jsonl missing)")
    table = ScaleTable.from_dict(read_json(scales_path))
    dec_by_family = steering_decoys(cfg)
    if not dec_by_family:
        raise RuntimeError("no decoy bank: run make-keys --set decoy first")
    scorer = LikelihoodScorer(backend, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
    st = cfg["scoring"]
    neutral_ctx = backend.chat_prefix_ids(st["system_template"].format(prefix=cfg["keys"]["neutral_prefix"]), st.get("user_message", SCORING_USER))
    decoy_ctx = {fam: [backend.chat_prefix_ids(st["system_template"].format(prefix=d["prefix"]), st.get("user_message", SCORING_USER))
                       for d in decs] for fam, decs in dec_by_family.items()}
    fam_of = {k.key_id: k.family for k in keys}
    out_w = JsonlWriter(run_dir / "pool_specific_scores.jsonl")
    existing = {r["id"]: r for r in load_jsonl(run_dir / "pool_specific_scores.jsonl")}
    pool_sds: dict[str, list[tuple[str, float]]] = {}
    t0 = time.time()
    for n_done, (pid, prec) in enumerate(pools.items(), 1):
        texts = [prec["prefix_text"]] + prec["cumulative_texts"] + prec["candidate_texts"]
        toks = [scorer.tokenize(t) for t in texts]
        ns = [len(t) for t in toks]
        rec = existing.get(pid)
        if rec is None:
            valid = [i for i, t in enumerate(toks) if t]
            ell0 = [None] * len(texts)
            for i, v in zip(valid, backend.logprob_sums(neutral_ctx, [toks[i] for i in valid])):
                ell0[i] = float(v)
            ell_d = {}
            for fam, ctxs in decoy_ctx.items():
                sums = [backend.logprob_sums(c, [toks[i] for i in valid]) for c in ctxs]
                m = [None] * len(texts)
                for j, i in enumerate(valid):
                    m[i] = float(sum(s[j] for s in sums) / len(sums))
                ell_d[fam] = m
            rec = {"id": pid, "n_tokens": ns, "ell_neutral": ell0, "ell_decoy_mean": ell_d}
            out_w.write(rec)
            existing[pid] = rec
        b = length_bin(prec["length"])
        for k in keys:
            ps = pscores.get(f"{pid}.{k.key_id}")
            if not ps or ps["prefix_score"] is None:
                continue
            fam = fam_of[k.key_id]
            if fam not in rec["ell_decoy_mean"]:
                continue
            # keyed sums: prefix = prefix_score * n; cumulative = stored contrast sums; (local texts unused for 'specific')
            contrast = [ps["prefix_score"] * rec["n_tokens"][0]] + list(ps["cumulative_contrast_sums"])
            S = []
            for i, c in enumerate(contrast):
                e0, ed, n = rec["ell_neutral"][i], rec["ell_decoy_mean"][fam][i], rec["n_tokens"][i]
                S.append(None if (c is None or e0 is None or ed is None or not n) else ((c + e0) - ed) / n)
            if S[0] is None:
                continue
            gains = [s - S[0] for s in S[1:] if s is not None]
            sd = within_pool_sd(gains)
            if math.isfinite(sd):
                pool_sds.setdefault(entry_key(fam, prec["unit_setting"], "specific", b), []).append((prec["group_id"], sd))
        if n_done % 25 == 0:
            log(f"[fit-specific] {n_done}/{len(pools)} pools ({time.time() - t0:.0f}s)")
    out_w.close()
    new = fit_scale_table(pool_sds, table.scorer_id, partition, table.numerical_discrepancy, table.toy,
                          min_pools=(1 if table.toy else 30), meta={"objective": "specific", "steering_decoys": dec_by_family})
    for k, v in new.entries.items():
        table.entries[k] = v
    table.meta.setdefault("specific", {})["steering_decoys"] = dec_by_family
    write_json(scales_path, table.to_dict())
    added = {k: v for k, v in new.entries.items()}
    log(f"[fit-specific] added {len(added)} entries to {scales_path}")
    return {"entries_added": len(added), "sample": {k: v for k, v in list(added.items())[:4]}}


# --------------------------------------------------------------------------- #
# scoring a run: detection records without labels (join happens in evaluation)
# --------------------------------------------------------------------------- #

def collect_run_texts(run_dir: Path) -> tuple[list[dict], list[dict]]:
    gens = []
    refs = []
    for p in sorted(run_dir.glob("generations*.jsonl")):
        gens.extend(load_jsonl(p))
    for p in sorted(run_dir.glob("references*.jsonl")):
        refs.extend(load_jsonl(p))
    return gens, refs


def score_run(run_dir: Path, backend: Backend, keys: list[KeyRecord], max_input_tokens: int, min_visible_tokens: int,
              length_views: list[int], all_keys: bool = True, log=print) -> Path:
    """Score every text of a run under its assigned key (and all keys if requested); writes detections_raw.jsonl.

    The records contain only text hash, key id, score and eligibility fields.
    """
    gens, refs = collect_run_texts(run_dir)
    scorer = LikelihoodScorer(backend, max_input_tokens=max_input_tokens)
    out = JsonlWriter(run_dir / "detections_raw.jsonl")
    texts = [(g["text_hash"], g["text"], g.get("assigned_key_id")) for g in gens] + [(r["text_hash"], r["text"], None) for r in refs]
    # de-duplicate identical texts
    uniq: dict[str, str] = {}
    for th, t, _ in texts:
        uniq.setdefault(th, t)
    assigned: dict[str, set[str]] = {}
    for th, _, ak in texts:
        if ak:
            assigned.setdefault(th, set()).add(ak)
    key_specs = {k.key_id: k.to_keyspec() for k in keys}
    hashes = sorted(uniq)
    n_done = 0
    for kid, ks in key_specs.items():
        todo = [th for th in hashes if (all_keys or kid in assigned.get(th, ())) and f"{th}.{kid}" not in out]
        for s in range(0, len(todo), 64):
            chunk = todo[s:s + 64]
            res = scorer.score_texts([uniq[th] for th in chunk], ks)
            for th, r in zip(chunk, res):
                n_vis = r.n_tokens
                rec = {"id": f"{th}.{kid}", "text_hash": th, "key_id": kid, "scorer_id": scorer.scorer_id, "score": r.score,
                       "status": r.status, "visible_tokens": n_vis, "eligible": (r.status == "ok" and n_vis >= min_visible_tokens),
                       "contrast_sum": r.contrast_sum, "error": r.error}
                if kid in assigned.get(th, ()):
                    views = scorer.score_prefix_views(uniq[th], ks, [L for L in length_views if L < n_vis])
                    rec["prefix_views"] = {str(L): {"score": v.score, "status": v.status, "n_tokens": v.n_tokens} for L, v in views.items()}
                    # within-text t-statistic (amendment): mean per-token contrast / (sd / sqrt(n)); frac of tokens > 0
                    d = scorer.per_token_contrasts([uniq[th]], ks)[0]
                    if d and len(d) >= 2:
                        arr = np.asarray(d, dtype=np.float64)
                        sd = float(arr.std(ddof=1))
                        rec["t_stat"] = float(arr.mean() / (sd / math.sqrt(arr.size))) if sd > 0 else None
                        rec["frac_pos"] = float((arr > 0).mean())
                        rec["t_n"] = int(arr.size)
                out.write(rec)
                n_done += 1
        log(f"[score] key {kid}: {len(todo)} texts scored (total {n_done})")
    out.close()
    # back-fill: within-text t-statistics for assigned keys whose records predate the t-detector (side file merged at load)
    have = {r["id"] for r in load_jsonl(run_dir / "detections_raw.jsonl") if r.get("t_stat") is not None or r.get("t_n") is not None}
    tw = JsonlWriter(run_dir / "tstats.jsonl")
    n_bf = 0
    for th, ks in assigned.items():
        for kid in ks:
            rid = f"{th}.{kid}"
            if rid in have or rid in tw or kid not in key_specs:
                continue
            d = scorer.per_token_contrasts([uniq[th]], key_specs[kid])[0]
            if d and len(d) >= 2:
                arr = np.asarray(d, dtype=np.float64)
                sd = float(arr.std(ddof=1))
                tw.write({"id": rid, "text_hash": th, "key_id": kid, "t_stat": float(arr.mean() / (sd / math.sqrt(arr.size))) if sd > 0 else None,
                          "frac_pos": float((arr > 0).mean()), "t_n": int(arr.size)})
                n_bf += 1
    tw.close()
    if n_bf:
        log(f"[score] back-filled t-statistics for {n_bf} (text, assigned key) pairs")
    return run_dir / "detections_raw.jsonl"
