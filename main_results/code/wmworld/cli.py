"""Command-line interface (spec Section 12).  Each command processes one stage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import Config, config_from_protocol, load_config
from .records import content_hash, read_json, write_json


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cfg_from_args(args) -> Config:
    if getattr(args, "protocol", None):
        proto = read_json(args.protocol)
        cfg = config_from_protocol(proto)
        cfg.raw["_protocol"] = {"path": str(Path(args.protocol).resolve()), "hash": proto.get("content_hash")}
        return cfg
    if not getattr(args, "config", None):
        raise SystemExit("either --config or --protocol is required")
    return load_config(args.config)


def _tokenizer_funcs(cfg: Config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["models"]["shared"]["id"], revision=cfg.model_revision)

    def encode(t):
        return [int(x) for x in tok(t, add_special_tokens=False)["input_ids"]]

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    return tok, encode, decode


# --------------------------------------------------------------------------- #
# prepare-data
# --------------------------------------------------------------------------- #

def cmd_prepare_data(args) -> None:
    from .data.prepare import load_domain_examples, prepare_domain, write_prepared
    cfg = _cfg_from_args(args)
    cfg.validate(require_revision=True)
    tok, encode, decode = _tokenizer_funcs(cfg)
    raw_root = cfg.path_of("data_raw")
    out_dir = cfg.path_of("data_prepared")
    budgets = dict(cfg["data"]["groups_per_domain"])
    prepared_all, reports = [], []
    domains = args.domains.split(",") if getattr(args, "domains", None) else list(cfg["data"]["domains"])
    extra_budgets = json.loads(args.budgets) if getattr(args, "budgets", None) else {}
    if getattr(args, "append", False):
        # keep every already-prepared domain untouched; add/replace only the requested ones
        from .data.prepare import load_prepared
        prepared_all = [p for p in load_prepared(out_dir) if p.domain not in domains]
        old_manifest = read_json(out_dir / "manifest.json") if (out_dir / "manifest.json").exists() else {}
        reports = [r for r in old_manifest.get("reports", []) if r.get("domain") not in domains]
    for domain in domains:
        _log(f"loading {domain}")
        exs = load_domain_examples(domain, raw_root, encode, decode, int(cfg["data"].get("writingprompts_candidate_cap", 6000)), cfg.seed)
        _log(f"{domain}: {len(exs)} candidates loaded; filtering, dedup, partitioning")
        dom_budgets = extra_budgets.get(domain, (cfg["data"].get("domain_budgets") or {}).get(domain, budgets))
        prepared, report = prepare_domain(domain, exs, dom_budgets, encode, decode, cfg.seed,
                                          reference_cap_tokens=int(cfg["data"].get("reference_cap_tokens", 512)),
                                          min_reference_tokens=int(cfg["detection"]["min_visible_tokens"]),
                                          allow_shortfall=bool(cfg["data"].get("allow_shortfall", False)))
        _log(f"{domain}: {json.dumps({k: v for k, v in report.items() if k != 'domain'})}")
        prepared_all.extend(prepared)
        reports.append(report)
    manifest = write_prepared(out_dir, prepared_all, reports, {"seed": cfg.seed, "groups_per_domain": budgets,
                                                                "config_hash": cfg.content_hash()},
                              {"tokenizer": cfg["models"]["shared"]["id"], "revision": cfg.model_revision})
    _log(f"wrote {out_dir}/manifest.json (hash {manifest['content_hash']}); {len(prepared_all)} rows")


# --------------------------------------------------------------------------- #
# make-keys
# --------------------------------------------------------------------------- #

def cmd_make_keys(args) -> None:
    from .key_catalog import SEED_WORDS_RARE
    from .keys import FAMILIES, build_keys, guard_keys, keys_manifest, score_catalog_rarity
    from .scoring import LikelihoodScorer
    from .stages import build_backend, keys_path
    cfg = _cfg_from_args(args)
    cfg.validate(require_revision=True)
    key_set = args.set
    out = keys_path(cfg, key_set)
    if out.exists() and not args.force:
        _log(f"{out} exists; use --force to regenerate (keys must not be re-drawn casually)")
        return
    backend = build_backend(cfg, args.device)
    exclude = set()
    families = list(FAMILIES)
    vocab = None
    if key_set == "development":
        per_family = int(cfg["keys"]["development_per_family"])
        seed = cfg.seed
    elif key_set in ("randomwords", "randomwords_rare"):
        # seed-phrase keys (random unrelated words), an extra family outside the spec's five; own key set.
        # randomwords_rare draws from the low-incidence vocabulary (SEED_WORDS_RARE) so chance occurrences are negligible
        per_family = int(cfg["keys"].get("randomwords_per_family", 12))
        seed = cfg.seed + (3 if key_set == "randomwords" else 4)
        families = ["random_words"]
        vocab = SEED_WORDS_RARE if key_set == "randomwords_rare" else None
    elif key_set in ("decoy_rw", "decoy_rwr"):
        # same-family decoy bank for seed-phrase targets: 100 PRNG phrases from the SAME vocabulary and strata as the
        # targets (exchangeable with a wrong key), excluding the exact target phrases.  The catalog bank (persona/digit
        # decoys) is family-mismatched for random-word keys and inflates "beats all decoys".
        per_family = int(cfg["keys"].get("decoys_rw", 100))
        seed = cfg.seed + (6 if key_set == "decoy_rw" else 7)
        families = ["random_words"]
        vocab = SEED_WORDS_RARE if key_set == "decoy_rwr" else None
        tgt = keys_path(cfg, "randomwords" if key_set == "decoy_rw" else "randomwords_rare")
        if tgt.exists():
            exclude |= {k["prefix"] for k in read_json(tgt)["keys"]}
    elif key_set == "decoy":
        # decoy bank for the SECONDARY rank detector: never used as target keys; excludes every target prefix
        per_family = int(cfg["keys"].get("decoys_per_family", 20))
        seed = cfg.seed + 2
        for other in ("development", "confirmation"):
            p = keys_path(cfg, other)
            if p.exists():
                exclude |= {k["prefix"] for k in read_json(p)["keys"]}
    else:
        per_family = int(cfg["keys"]["confirmation_per_selected_family"])
        seed = cfg.seed + 1
        proto = cfg.raw.get("_protocol")
        if proto:
            p = read_json(proto["path"])
            families = list(p.get("selected_families") or families)
        dev = keys_path(cfg, "development")
        if dev.exists():
            exclude = {k["prefix"] for k in read_json(dev)["keys"]}
    if any(f not in ("random_digits", "random_words") for f in families):
        _log("scoring catalog rarity with the reference model")
        rarity = score_catalog_rarity(backend)
    else:
        rarity = []          # PRNG families do not use the catalog
    keys = build_keys(backend, key_set, per_family, seed, rarity, cfg.model_revision, exclude_prefixes=exclude, families=families, vocab=vocab)
    scorer = LikelihoodScorer(backend, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
    guard_keys(backend, keys, scorer)
    m = keys_manifest(keys, key_set, seed, asdict(backend.info), rarity,
                      extra={"families": families, "per_family": per_family, "excluded_prefixes": len(exclude)})
    write_json(out, m)
    _log(f"wrote {out}: {len(keys)} keys; excluded by guard: {sum(1 for k in keys if k.guard_status != 'ok')}")
    for k in keys:
        _log(f"  {k.key_id} {k.family:14s} tok={k.token_length:3d} {k.guard_status} :: {k.prefix[:70]}")


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def cmd_generate(args) -> None:
    from .stages import build_backend, budget_manifest, enumerate_stage, run_generate_stage
    cfg = _cfg_from_args(args)
    cfg.validate(require_revision=True)
    stage = args.stage
    run_dir = cfg.path_of("runs") / (args.run_name or stage)
    overrides = json.loads(args.override) if args.override else None
    arms = args.arms.split(",") if args.arms else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if args.dry_run:
        from .stages import measure_wrapper_tokens
        items, manifest = enumerate_stage(cfg, stage, arms, seeds, overrides, args.run_name)
        smoke_bench = None
        bench_path = cfg.path_of("runs") / "smoke" / "benchmark.json"
        if bench_path.exists():
            smoke_bench = read_json(bench_path)
        tok, encode, decode = _tokenizer_funcs(cfg)

        def chat_prefix_ids(system, user):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

        wt = measure_wrapper_tokens(cfg, encode, chat_prefix_ids)
        b = budget_manifest(cfg, stage, items, manifest, args.wall_hours, smoke_bench, wrapper_tokens=wt)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "budget.json", b)
        write_json(run_dir / "manifest.json", manifest)
        e = b["estimates"]["total"]
        _log(f"DRY RUN {stage}: prompts={manifest['n_prompts']} arms={manifest['arms']} seeds={manifest['seeds']} "
             f"keys={manifest['n_keys']} outputs={e['outputs']}")
        _log(f"  max proposal tokens={e['max_proposal_tokens']:,} max scoring tokens={e['max_scoring_tokens']:,} "
             f"max scoring sequences={e['max_scoring_sequences']:,}")
        for arm, a in b["estimates"]["per_arm"].items():
            _log(f"  {arm:16s} outputs={a['outputs']:5d} proposal<= {a['max_proposal_tokens']:>12,} scoring<= {a['max_scoring_tokens']:>14,}")
        if b["estimated_wall_seconds_1gpu"]:
            _log(f"  estimated wall (1 GPU, from smoke benchmark): {b['estimated_wall_seconds_1gpu'] / 3600:.2f} h")
        _log(f"  limits: {b['limits']}  -> wrote {run_dir}/budget.json")
        return
    backend = build_backend(cfg, args.device)
    summary = run_generate_stage(cfg, stage, backend, run_dir, shard=args.shard, num_shards=args.num_shards, arms=arms,
                                 seeds=seeds, config_overrides=overrides, label=args.run_name, max_items=args.max_items, log=_log)
    _log(json.dumps({k: v for k, v in summary.items() if k != "backend"}, default=str))


# --------------------------------------------------------------------------- #
# fit-scorer
# --------------------------------------------------------------------------- #

def cmd_fit_scorer(args) -> None:
    from .stages import build_backend, load_key_records, run_fit_scorer, run_fit_specific
    cfg = _cfg_from_args(args)
    cfg.validate(require_revision=True)
    partition = args.partition
    toy = partition != "scale_fit" or args.toy
    out = cfg.resolve_path(args.output) if args.output else (cfg.path_of("assets") / ("scales_smoke.json" if toy else "scales.json"))
    if args.objective == "specific":
        keys = load_key_records(cfg, args.keys)
        backend = build_backend(cfg, args.device)
        res = run_fit_specific(cfg, partition, backend, cfg.path_of("runs") / f"fit_scorer_{partition}", out, keys, log=_log)
        _log(json.dumps(res))
        return
    fs = cfg.raw.get("fit_scorer", {}) or {}
    pool_lengths = fs.get("pool_lengths_toy" if toy else "pool_lengths", [16, 32] if toy else [32, 96, 192, 384])
    unit_settings = fs.get("unit_settings", [{"unit": "tokens", "unit_tokens": 16, "sentence_cap_tokens": 64}])
    objectives = fs.get("objectives", ["global", "local"])
    keys = load_key_records(cfg, args.keys)
    backend = build_backend(cfg, args.device)
    res = run_fit_scorer(cfg, partition, backend, cfg.path_of("runs") / f"fit_scorer_{partition}", out, toy, pool_lengths,
                         unit_settings, objectives, keys, n_candidates=int(fs.get("candidates", 8)),
                         min_pools=(1 if toy else None), repeat_subset=int(fs.get("repeat_subset", 12)), log=_log)
    _log(json.dumps(res))


# --------------------------------------------------------------------------- #
# evaluate / freeze / calibrate / detect / attacks / topic-null / report
# --------------------------------------------------------------------------- #

def cmd_evaluate(args) -> None:
    from .evaluation import evaluate_run
    cfg = _cfg_from_args(args) if (args.config or args.protocol) else None
    run_dir = Path(args.run) if args.run else None
    evaluate_run(cfg, run_dir, device=args.device, score=not args.no_score, log=_log, protocol_path=args.protocol,
                 use_decoys=not args.no_decoys)


def cmd_freeze(args) -> None:
    from .protocol import freeze_protocol
    cfg = _cfg_from_args(args)
    out = freeze_protocol(cfg, Path(args.run), Path(args.output), selected=json.loads(args.select) if args.select else None,
                          rationale=args.rationale, log=_log)
    _log(f"wrote {out}")


def cmd_calibrate(args) -> None:
    from .evaluation import calibrate_from_null
    cfg = _cfg_from_args(args)
    calibrate_from_null(cfg, device=args.device, log=_log)


def cmd_detect(args) -> None:
    """Standalone detector: text + key id + calibration manifest only."""
    from .calibration import Calibration, detect
    from .runtime import QwenRuntime, RuntimeConfig
    from .scoring import LikelihoodScorer
    from .textnorm import normalize_text
    cal = Calibration.from_dict(read_json(args.calibration))
    key = cal.keyspec(args.key_id)
    text = normalize_text(Path(args.text_file).read_text(encoding="utf-8"))
    bi = cal.backend_info or {}
    rt = QwenRuntime(RuntimeConfig(model_id=bi.get("model_id", "Qwen/Qwen3.5-0.8B"), revision=bi.get("revision"),
                                   device=args.device or "auto", max_input_tokens=cal.max_input_tokens,
                                   kernels=bi.get("extra", {}).get("kernels_policy", "auto")))
    scorer = LikelihoodScorer(rt, max_input_tokens=cal.max_input_tokens)
    res = detect(text, key, cal, scorer)
    out = {"decision": res.decision, "raw_score": res.score, "p_generator": res.p_generator, "p_human": res.p_human,
           "p_final": res.p_final, "visible_tokens": res.visible_tokens, "eligibility": res.status, "key_id": res.key_id,
           "scorer_id": res.scorer_id, "calibration_id": res.calibration_id, "components": res.components,
           "textnorm_version": cal.textnorm_version}
    if args.mode in ("rank", "both"):
        from .calibration import detect_secondary
        if res.decision in ("detected", "not_detected"):
            decoys = [d for d in cal.secondary.get("decoys", []) if d != args.key_id]
            sc = {args.key_id: res.score}
            for d in decoys:
                sc[d] = scorer.score(text, cal.keyspec(d)).score
            out["secondary_rank_detector"] = detect_secondary(sc, args.key_id, cal)
        else:
            out["secondary_rank_detector"] = {"status": res.decision}
    print(json.dumps(out, indent=2))


def cmd_topic_null(args) -> None:
    from .topic_null import run_topic_null
    cfg = _cfg_from_args(args)
    run_topic_null(cfg, device=args.device, log=_log)


def cmd_attack(args) -> None:
    from .attacks import run_attacks
    cfg = _cfg_from_args(args)
    run_attacks(cfg, run_name=args.run_name, device=args.device, subset=args.subset, log=_log,
                max_groups_per_domain=int(args.max_groups_per_domain) if args.max_groups_per_domain else None)


def cmd_code_eval(args) -> None:
    from .code_eval import evaluate_code_run
    cfg = _cfg_from_args(args)
    evaluate_code_run(cfg, Path(args.run), timeout=float(args.timeout), log=_log)


def cmd_report(args) -> None:
    from .report import build_report
    cfg = _cfg_from_args(args)
    build_report(cfg, log=_log)


def cmd_ifeval(args) -> None:
    from .ifeval import run_ifeval
    cfg = _cfg_from_args(args)
    run_ifeval(cfg, arms=args.arms.split(",") if args.arms else None, device=args.device, evaluate_only=args.evaluate_only,
               shard=args.shard, num_shards=args.num_shards, log=_log, key_set=args.key_set, limit=args.limit,
               budget=args.budget, run_name=args.run_name)


def cmd_selftest(args) -> None:
    """Model-backed checks on the real runtime (spec Section 14: loading, template, likelihood, state isolation)."""
    from .selftest import run_selftest
    cfg = _cfg_from_args(args)
    cfg.validate(require_revision=True)
    res = run_selftest(cfg, device=args.device, log=_log)
    out = cfg.path_of("runs") / "selftest.json"
    write_json(out, res)
    _log(f"wrote {out}; all_passed={res['all_passed']}")
    if not res["all_passed"]:
        sys.exit(1)


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m wmworld.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, protocol=True):
        sp.add_argument("--config")
        if protocol:
            sp.add_argument("--protocol")
        sp.add_argument("--device", default=None)

    s = sub.add_parser("prepare-data"); common(s); s.set_defaults(fn=cmd_prepare_data)

    s.add_argument("--domains", default=None, help="comma list; default: config domains")

    s.add_argument("--budgets", default=None, help='JSON {domain: {partition: n}} overriding groups_per_domain for those domains')

    s.add_argument("--append", action="store_true", help="keep other prepared domains; add/replace only --domains")
    s = sub.add_parser("make-keys"); common(s); s.add_argument("--set", required=True, choices=["development", "confirmation", "decoy", "randomwords", "randomwords_rare", "decoy_rw", "decoy_rwr"])
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_make_keys)
    s = sub.add_parser("generate"); common(s); s.add_argument("--stage", required=True); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--shard", type=int, default=0); s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--arms"); s.add_argument("--seeds"); s.add_argument("--override", help="JSON generation overrides (ablations)")
    s.add_argument("--run-name"); s.add_argument("--max-items", type=int); s.add_argument("--wall-hours", type=float)
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("fit-scorer"); common(s); s.add_argument("--partition", required=True); s.add_argument("--keys", default="development")
    s.add_argument("--output"); s.add_argument("--toy", action="store_true")
    s.add_argument("--objective", default="primary", choices=["primary", "specific"],
                   help="primary = global+local sigma table; specific = add entries for the decoy-relative objective (amendment)")
    s.set_defaults(fn=cmd_fit_scorer)
    s = sub.add_parser("evaluate"); common(s); s.add_argument("--run"); s.add_argument("--no-score", action="store_true")
    s.add_argument("--no-decoys", action="store_true", help="skip the decoy bank (secondary rank detector then uses other target keys)")
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("freeze"); common(s, protocol=False); s.add_argument("--run", required=True); s.add_argument("--output", required=True)
    s.add_argument("--select", help="JSON list of selected configurations, e.g. '[{\"arm\":\"llm_global\"}]'")
    s.add_argument("--rationale", default=""); s.set_defaults(fn=cmd_freeze)
    s = sub.add_parser("calibrate"); common(s); s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("detect"); s.add_argument("--text-file", required=True); s.add_argument("--key-id", required=True)
    s.add_argument("--calibration", required=True); s.add_argument("--device")
    s.add_argument("--mode", default="primary", choices=["primary", "rank", "both"],
                   help="primary = spec's calibrated score; rank = secondary correct-key-vs-decoys detector (labelled amendment)")
    s.set_defaults(fn=cmd_detect)
    s = sub.add_parser("topic-null"); common(s); s.set_defaults(fn=cmd_topic_null)
    s = sub.add_parser("attack"); common(s); s.add_argument("--run-name", default="positive_test"); s.add_argument("--subset", default="all")
    s.add_argument("--max-groups-per-domain", default=None, help="cap the attacked source groups per domain (hash order)")
    s.set_defaults(fn=cmd_attack)
    s = sub.add_parser("code-eval"); common(s); s.add_argument("--run", required=True); s.add_argument("--timeout", default="10")
    s.set_defaults(fn=cmd_code_eval)
    s = sub.add_parser("report"); common(s); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("ifeval"); common(s); s.add_argument("--arms"); s.add_argument("--evaluate-only", action="store_true")
    s.add_argument("--shard", type=int, default=0); s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--key-set", default=None); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--budget", type=int, default=2048); s.add_argument("--run-name", default="ifeval"); s.set_defaults(fn=cmd_ifeval)
    s = sub.add_parser("selftest"); common(s); s.set_defaults(fn=cmd_selftest)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
