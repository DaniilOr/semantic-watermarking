"""Evaluation (spec Sections 9 and 11): join layer after detection, metrics with
cluster-bootstrap uncertainty, toy calibration for development runs, real
calibration from ``null_calibration``, confound diagnostics, tables and plots.

The detector never sees labels; this module joins detection records with the
generation metadata *after* scoring.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import plots
from .calibration import Calibration, build_calibration, detect_from_score, tail_rank_p
from .config import Config
from .keys import load_keys
from .records import content_hash, load_jsonl, read_json, write_json
from .scoring import ScoreResult
from . import stages as _stages
from .stages import collect_run_texts, keys_path, score_run


def build_backend(cfg, device=None):
    return _stages.build_backend(cfg, device)
from .textnorm import seed_from


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_mean(values: list[float], clusters: list[str], replicates: int, seed: int) -> dict:
    """Mean with a paired cluster bootstrap over source groups."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0, "n_clusters": 0}
    cl = np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, uniq.size, size=(replicates, uniq.size))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": int(v.size), "n_clusters": int(uniq.size)}


def auroc(pos: list[float], neg: list[float]) -> float | None:
    pos = [p for p in pos if p is not None and np.isfinite(p)]
    neg = [n for n in neg if n is not None and np.isfinite(n)]
    if not pos or not neg:
        return None
    from sklearn.metrics import roc_auc_score
    y = [1] * len(pos) + [0] * len(neg)
    return float(roc_auc_score(y, pos + neg))


def bootstrap_auroc(pos: list[tuple[str, float]], neg: list[tuple[str, float]], replicates: int, seed: int) -> dict:
    """AUROC with a cluster bootstrap over source groups (ids in the tuples)."""
    a = auroc([s for _, s in pos], [s for _, s in neg])
    if a is None:
        return {"auroc": None, "ci95": [None, None], "n_pos": len(pos), "n_neg": len(neg)}
    groups = sorted({g for g, _ in pos} | {g for g, _ in neg})
    gi = {g: i for i, g in enumerate(groups)}
    pos_by = defaultdict(list); neg_by = defaultdict(list)
    for g, s in pos:
        pos_by[gi[g]].append(s)
    for g, s in neg:
        neg_by[gi[g]].append(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(min(replicates, 500)):
        pick = rng.integers(0, len(groups), size=len(groups))
        p = [s for i in pick for s in pos_by.get(i, [])]
        n = [s for i in pick for s in neg_by.get(i, [])]
        v = auroc(p, n)
        if v is not None:
            vals.append(v)
    return {"auroc": a, "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None],
            "n_pos": len(pos), "n_neg": len(neg)}


# --------------------------------------------------------------------------- #
# leakage / topic diagnostics
# --------------------------------------------------------------------------- #

STOP = {"you", "your", "the", "every", "day", "is", "favorite", "enjoy", "collect", "play", "eat", "animal", "color"}


def distinctive_terms(key_rec: dict) -> list[str]:
    terms = []
    for c in key_rec.get("completions", []):
        for w in re.findall(r"[a-z]+", c.lower()):
            if w not in STOP and len(w) > 2:
                terms.append(w)
    if key_rec.get("family") == "random_digits":
        terms.append(key_rec["prefix"])
    return sorted(set(terms))


def _category_vocab(key_rec: dict) -> set[str]:
    """Words of every catalog completion in the key's template categories (semantic drift toward the key's subject)."""
    from .key_catalog import COMPLETIONS
    words: set[str] = set()
    for tid in key_rec.get("template_ids", []):
        for c in COMPLETIONS.get(tid, []):
            for w in re.findall(r"[a-z]+", c.lower()):
                if w not in STOP and len(w) > 3:
                    words.add(w)
    return words


def leakage(text: str, key_rec: dict) -> dict:
    low = text.lower()
    exact = key_rec["prefix"].lower() in low
    terms = distinctive_terms(key_rec)
    hits = [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", low)]
    cat = _category_vocab(key_rec)
    text_words = set(re.findall(r"[a-z]+", low))
    cat_hits = sorted(text_words & cat)
    return {"exact_prefix": exact, "term_hits": hits, "n_terms": len(terms), "any": bool(exact or hits),
            "category_hits": cat_hits, "category_any": bool(cat_hits)}


# --------------------------------------------------------------------------- #
# evaluate a run
# --------------------------------------------------------------------------- #

def quality_proxy_run(run_dir: Path, backend, cfg: Config, manifest: dict, gens: list[dict], log=print) -> None:
    """Destructiveness proxy: mean per-token log-likelihood of each output under the task-conditioned generator
    (system prompt + task, no key).  The quality evaluator may see the task (spec Section 11); the detector never does.
    Resumable; written to quality_proxy.jsonl."""
    from .data.prepare import load_prepared
    from .generation import GENERATOR_SYSTEM
    from .records import JsonlWriter
    prompts = {p.prompt_id: p for p in load_prepared(cfg.path_of("data_prepared"), partition=manifest["partition"])}
    w = JsonlWriter(run_dir / "quality_proxy.jsonl", id_key="output_id")
    todo = [g for g in gens if g["id"] not in w and g["prompt_id"] in prompts and g["status"] == "ok" and g["text"]]
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        by_prompt: dict[str, list[dict]] = defaultdict(list)
        for g in chunk:
            by_prompt[g["prompt_id"]].append(g)
        for pid, gs in by_prompt.items():
            ctx = backend.chat_prefix_ids(GENERATOR_SYSTEM, prompts[pid].prompt)
            toks = [backend.encode_text(g["text"]) for g in gs]
            ok = [j for j, t in enumerate(toks) if t and len(ctx) + len(t) <= backend.max_input_tokens]
            if not ok:
                continue
            lps = backend.logprob_tokens(ctx, [toks[j] for j in ok])
            for j, lp in zip(ok, lps):
                arr = np.asarray(lp, dtype=np.float64)
                w.write({"output_id": gs[j]["id"], "mean_logp_task": float(arr.mean()), "n": int(arr.size),
                         "min_logp_task": float(arr.min()), "frac_below_minus8": float((arr < -8).mean())})
    w.close()
    log(f"[quality] task-conditioned log-likelihood computed for {len(todo)} outputs")


def text_stats(text: str) -> dict:
    """Model-free degeneration measures: distinct-2/3 ratios and the longest run of a repeated 4-gram."""
    w = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(w) < 4:
        return {"distinct2": None, "distinct3": None, "max_repeat4": 0}
    g2 = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
    g3 = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    g4 = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    counts: dict = {}
    for g in g4:
        counts[g] = counts.get(g, 0) + 1
    # word-level loops: an 8-word sequence (alphabetic words only) occurring twice or more; ignores markdown/punctuation
    # structure, which inflates max_repeat4 in list-style answers
    words = re.findall(r"[a-z]+", text.lower())
    rep8 = 0
    if len(words) >= 9:
        c8: dict = {}
        for i in range(len(words) - 7):
            g = tuple(words[i:i + 8]); c8[g] = c8.get(g, 0) + 1
        rep8 = sum(v - 1 for v in c8.values() if v > 1)
    return {"distinct2": len(set(g2)) / len(g2), "distinct3": len(set(g3)) / len(g3), "max_repeat4": max(counts.values()),
            "loop8": int(rep8 > 0), "rep8_per_100w": (100.0 * rep8 / len(words)) if words else 0.0}


def _load_scores(run_dir: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in load_jsonl(run_dir / "detections_raw.jsonl"):
        out[r["text_hash"]][r["key_id"]] = r
    if (run_dir / "tstats.jsonl").exists():          # back-filled t-statistics for older records
        for r in load_jsonl(run_dir / "tstats.jsonl"):
            rec = out.get(r["text_hash"], {}).get(r["key_id"])
            if rec is not None and rec.get("t_stat") is None:
                rec["t_stat"], rec["frac_pos"], rec["t_n"] = r["t_stat"], r["frac_pos"], r["t_n"]
    return out


def _is_base_arm(arm: str) -> bool:
    return arm == "base" or arm.startswith("base+")


def _attack_of(arm: str) -> str:
    return arm.split("+", 1)[1] if "+" in arm else ""


DECOY_BANK_FOR_KEY_SET = {"randomwords": "decoy_rw", "randomwords_rare": "decoy_rwr"}


def load_decoys(cfg: Config, key_set: str | None = None, families: set[str] | None = None) -> list:
    """Decoy bank for the rank detector: the same-family bank when one exists for the run's key set or when every target
    key is a seed phrase (random-word targets need random-word decoys to be exchangeable with a wrong key), otherwise the
    catalog bank."""
    names = [DECOY_BANK_FOR_KEY_SET[key_set]] if key_set in DECOY_BANK_FOR_KEY_SET else (["decoy_rw"] if families and set(families) == {"random_words"} else [])
    for name in names + ["decoy"]:
        p = keys_path(cfg, name)
        if p.exists():
            return load_keys(read_json(p))
    return []


def evaluate_run(cfg: Config | None, run_dir: Path, device: str | None = None, score: bool = True, log=print,
                 protocol_path: str | None = None, use_decoys: bool = True) -> dict:
    t0 = time.time()
    manifest = read_json(run_dir / "manifest.json")
    stage = manifest["stage"]
    key_set = manifest.get("key_set") or "development"
    if cfg is None:
        raise ValueError("--config or --protocol is required to evaluate a run")
    keys = load_keys(read_json(keys_path(cfg, key_set)))
    key_recs = {k.key_id: asdict(k) for k in keys}
    decoys = load_decoys(cfg, key_set, {k.family for k in keys}) if use_decoys else []
    # steering decoys (objective 'specific') are excluded from the detection decoy bank: specificity is measured
    # against prefixes the generator never optimized against
    sd_path = cfg.path_of("assets") / "steering_decoys.json"
    steering_ids = {d["key_id"] for fam in (read_json(sd_path)["by_family"].values() if sd_path.exists() else []) for d in fam}
    decoy_ids = [d.key_id for d in decoys if d.key_id not in steering_ids]
    min_vis = int(cfg["detection"]["min_visible_tokens"])
    alpha = float(cfg["detection"]["alpha"])
    views = list(cfg["evaluation"]["length_views"])
    reps = int(cfg["evaluation"]["bootstrap_replicates"])
    gens, refs = collect_run_texts(run_dir)
    log(f"[evaluate] {stage}: {len(gens)} generations, {len(refs)} references; {len(keys)} target keys, {len(decoys)} decoys"
        f" (bank {decoys[0].key_set if decoys else 'none'})")
    if score:
        backend = build_backend(cfg, device)
        score_run(run_dir, backend, keys + decoys, int(cfg["scoring"]["max_input_tokens"]), min_vis, views, all_keys=True, log=log)
        quality_proxy_run(run_dir, backend, cfg, manifest, gens, log=log)
    scores = _load_scores(run_dir)
    qproxy = {r["output_id"]: r for r in load_jsonl(run_dir / "quality_proxy.jsonl")} if (run_dir / "quality_proxy.jsonl").exists() else {}
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- join -------------------------------------------------------------- #
    rows = []
    for g in gens:
        ak = g.get("assigned_key_id")
        sc = scores.get(g["text_hash"], {})
        s_a = sc.get(ak, {}) if ak else {}
        rows.append({
            "output_id": g["id"], "arm": g["arm"], "domain": g["domain"], "group_id": g["group_id"], "seed": g["seed"],
            "key_id": g.get("key_id"), "assigned_key_id": ak, "family": key_recs.get(ak, {}).get("family") if ak else None,
            "visible_tokens": g["visible_tokens"], "accepted_tokens": g["accepted_tokens"], "stop_reason": g["stop_reason"],
            "status": g["status"], "ends_mid_sentence": g["ends_mid_sentence"],
            "score": s_a.get("score"), "score_status": s_a.get("status"), "eligible": bool(s_a.get("eligible", False)),
            "contrast_sum": s_a.get("contrast_sum"), "prefix_views": s_a.get("prefix_views", {}),
            "n_decisions": g.get("n_decisions", 0), "n_active": g.get("n_active_decisions", 0),
            "n_inactive_no_signal": g.get("n_inactive_no_signal", 0), "cost": g.get("cost", {}),
            "text_hash": g["text_hash"], "marked": g.get("key_id") is not None and not _is_base_arm(g["arm"]),
            "leak": leakage(g["text"], key_recs[ak]) if ak else None,
            "logp_task": qproxy.get(g["id"], {}).get("mean_logp_task"),
            "frac_lowlogp": qproxy.get(g["id"], {}).get("frac_below_minus8"),
            "tstats": text_stats(g["text"]),
        })
    ref_rows = []
    for r in refs:
        ref_rows.append({"group_id": r["group_id"], "domain": r["domain"], "text_hash": r["text_hash"],
                         "visible_tokens": r["visible_tokens"], "scores": scores.get(r["text_hash"], {})})
    arms = manifest["arms"]
    fam_of = {k: v["family"] for k, v in key_recs.items()}
    families = sorted(set(fam_of.values()))

    for r in rows:                     # attack runs carry arms "<parent arm>+<attack>[params]": base+... rows are the (attacked) controls
        r["is_base"] = _is_base_arm(r["arm"])
        r["attack"] = _attack_of(r["arm"])
    # ---- toy / real calibration ------------------------------------------- #
    cal_path = cfg.path_of("assets") / "calibration.json"
    frozen_stage = (stage in ("positive_test", "null_test", "topic_null") or stage.startswith("attacks_")
                    or bool((cfg.raw.get("stages", {}).get(stage) or {}).get("frozen")))
    real_cal = Calibration.from_dict(read_json(cal_path)) if (cal_path.exists() and frozen_stage) else None
    if real_cal is None:
        by_key = {}
        for k in key_recs:
            gcoh = [(r["text_hash"], scores[r["text_hash"]][k]["score"]) for r in rows if r["is_base"]
                    and scores.get(r["text_hash"], {}).get(k, {}).get("eligible")]
            hcoh = [(rr["text_hash"], rr["scores"][k]["score"]) for rr in ref_rows if rr["scores"].get(k, {}).get("eligible")]
            by_key[k] = {"generator": gcoh, "human_web": hcoh}
        cal = build_calibration(by_key, {k: keys[i].to_keyspec() for i, k in enumerate(key_recs)}, cfg["scoring"]["scorer_id"], alpha,
                                min_vis, int(cfg["scoring"]["max_input_tokens"]), int(cfg["detection"]["minimum_calibration_per_cohort"]),
                                f"{stage}_base_outputs", manifest.get("config_hash", ""), True, {},
                                meta={"note": "TOY calibration from this run's own base outputs and references; exploratory, in-sample for base"})
        cal_label = "toy"
    else:
        cal = real_cal
        cal_label = "frozen"
    write_json(eval_dir / f"calibration_{cal_label}.json", cal.to_dict())

    def pval(text_hash: str, key_id: str) -> dict | None:
        sc = scores.get(text_hash, {}).get(key_id)
        if not sc:
            return None
        res = ScoreResult(score=sc["score"], status=sc["status"], n_tokens=sc["visible_tokens"], error=sc.get("error"))
        d = detect_from_score(res, sc["visible_tokens"], cal.keyspec(key_id), cal)
        return {"decision": d.decision, "p_final": d.p_final, "p_generator": d.p_generator, "p_human": d.p_human}

    for r in rows:
        r["det"] = pval(r["text_hash"], r["assigned_key_id"]) if r["assigned_key_id"] else None

    # ---- per-arm summaries ------------------------------------------------ #
    base_by = {(r["group_id"], r["seed"], r["attack"]): r for r in rows if r["is_base"]}   # attack runs pair within the same attack
    per_arm = {}
    shifts_by_arm: dict[str, list[float]] = {}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        el = [r for r in ar if r["eligible"]]
        det = [r for r in el if r["det"] and r["det"]["decision"] == "detected"]
        shifts, clusters = [], []
        for r in el:
            b = base_by.get((r["group_id"], r["seed"], r["attack"]))
            if b and b["assigned_key_id"] == r["assigned_key_id"] and b["eligible"] and b["score"] is not None and r["score"] is not None:
                shifts.append(r["score"] - b["score"]); clusters.append(r["group_id"])
        shifts_by_arm[arm] = shifts
        costs = [r["cost"] for r in ar if r["cost"]]
        acc_tokens = sum(r["accepted_tokens"] for r in ar) or 1
        per_arm[arm] = {
            "n": len(ar), "n_ok": sum(1 for r in ar if r["status"] == "ok"), "n_eligible": len(el),
            "coverage": len(el) / max(1, len(ar)),
            "visible_tokens_mean": float(np.mean([r["visible_tokens"] for r in ar])) if ar else None,
            "visible_tokens_median": float(np.median([r["visible_tokens"] for r in ar])) if ar else None,
            "stop_reasons": dict(zip(*np.unique([r["stop_reason"] for r in ar], return_counts=True))) if ar else {},
            "ends_mid_sentence_frac": float(np.mean([r["ends_mid_sentence"] for r in ar])) if ar else None,
            "score_mean": cluster_bootstrap_mean([r["score"] for r in el], [r["group_id"] for r in el], reps, seed_from("bs", arm)),
            "paired_shift_vs_base": cluster_bootstrap_mean(shifts, clusters, reps, seed_from("bs2", arm)),
            "paired_shift_positive_frac": float(np.mean([s > 0 for s in shifts])) if shifts else None,
            f"flag_rate_{cal_label}_alpha{alpha}_conditional": {"k": len(det), "n": len(el), "rate": len(det) / max(1, len(el)),
                                                                 "wilson95": wilson_interval(len(det), len(el))},
            f"flag_rate_{cal_label}_alpha{alpha}_all_supplied": {"k": len(det), "n": len(ar), "rate": len(det) / max(1, len(ar))},
            "inactive_no_signal_frac": float(np.mean([r["n_inactive_no_signal"] / max(1, r["n_decisions"]) for r in ar])) if ar else None,
            "cost": {k: float(np.mean([c.get(k, 0) for c in costs])) for k in ("proposal_tokens", "scoring_tokens", "scoring_forward_calls",
                                                                              "proposal_forward_calls", "wall_seconds", "peak_memory_bytes")} if costs else {},
            "cost_per_accepted_token": {k: sum(c.get(k, 0) for c in costs) / acc_tokens for k in ("proposal_tokens", "scoring_tokens", "wall_seconds")} if costs else {},
            "leakage": {"exact_prefix_frac": float(np.mean([r["leak"]["exact_prefix"] for r in ar if r["leak"]])) if any(r["leak"] for r in ar) else None,
                        "any_term_frac": float(np.mean([r["leak"]["any"] for r in ar if r["leak"]])) if any(r["leak"] for r in ar) else None,
                        "category_any_frac": float(np.mean([r["leak"]["category_any"] for r in ar if r["leak"]])) if any(r["leak"] for r in ar) else None},
        }
        # destructiveness proxy: task-conditioned log-likelihood per token, paired against base on the same prompt/seed
        ql = [r for r in ar if r.get("logp_task") is not None]
        qd, qc = [], []
        for r in ql:
            b = base_by.get((r["group_id"], r["seed"], r["attack"]))
            if b and b.get("logp_task") is not None:
                qd.append(r["logp_task"] - b["logp_task"]); qc.append(r["group_id"])
        per_arm[arm]["quality_proxy"] = {
            "mean_logp_task": cluster_bootstrap_mean([r["logp_task"] for r in ql], [r["group_id"] for r in ql], reps, seed_from("q", arm)),
            "paired_delta_vs_base": cluster_bootstrap_mean(qd, qc, reps, seed_from("qd", arm)),
            "frac_tokens_below_minus8": float(np.mean([r["frac_lowlogp"] for r in ql])) if ql else None,
            "short_outputs_frac": float(np.mean([r["visible_tokens"] < 32 for r in ar])) if ar else None,
            "distinct2_mean": float(np.mean([r["tstats"]["distinct2"] for r in ar if r["tstats"]["distinct2"] is not None])) if ar else None,
            "max_repeat4_mean": float(np.mean([r["tstats"]["max_repeat4"] for r in ar])) if ar else None,
            "degenerate_frac": float(np.mean([r["tstats"]["max_repeat4"] >= 3 for r in ar])) if ar else None,
            "loop8_frac": float(np.mean([r["tstats"].get("loop8", 0) for r in ar])) if ar else None,
            "rep8_per_100w_mean": float(np.mean([r["tstats"].get("rep8_per_100w", 0.0) for r in ar])) if ar else None,
        }
        # nonverbatim subset (spec Section 11)
        nv = [r for r in el if r["leak"] and not r["leak"]["any"]]
        if nv:
            nvd = [r for r in nv if r["det"] and r["det"]["decision"] == "detected"]
            per_arm[arm]["nonverbatim_subset"] = {"n": len(nv), "flag_rate": len(nvd) / len(nv),
                                                  "score_mean": float(np.mean([r["score"] for r in nv]))}

    # ---- discrimination: AUROC per key / family (negatives = base outputs under that key) --------- #
    def neg_for(key_id):
        return [(r["group_id"], scores[r["text_hash"]][key_id]["score"]) for r in rows if r["is_base"]
                and scores.get(r["text_hash"], {}).get(key_id, {}).get("eligible")]

    null_stats = {}
    for k in list(key_recs) + decoy_ids:
        vals = [s for _, s in neg_for(k)]
        null_stats[k] = {"mean": float(np.mean(vals)) if vals else None, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "n": len(vals)}
    # ---- secondary rank-based detector: correct key vs decoy keys on the same text (labelled) ------------ #
    from .calibration import key_z, rank_statistic

    def z_of(th: str, k: str):
        sc = scores.get(th, {}).get(k)
        st = null_stats.get(k) or {}
        if not sc or sc.get("score") is None or not sc.get("eligible") or not st.get("sd"):
            return None
        return key_z(sc["score"], st["mean"], st["sd"])

    def rank_stats(th: str, k: str, decoys: list[str]) -> tuple[float | None, float | None]:
        zt = z_of(th, k)
        zd = [z for z in (z_of(th, d) for d in decoys) if z is not None]
        if zt is None or not zd:
            return None, None
        return rank_statistic(zt, zd), zt - float(np.mean(zd))

    all_key_list = sorted(key_recs)
    use_bank = bool(decoy_ids) and all(null_stats.get(d, {}).get("sd") for d in decoy_ids)

    def decoys_for(k: str) -> list[str]:
        return decoy_ids if use_bank else [d for d in all_key_list if d != k]

    null_rank, null_zdiff = [], []
    for r in rows:
        if (not r["is_base"]):
            continue
        for k in all_key_list:
            rf, zd = rank_stats(r["text_hash"], k, decoys_for(k))
            if rf is not None:
                null_rank.append(rf); null_zdiff.append(zd)
    null_rank.sort(); null_zdiff.sort()
    min_p_rank = tail_rank_p(null_rank, 1.0) if null_rank else None      # discrete floor: p when all decoys are beaten
    for r in rows:
        ak = r["assigned_key_id"]
        if not ak:
            r["rank_det"] = None
            continue
        rf, zd = rank_stats(r["text_hash"], ak, decoys_for(ak))
        rf_fam, _ = rank_stats(r["text_hash"], ak, key_recs[ak]["wrong_key_ids"])
        r["rank_det"] = {"rank_frac": rf, "rank_frac_family": rf_fam, "zdiff": zd,
                         "p_rank": tail_rank_p(null_rank, rf) if (rf is not None and null_rank) else None,
                         "p_zdiff": tail_rank_p(null_zdiff, zd) if (zd is not None and null_zdiff) else None}
    rank_detector = {"note": "SECONDARY detector (amendment): correct key ranked against decoy keys on the same text; "
                             + ("decoy bank (never targets)" if use_bank else "other target keys of the set as decoys")
                             + "; null cohort = base outputs under every target key (pooled, correlated), toy in development",
                     "decoy_bank": use_bank, "n_decoys": len(decoy_ids) if use_bank else len(all_key_list) - 1,
                     "null_cohort_size": len(null_rank), "min_achievable_p_rank_frac": min_p_rank, "per_arm": {}}
    for arm in arms:
        el = [r for r in rows if r["arm"] == arm and r.get("rank_det") and r["rank_det"]["rank_frac"] is not None]
        if not el:
            continue
        pos_r = [(r["group_id"], r["rank_det"]["rank_frac"]) for r in el]
        pos_z = [(r["group_id"], r["rank_det"]["zdiff"]) for r in el]
        neg_r = [(r["group_id"], v) for r in rows if r["is_base"] for v in
                 [rank_stats(r["text_hash"], r["assigned_key_id"], decoys_for(r["assigned_key_id"]))[0]] if v is not None]
        neg_z = [(r["group_id"], v) for r in rows if r["is_base"] for v in
                 [rank_stats(r["text_hash"], r["assigned_key_id"], decoys_for(r["assigned_key_id"]))[1]] if v is not None]
        rank_detector["per_arm"][arm] = {
            "n": len(el),
            "rank_frac_mean": float(np.mean([r["rank_det"]["rank_frac"] for r in el])),
            "top_among_all_keys_frac": float(np.mean([r["rank_det"]["rank_frac"] == 1.0 for r in el])),
            "rank_frac_family_mean": float(np.mean([r["rank_det"]["rank_frac_family"] for r in el if r["rank_det"]["rank_frac_family"] is not None])) if any(r["rank_det"]["rank_frac_family"] is not None for r in el) else None,
            "zdiff_mean": cluster_bootstrap_mean([r["rank_det"]["zdiff"] for r in el], [r["group_id"] for r in el], reps, seed_from("zd", arm)),
            f"flag_rate_rank_frac_alpha{alpha}": {"k": sum(1 for r in el if r["rank_det"]["p_rank"] is not None and r["rank_det"]["p_rank"] <= alpha), "n": len(el)},
            f"flag_rate_zdiff_alpha{alpha}": {"k": sum(1 for r in el if r["rank_det"]["p_zdiff"] is not None and r["rank_det"]["p_zdiff"] <= alpha), "n": len(el)},
            "auroc_rank_frac_vs_base": bootstrap_auroc(pos_r, neg_r, reps, seed_from("aucr", arm)) if not _is_base_arm(arm) else None,
            "auroc_zdiff_vs_base": bootstrap_auroc(pos_z, neg_z, reps, seed_from("aucz", arm)) if not _is_base_arm(arm) else None,
        }

    # ---- within-text t-statistic detector (amendment): t = mean(d)/(sd(d)/sqrt(n)) on per-token contrasts -------- #
    for r in rows:
        sc = scores.get(r["text_hash"], {}).get(r["assigned_key_id"] or "", {}) if r["assigned_key_id"] else {}
        r["t_stat"] = sc.get("t_stat"); r["frac_pos"] = sc.get("frac_pos")
    null_t = sorted(r["t_stat"] for r in rows if r["is_base"] and r["eligible"] and r["t_stat"] is not None)
    t_detector = {"note": "SECONDARY detector (amendment): within-text t-statistic of per-token keyed-minus-neutral contrasts under the "
                          "assigned key; null cohort = base outputs under their assigned keys (toy in development)",
                  "null_cohort_size": len(null_t), "per_arm": {}}
    for arm in arms:
        el = [r for r in rows if r["arm"] == arm and r["eligible"] and r["t_stat"] is not None]
        if not el:
            continue
        for r in el:
            r["p_t"] = tail_rank_p(null_t, r["t_stat"]) if null_t else None
        pos = [(r["group_id"], r["t_stat"]) for r in el]
        neg = [(r["group_id"], r["t_stat"]) for r in rows if r["is_base"] and r["eligible"] and r["t_stat"] is not None]
        t_detector["per_arm"][arm] = {
            "n": len(el), "t_mean": cluster_bootstrap_mean([r["t_stat"] for r in el], [r["group_id"] for r in el], reps, seed_from("t", arm)),
            "frac_pos_mean": float(np.mean([r["frac_pos"] for r in el])),
            f"flag_rate_t_alpha{alpha}": {"k": sum(1 for r in el if r.get("p_t") is not None and r["p_t"] <= alpha), "n": len(el)},
            "auroc_t_vs_base": bootstrap_auroc(pos, neg, reps, seed_from("auct", arm)) if not _is_base_arm(arm) else None,
        }

    per_arm_key = {}
    per_arm_family = {}
    for arm in arms:
        if _is_base_arm(arm):
            continue
        per_arm_key[arm] = {}
        pooled_pos: dict[str, list] = defaultdict(list); pooled_neg: dict[str, list] = defaultdict(list)
        pooled_p_pos: dict[str, list] = defaultdict(list); pooled_p_neg: dict[str, list] = defaultdict(list)
        for k in key_recs:
            pos = [(r["group_id"], r["score"]) for r in rows if r["arm"] == arm and r["assigned_key_id"] == k and r["eligible"]]
            neg = neg_for(k)
            per_arm_key[arm][k] = bootstrap_auroc(pos, neg, reps, seed_from("auc", arm, k))
            per_arm_key[arm][k]["null"] = null_stats[k]
            mu, sd = null_stats[k]["mean"], null_stats[k]["sd"]
            if mu is not None and sd:
                fam = fam_of[k]
                pooled_pos[fam] += [(g, (s - mu) / sd) for g, s in pos]
                pooled_neg[fam] += [(g, (s - mu) / sd) for g, s in neg]
                pooled_pos["all"] += [(g, (s - mu) / sd) for g, s in pos]
                pooled_neg["all"] += [(g, (s - mu) / sd) for g, s in neg]
            for r in rows:
                if r["arm"] == arm and r["assigned_key_id"] == k and r["eligible"] and r["det"] and r["det"]["p_final"] is not None:
                    pooled_p_pos[fam_of[k]].append((r["group_id"], -r["det"]["p_final"])); pooled_p_pos["all"].append((r["group_id"], -r["det"]["p_final"]))
            for r in rows:
                if r["is_base"] and scores.get(r["text_hash"], {}).get(k, {}).get("eligible"):
                    d = pval(r["text_hash"], k)
                    if d and d["p_final"] is not None:
                        pooled_p_neg[fam_of[k]].append((r["group_id"], -d["p_final"])); pooled_p_neg["all"].append((r["group_id"], -d["p_final"]))
        per_arm_family[arm] = {fam: {"auroc_zscored": bootstrap_auroc(pooled_pos[fam], pooled_neg[fam], reps, seed_from("aucf", arm, fam)),
                                     f"auroc_neg_p_{cal_label}": bootstrap_auroc(pooled_p_pos[fam], pooled_p_neg[fam], reps, seed_from("aucp", arm, fam))}
                               for fam in list(families) + ["all"]}

    # ---- key specificity / confusion ------------------------------------ #
    key_ids = sorted(key_recs)
    kidx = {k: i for i, k in enumerate(key_ids)}
    confusion = {}
    specificity = {}
    for arm in arms:
        marked = [r for r in rows if r["arm"] == arm and r["marked"] and r["eligible"]]
        if not marked:
            continue
        M = np.zeros((len(key_ids), len(key_ids)), dtype=int)
        top1_family = top1_all = 0
        wrong_diffs, wrong_clusters = [], []
        false_attr = 0; false_attr_n = 0
        for r in marked:
            sc = scores.get(r["text_hash"], {})
            k = r["assigned_key_id"]
            zs = {kk: (v["score"] - null_stats[kk]["mean"]) / null_stats[kk]["sd"] for kk, v in sc.items()
                  if kk in key_recs and v.get("score") is not None and null_stats.get(kk, {}).get("sd")}   # target keys only
            if not zs or k not in zs:
                continue
            best_all = max(zs, key=zs.get)
            M[kidx[k], kidx[best_all]] += 1
            top1_all += int(best_all == k)
            fam_keys = [kk for kk in key_recs[k]["wrong_key_ids"] if kk in zs]
            if fam_keys:
                best_fam = max([k] + fam_keys, key=zs.get)
                top1_family += int(best_fam == k)
                wrong_mean = float(np.mean([sc[kk]["score"] for kk in fam_keys]))
                wrong_diffs.append(r["score"] - wrong_mean); wrong_clusters.append(r["group_id"])
                for kk in fam_keys:            # false attribution: a wrong key flags this text
                    d = pval(r["text_hash"], kk)
                    false_attr_n += 1
                    false_attr += int(bool(d and d["decision"] == "detected"))
        confusion[arm] = {"matrix": M.tolist(), "labels": key_ids}
        specificity[arm] = {"n": len(marked), "top1_among_all_keys": top1_all / len(marked),
                            "top1_within_family_stratum": top1_family / max(1, len(marked)),
                            "score_minus_wrong_key_mean": cluster_bootstrap_mean(wrong_diffs, wrong_clusters, reps, seed_from("wk", arm)),
                            f"false_attribution_rate_{cal_label}": {"k": false_attr, "n": false_attr_n,
                                                                    "rate": false_attr / max(1, false_attr_n)}}

    # ---- per domain / per family / length views ----------------------------- #
    per_domain = {}
    for arm in arms:
        per_domain[arm] = {}
        for dom in cfg["data"]["domains"]:
            el = [r for r in rows if r["arm"] == arm and r["domain"] == dom and r["eligible"]]
            det = [r for r in el if r["det"] and r["det"]["decision"] == "detected"]
            per_domain[arm][dom] = {"n": len(el), "score_mean": float(np.mean([r["score"] for r in el])) if el else None,
                                    "flag_rate": len(det) / max(1, len(el)), "k": len(det),
                                    "visible_tokens_mean": float(np.mean([r["visible_tokens"] for r in el])) if el else None}
    length_view_stats = {}
    for arm in arms:
        length_view_stats[arm] = {}
        for L in views:
            vals = [r["prefix_views"].get(str(L), {}).get("score") for r in rows if r["arm"] == arm]
            vals = [v for v in vals if v is not None]
            length_view_stats[arm][str(L)] = {"n": len(vals), "score_mean": float(np.mean(vals)) if vals else None}
    # score vs length correlation
    length_corr = {}
    for arm in arms:
        pts = [(r["visible_tokens"], r["score"]) for r in rows if r["arm"] == arm and r["eligible"]]
        if len(pts) > 3:
            x, y = zip(*pts)
            length_corr[arm] = float(np.corrcoef(x, y)[0, 1])
    # reference cohort scores under assigned keys are not defined; report reference null stats per key
    ref_stats = {k: {"n": sum(1 for rr in ref_rows if rr["scores"].get(k, {}).get("eligible")),
                     "mean": float(np.mean([rr["scores"][k]["score"] for rr in ref_rows if rr["scores"].get(k, {}).get("eligible")]))
                     if any(rr["scores"].get(k, {}).get("eligible") for rr in ref_rows) else None} for k in key_recs}

    results = {
        "stage": stage, "run_dir": str(run_dir), "calibration": cal_label, "calibration_id": cal.calibration_id, "alpha": alpha,
        "n_generations": len(gens), "n_references": len(refs), "arms": arms, "keys": key_ids, "families": families,
        "per_arm": per_arm, "per_arm_key_auroc": per_arm_key, "per_arm_family": per_arm_family, "specificity": specificity,
        "confusion": confusion, "per_domain": per_domain, "length_views": length_view_stats, "score_length_correlation": length_corr,
        "null_stats_by_key": null_stats, "reference_stats_by_key": ref_stats, "rank_detector": rank_detector, "t_detector": t_detector,
        "notes": ["Development/smoke p-values use TOY calibration from this run's own base outputs (in-sample for base); "
                  "they are exploratory and not a false-positive estimate." if cal_label == "toy" else "frozen calibration"],
        "elapsed_seconds": time.time() - t0,
    }
    write_json(eval_dir / "results.json", results)
    _write_tables(eval_dir, rows, per_arm, per_arm_key, per_domain, confusion)
    fig_paths = _make_plots(eval_dir / "figures", rows, arms, shifts_by_arm, per_domain, confusion, per_arm, views, cal_label)
    results["figures"] = fig_paths
    write_json(eval_dir / "results.json", results)
    _write_run_report(eval_dir, results, cfg)
    _write_samples(eval_dir, cfg, manifest, gens, rows, key_recs)
    log(f"[evaluate] wrote {eval_dir}/results.json, tables, figures, report.md, samples.md in {time.time() - t0:.0f}s")
    return results


def _write_samples(eval_dir: Path, cfg: Config, manifest: dict, gens: list[dict], rows: list[dict], key_recs: dict) -> None:
    """Human-readable side-by-side comparison of every prompt's outputs across arms (for visual inspection)."""
    from .data.prepare import load_prepared
    prompts = {p.prompt_id: p for p in load_prepared(cfg.path_of("data_prepared"), partition=manifest["partition"])}
    by_row = {r["output_id"]: r for r in rows}
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for g in gens:
        by_prompt[g["prompt_id"]].append(g)
    arms = manifest["arms"]
    L = [f"# Sample outputs: `{manifest['stage']}`\n",
         "Each block shows the task, the secret prefix assigned to that prompt, and every arm's output with its whole-text score "
         "F_k(y) under that prefix (nats/token; higher = the key-conditioned world prefers the text more than the neutral world), "
         "the toy/frozen p-value, and literal-leakage flags. `base` is the clean generator.\n"]
    for pid in sorted(by_prompt, key=lambda x: (prompts[x].domain if x in prompts else "", x)):
        outs = by_prompt[pid]
        p = prompts.get(pid)
        ak = outs[0].get("assigned_key_id")
        L.append(f"\n---\n\n## {pid}  (domain: {p.domain if p else '?'}, group {outs[0]['group_id']})\n")
        L.append("**Task:**\n\n> " + (p.prompt.replace("\n", "\n> ") if p else "(prompt text unavailable)") + "\n")
        if ak and ak in key_recs:
            L.append(f"**Assigned key `{ak}`** ({key_recs[ak]['family']}): `{key_recs[ak]['prefix']}`\n")
        for arm in arms:
            for g in [x for x in outs if x["arm"] == arm]:
                r = by_row.get(g["id"], {})
                det = r.get("det") or {}
                leak = r.get("leak") or {}
                sc = r.get("score")
                L.append(f"\n### {arm}  (seed {g['seed']}, {g['visible_tokens']} tokens, stop={g['stop_reason']}, "
                         f"F={sc if sc is None else round(sc, 4)}, p={det.get('p_final') if det.get('p_final') is None else round(det['p_final'], 4)}, "
                         f"decision={det.get('decision')}, leak_terms={leak.get('term_hits', [])}, exact_prefix={leak.get('exact_prefix')})\n")
                L.append("```text\n" + g["text"].replace("```", "'''") + "\n```\n")
    (eval_dir / "samples.md").write_text("\n".join(L), encoding="utf-8")
    with (eval_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "domain", "assigned_key_id", "key_prefix", "arm", "seed", "score", "p_final", "decision", "prompt", "text"])
        for pid, outs in by_prompt.items():
            p = prompts.get(pid)
            for g in outs:
                r = by_row.get(g["id"], {})
                det = r.get("det") or {}
                ak = g.get("assigned_key_id")
                w.writerow([pid, g["domain"], ak, key_recs.get(ak, {}).get("prefix") if ak else None, g["arm"], g["seed"], r.get("score"),
                            det.get("p_final"), det.get("decision"), p.prompt if p else None, g["text"]])


def _write_tables(eval_dir: Path, rows, per_arm, per_arm_key, per_domain, confusion) -> None:
    with (eval_dir / "outputs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["output_id", "arm", "domain", "group_id", "seed", "key_id", "assigned_key_id", "family", "visible_tokens",
                    "stop_reason", "eligible", "score", "contrast_sum", "p_final", "decision", "leak_any", "leak_exact",
                    "proposal_tokens", "scoring_tokens", "wall_seconds"])
        for r in rows:
            d = r["det"] or {}
            w.writerow([r["output_id"], r["arm"], r["domain"], r["group_id"], r["seed"], r["key_id"], r["assigned_key_id"], r["family"],
                        r["visible_tokens"], r["stop_reason"], r["eligible"], r["score"], r["contrast_sum"], d.get("p_final"),
                        d.get("decision"), (r["leak"] or {}).get("any"), (r["leak"] or {}).get("exact_prefix"),
                        r["cost"].get("proposal_tokens"), r["cost"].get("scoring_tokens"), r["cost"].get("wall_seconds")])
    with (eval_dir / "per_arm.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n", "n_eligible", "visible_tokens_mean", "score_mean", "score_ci_lo", "score_ci_hi", "paired_shift",
                    "shift_ci_lo", "shift_ci_hi", "flag_rate_conditional", "flag_k", "proposal_tokens_mean", "scoring_tokens_mean",
                    "wall_seconds_mean", "leak_any_frac"])
        for arm, a in per_arm.items():
            fr = [v for k, v in a.items() if k.startswith("flag_rate_") and k.endswith("_conditional")][0]
            w.writerow([arm, a["n"], a["n_eligible"], a["visible_tokens_mean"], a["score_mean"]["mean"], *a["score_mean"]["ci95"],
                        a["paired_shift_vs_base"]["mean"], *a["paired_shift_vs_base"]["ci95"], fr["rate"], fr["k"],
                        a["cost"].get("proposal_tokens"), a["cost"].get("scoring_tokens"), a["cost"].get("wall_seconds"),
                        a["leakage"]["any_term_frac"]])
    with (eval_dir / "per_arm_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "key_id", "auroc", "ci_lo", "ci_hi", "n_pos", "n_neg", "null_mean", "null_sd"])
        for arm, d in per_arm_key.items():
            for k, v in d.items():
                w.writerow([arm, k, v["auroc"], *v["ci95"], v["n_pos"], v["n_neg"], v["null"]["mean"], v["null"]["sd"]])
    with (eval_dir / "per_arm_domain.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "domain", "n", "score_mean", "flag_rate", "k", "visible_tokens_mean"])
        for arm, d in per_domain.items():
            for dom, v in d.items():
                w.writerow([arm, dom, v["n"], v["score_mean"], v["flag_rate"], v["k"], v["visible_tokens_mean"]])
    with (eval_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for arm, c in confusion.items():
            w.writerow([arm] + c["labels"])
            for lab, row in zip(c["labels"], c["matrix"]):
                w.writerow([lab] + row)


def _make_plots(fig_dir: Path, rows, arms, shifts_by_arm, per_domain, confusion, per_arm, views, cal_label) -> dict:
    out = {}
    out["score_distributions"] = plots.score_distributions({a: [r["score"] for r in rows if r["arm"] == a and r["eligible"]] for a in arms},
                                                           fig_dir, "Clean score distributions (assigned key)")
    out["score_vs_length"] = plots.score_vs_length({a: [(r["visible_tokens"], r["score"]) for r in rows if r["arm"] == a and r["eligible"]] for a in arms}, fig_dir)
    curves = {}
    for a in arms:
        pts = []
        for L in views:
            vals = [r["prefix_views"].get(str(L), {}).get("score") for r in rows if r["arm"] == a]
            vals = [v for v in vals if v is not None]
            if vals:
                pts.append((L, float(np.mean(vals)), len(vals)))
        full = [r for r in rows if r["arm"] == a and r["eligible"] and r["det"]]
        curves[a] = pts
    out["prefix_score_vs_length"] = plots.detection_vs_length(curves, fig_dir, name="prefix_score_vs_length",
                                                              title="Mean prefix score versus prefix length (correlated views)")
    out["per_domain_flags"] = plots.per_domain_bars({a: {d: v["flag_rate"] for d, v in per_domain[a].items()} for a in arms}, fig_dir,
                                                    f"flag rate ({cal_label} calibration)", "Per-domain flag rates (base = false positives)", "per_domain_flags")
    for arm, c in confusion.items():
        out[f"confusion_{arm}"] = plots.confusion_heatmap(np.asarray(c["matrix"]), c["labels"], fig_dir, f"Key confusion: {arm}", name=f"key_confusion_{arm}")
    pts = []
    for a in arms:
        pa = per_arm[a]
        cpt = pa["cost_per_accepted_token"]
        fr = [v for k, v in pa.items() if k.startswith("flag_rate_") and k.endswith("_conditional")][0]["rate"]
        pts.append({"arm": a, "cost_per_token": (cpt.get("proposal_tokens", 0) + cpt.get("scoring_tokens", 0)) if cpt else 0,
                    "detection": fr, "quality": None})
    out["tradeoff"] = plots.tradeoff(pts, fig_dir)
    out["dilution"] = plots.dilution_curves({}, fig_dir)
    out["paired_shift"] = plots.paired_shift(shifts_by_arm, fig_dir)
    return out


def _write_run_report(eval_dir: Path, res: dict, cfg: Config) -> None:
    L = []
    L.append(f"# Run evaluation: `{res['stage']}`\n")
    L.append(f"Calibration: **{res['calibration']}** (`{res['calibration_id']}`), alpha={res['alpha']}. "
             f"Generations: {res['n_generations']}, references: {res['n_references']}. Keys: {len(res['keys'])}.\n")
    for n in res["notes"]:
        L.append(f"> {n}\n")
    L.append("\n## Per arm\n")
    L.append("| arm | n | eligible | tokens (mean) | F mean [95% CI] | paired shift vs base [95% CI] | flag rate (cond.) | leak any | cat. leak | Δ logp(task) vs base [95% CI] | proposal tok/out | scoring tok/out | s/out |")
    L.append("|---|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|")
    for arm, a in res["per_arm"].items():
        fr = [v for k, v in a.items() if k.startswith("flag_rate_") and k.endswith("_conditional")][0]
        sm, ps = a["score_mean"], a["paired_shift_vs_base"]
        L.append(f"| {arm} | {a['n']} | {a['n_eligible']} | {a['visible_tokens_mean'] and round(a['visible_tokens_mean'], 1)} | "
                 f"{sm['mean']:.4f} [{sm['ci95'][0]:.4f}, {sm['ci95'][1]:.4f}] | "
                 + (f"{ps['mean']:.4f} [{ps['ci95'][0]:.4f}, {ps['ci95'][1]:.4f}] (n={ps['n']})" if ps['n'] else "n/a")
                 + f" | {fr['k']}/{fr['n']} = {fr['rate']:.3f} | {a['leakage']['any_term_frac']} | {a['leakage'].get('category_any_frac')} | "
                 + (lambda q: (f"{q['mean']:.3f} [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}] (n={q['n']})" if q and q.get('n') else "n/a"))((a.get('quality_proxy') or {}).get('paired_delta_vs_base'))
                 + f" | {a['cost'].get('proposal_tokens', 0):.0f} | {a['cost'].get('scoring_tokens', 0):.0f} | {a['cost'].get('wall_seconds', 0):.1f} |")
    L.append("\n## Discrimination (negatives = base outputs under the same key; z-scored per key, pooled)\n")
    for arm, fams in res["per_arm_family"].items():
        parts = []
        for fam, v in fams.items():
            z = v["auroc_zscored"]
            parts.append(f"{fam}: {z['auroc'] if z['auroc'] is None else round(z['auroc'], 3)} (n+={z['n_pos']}, n-={z['n_neg']})")
        L.append(f"- **{arm}**: " + "; ".join(parts))
    L.append("\n## Key specificity\n")
    for arm, s in res["specificity"].items():
        d = s["score_minus_wrong_key_mean"]
        fa = [v for k, v in s.items() if k.startswith("false_attribution")][0]
        L.append(f"- **{arm}**: top-1 within family/stratum {s['top1_within_family_stratum']:.3f}, among all keys {s['top1_among_all_keys']:.3f}; "
                 f"F(correct) - mean F(wrong, same family) = {d['mean']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]; "
                 f"false attribution {fa['k']}/{fa['n']}")
    rd = res.get("rank_detector") or {}
    if rd.get("per_arm"):
        L.append(f"\n## Secondary rank-based detector (labelled amendment; {rd['n_decoys']} decoy keys; null cohort {rd['null_cohort_size']}; "
                 f"smallest achievable p for 'beats all decoys' = {rd['min_achievable_p_rank_frac'] if rd['min_achievable_p_rank_frac'] is None else round(rd['min_achievable_p_rank_frac'], 4)})\n")
        L.append("| arm | n | mean fraction of decoys beaten | beats all decoys | z(correct) - mean z(decoys) [95% CI] | AUROC rank-frac vs base | AUROC z-diff vs base | flags z-diff |")
        L.append("|---|---:|---:|---:|---|---:|---:|---:|")
        for arm, v in rd["per_arm"].items():
            ar = v["auroc_rank_frac_vs_base"] or {}; az = v["auroc_zdiff_vs_base"] or {}
            fz = [x for k, x in v.items() if k.startswith("flag_rate_zdiff")][0]
            zd = v["zdiff_mean"]
            L.append(f"| {arm} | {v['n']} | {v['rank_frac_mean']:.3f} | {v['top_among_all_keys_frac']:.3f} | {zd['mean']:.3f} [{zd['ci95'][0]:.3f}, {zd['ci95'][1]:.3f}] | "
                     f"{ar.get('auroc') if ar.get('auroc') is None else round(ar['auroc'], 3)} | {az.get('auroc') if az.get('auroc') is None else round(az['auroc'], 3)} | {fz['k']}/{fz['n']} |")
    td = res.get("t_detector") or {}
    if td.get("per_arm"):
        L.append(f"\n## Secondary within-text t-statistic detector (labelled amendment; null cohort {td['null_cohort_size']})\n")
        L.append("| arm | n | t mean [95% CI] | frac tokens > 0 | AUROC t vs base | flags t |")
        L.append("|---|---:|---|---:|---:|---:|")
        for arm, v in td["per_arm"].items():
            a = v["auroc_t_vs_base"] or {}; f = [x for k, x in v.items() if k.startswith("flag_rate_t")][0]; tm = v["t_mean"]
            L.append(f"| {arm} | {v['n']} | {tm['mean']:.2f} [{tm['ci95'][0]:.2f}, {tm['ci95'][1]:.2f}] | {v['frac_pos_mean']:.3f} | "
                     f"{a.get('auroc') if a.get('auroc') is None else round(a['auroc'], 3)} | {f['k']}/{f['n']} |")
    L.append("\n## Per domain (flag rate; for `base` this is the false-positive rate under the toy/frozen calibration)\n")
    L.append("| arm | " + " | ".join(cfg["data"]["domains"]) + " |")
    L.append("|---|" + "---|" * len(cfg["data"]["domains"]))
    for arm, d in res["per_domain"].items():
        L.append(f"| {arm} | " + " | ".join(f"{v['k']}/{v['n']} ({v['score_mean'] if v['score_mean'] is None else round(v['score_mean'], 4)})" for v in d.values()) + " |")
    L.append("\n## Length\n")
    L.append("Score-length correlation per arm: " + ", ".join(f"{a}: {c:.3f}" for a, c in res["score_length_correlation"].items()))
    L.append("\nPrefix views (mean F at prefix length; correlated views of the same outputs):\n")
    for arm, d in res["length_views"].items():
        L.append(f"- {arm}: " + ", ".join(f"{Lx}: {v['score_mean'] if v['score_mean'] is None else round(v['score_mean'], 4)} (n={v['n']})" for Lx, v in d.items()))
    L.append("\n## Figures\n")
    for k, v in res.get("figures", {}).items():
        L.append(f"- {k}: `{v[0]}`")
    (eval_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# frozen calibration from null_calibration (spec Section 9)
# --------------------------------------------------------------------------- #

def calibrate_from_null(cfg: Config, device: str | None = None, log=print) -> Path:
    run_dir = cfg.path_of("runs") / "null_calibration"
    manifest = read_json(run_dir / "manifest.json")
    keys = load_keys(read_json(keys_path(cfg, manifest["key_set"])))
    decoys = load_decoys(cfg, manifest["key_set"], {k.family for k in keys})
    min_vis = int(cfg["detection"]["min_visible_tokens"])
    backend = build_backend(cfg, device)
    # scoring is resumable: already-scored (text, key) pairs are skipped
    score_run(run_dir, backend, keys + decoys, int(cfg["scoring"]["max_input_tokens"]), min_vis, [], all_keys=True, log=log)
    bi = asdict(backend.info)
    scores = _load_scores(run_dir)
    gens, refs = collect_run_texts(run_dir)
    by_key = {}
    for k in keys:
        g = [(x["text_hash"], scores[x["text_hash"]][k.key_id]["score"]) for x in gens if x["arm"] == "base"
             and scores.get(x["text_hash"], {}).get(k.key_id, {}).get("eligible")]
        h = [(x["text_hash"], scores[x["text_hash"]][k.key_id]["score"]) for x in refs
             if scores.get(x["text_hash"], {}).get(k.key_id, {}).get("eligible")]
        by_key[k.key_id] = {"generator": g, "human_web": h}
    part_hash = content_hash(sorted(x["group_id"] for x in gens))
    cal = build_calibration(by_key, {k.key_id: k.to_keyspec() for k in keys}, cfg["scoring"]["scorer_id"], float(cfg["detection"]["alpha"]),
                            min_vis, int(cfg["scoring"]["max_input_tokens"]), int(cfg["detection"]["minimum_calibration_per_cohort"]),
                            "null_calibration", part_hash, False, bi,
                            meta={"eligibility": {"min_visible_tokens": min_vis, "max_input_tokens": cfg["scoring"]["max_input_tokens"]},
                                  "protocol": cfg.raw.get("_protocol")})
    # secondary rank-based detector assets from the generator cohort scored under every target key and decoy
    from .calibration import build_secondary
    bank = decoys if decoys else keys
    all_key_scores = {}
    for x in gens:
        if x["arm"] != "base":
            continue
        s = {k.key_id: scores[x["text_hash"]][k.key_id]["score"] for k in list(keys) + list(decoys)
             if scores.get(x["text_hash"], {}).get(k.key_id, {}).get("eligible")}
        if len(s) == len(keys) + len(decoys):
            all_key_scores[x["text_hash"]] = s
    sec = build_secondary(all_key_scores, [k.key_id for k in keys], [k.key_id for k in decoys] if decoys else None)
    cal.secondary = sec
    for d in decoys:                                     # decoy specs travel with the calibration so `detect` can score them
        cal.keys[d.key_id] = d.to_keyspec().to_dict()
    cal.calibration_id = "cal_" + content_hash(cal.to_dict(), length=12)
    out = cfg.path_of("assets") / "calibration.json"
    write_json(out, cal.to_dict())
    modes = cal.meta.get("modes", {})
    log(f"[calibrate] wrote {out} ({cal.calibration_id}); cohorts per key: " +
        ", ".join(f"{k}: G={len(v.generator_scores)} H={len(v.human_scores)} {modes.get(k)}" for k, v in list(cal.per_key.items())[:5]) + " ...")
    return out
