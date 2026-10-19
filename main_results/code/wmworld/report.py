"""Cross-stage report (spec Section 13): ``results.json``, ``report.md`` and the
figure index, answering the six research questions from whatever stages exist.
Missing stages are labelled missing; nothing is filled in.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .config import Config
from .records import read_json, write_json

RQ = [
    "Can active steering increase a key-specific whole-text score on unseen tasks without unacceptable quality loss?",
    "Does the correct key score better than matched incorrect keys, beyond generic topic and length effects?",
    "Does optimizing the cumulative text work better than optimizing isolated generation units when both use whole-text detection?",
    "Does detection survive paraphrasing and changes in sentence boundaries?",
    "Do rare natural prefixes or multiple phrases improve the quality-detection tradeoff over common prefixes and random digits?",
    "How do signal, quality, and cost vary with output length, steering coverage, and candidate count?",
]


def _load_results(run_dir: Path) -> dict | None:
    p = run_dir / "eval" / "results.json"
    return read_json(p) if p.exists() else None


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _arm_line(res: dict, arm: str) -> str:
    a = res["per_arm"].get(arm)
    if not a:
        return f"| {arm} | missing | | | | |"
    fr = [v for k, v in a.items() if k.startswith("flag_rate_") and k.endswith("_conditional")][0]
    ps = a["paired_shift_vs_base"]
    fam = res["per_arm_family"].get(arm, {}).get("all", {}).get("auroc_zscored", {})
    return (f"| {arm} | {a['n_eligible']}/{a['n']} | {_fmt(a['visible_tokens_mean'], 1)} | {_fmt(a['score_mean']['mean'])} "
            f"[{_fmt(a['score_mean']['ci95'][0])}, {_fmt(a['score_mean']['ci95'][1])}] | "
            + (f"{_fmt(ps['mean'])} [{_fmt(ps['ci95'][0])}, {_fmt(ps['ci95'][1])}]" if ps["n"] else "n/a")
            + f" | {_fmt(fam.get('auroc'), 3)} | {fr['k']}/{fr['n']} | {_fmt(a['leakage']['any_term_frac'], 3)} | "
            f"{_fmt(a['cost'].get('scoring_tokens'), 0)} / {_fmt(a['cost'].get('proposal_tokens'), 0)} / {_fmt(a['cost'].get('wall_seconds'), 1)} |")


def build_report(cfg: Config, log=print) -> Path:
    runs = cfg.path_of("runs")
    out_dir = runs / "report"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    stages = {}
    for name in ["smoke", "dev", "positive_test", "null_test", "topic_null"] + sorted(p.name for p in runs.glob("ablation*")) + sorted(p.name for p in runs.glob("attacks_*")):
        r = _load_results(runs / name)
        if r:
            stages[name] = r
            for k, paths in (r.get("figures") or {}).items():
                for p in paths:
                    if p.endswith(".png"):
                        shutil.copy(p, fig_dir / f"{name}_{Path(p).name}")
    selftest = read_json(runs / "selftest.json") if (runs / "selftest.json").exists() else None
    scales = read_json(cfg.path_of("assets") / "scales.json") if (cfg.path_of("assets") / "scales.json").exists() else None
    fit_diag = read_json(runs / "fit_scorer_scale_fit" / "diagnostics.json") if (runs / "fit_scorer_scale_fit" / "diagnostics.json").exists() else None
    ifeval = read_json(runs / "ifeval" / "results.json") if (runs / "ifeval" / "results.json").exists() else None
    calibration = read_json(cfg.path_of("assets") / "calibration.json") if (cfg.path_of("assets") / "calibration.json").exists() else None
    protocol = cfg.raw.get("_protocol")
    proto = read_json(protocol["path"]) if protocol else None
    judge = {}
    for name in stages:
        p = runs / name / "quality" / "model_judge.jsonl"
        if p.exists():
            from .quality import summarize_judge
            judge[name] = summarize_judge(runs / name)

    L = ["# wmworld report: prefix-conditioned world-model watermarking (Qwen3.5-0.8B)\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Model `{cfg['models']['shared']['id']}` @ `{cfg.model_revision}`, "
         f"dtype {cfg['models']['shared']['dtype']}, scorer `{cfg['scoring']['scorer_id']}`.\n",
         "**Exploratory (development) results and frozen (confirmation) results are separated below.** Development p-values use toy "
         "calibration from the run's own base outputs and are not false-positive estimates.\n"]
    L.append("\n## Stage status\n")
    L.append("| stage | status |\n|---|---|")
    for name in ["selftest", "fit_scorer_scale_fit", "smoke", "dev", "ablation", "freeze", "null_calibration", "null_test", "positive_test", "topic_null", "attacks", "ifeval"]:
        if name == "selftest":
            st = f"passed {sum(1 for v in selftest['checks'].values() if v['passed'])}/{len(selftest['checks'])}" if selftest else "MISSING"
        elif name == "fit_scorer_scale_fit":
            st = f"present (discrepancy {fit_diag['numerical_discrepancy']:.2e}, floor {fit_diag['floor']:.2e})" if fit_diag else "MISSING"
        elif name == "freeze":
            st = f"protocol {proto['content_hash']}" if proto else "MISSING (not frozen)"
        elif name == "null_calibration":
            st = f"calibration {calibration['calibration_id']} ({len(calibration['per_key'])} keys)" if calibration else "MISSING"
        elif name == "ablation":
            names = [n for n in stages if n.startswith("ablation")]
            st = f"{len(names)} configuration run(s)" if names else "MISSING"
        elif name == "attacks":
            names = [n for n in stages if n.startswith("attacks_")]
            st = ", ".join(names) if names else "MISSING"
        elif name == "ifeval":
            st = "present" if ifeval else "MISSING"
        else:
            st = "present" if name in stages else "MISSING"
        L.append(f"| {name} | {st} |")

    for name in ["smoke", "dev"]:
        if name in stages:
            r = stages[name]
            L.append(f"\n## {name} (exploratory; toy calibration `{r['calibration_id']}`)\n")
            L.append("| arm | eligible/n | tokens | F mean [95% CI] | paired shift vs base [95% CI] | z-AUROC (all keys) | toy flags | leak frac | scoring/proposal tok per out / s |")
            L.append("|---|---:|---:|---|---|---:|---:|---:|---|")
            for arm in r["arms"]:
                L.append(_arm_line(r, arm))
            L.append("\nPer-family z-scored AUROC (positives = marked outputs with the key; negatives = base outputs under the same key):\n")
            for arm, fams in r["per_arm_family"].items():
                L.append(f"- {arm}: " + "; ".join(f"{f}={_fmt(v['auroc_zscored']['auroc'], 3)} (n+={v['auroc_zscored']['n_pos']})" for f, v in fams.items()))
            L.append("\nKey specificity:\n")
            for arm, s in r["specificity"].items():
                d = s["score_minus_wrong_key_mean"]
                L.append(f"- {arm}: top-1 within family/stratum {s['top1_within_family_stratum']:.3f}; among all keys {s['top1_among_all_keys']:.3f}; "
                         f"F(correct) - mean F(wrong same-family) = {_fmt(d['mean'])} [{_fmt(d['ci95'][0])}, {_fmt(d['ci95'][1])}]")
            L.append("\nScore-length correlation: " + ", ".join(f"{a}: {c:.3f}" for a, c in r["score_length_correlation"].items()))
            if name in judge:
                L.append("\nModel judge (exploratory self-evaluation): " + json.dumps(judge[name]["model_judge"]))
    if "positive_test" in stages:
        r = stages["positive_test"]
        L.append(f"\n## Confirmation (frozen calibration `{r['calibration_id']}`, alpha={r['alpha']})\n")
        L.append("| arm | eligible/n | tokens | F mean [95% CI] | paired shift vs base | AUROC(-p) all | detected (cond.) | leak frac | cost |")
        L.append("|---|---:|---:|---|---|---:|---:|---:|---|")
        for arm in r["arms"]:
            L.append(_arm_line(r, arm))
        L.append("\nPer-domain flag rates (base rows are false positives under the frozen threshold):\n")
        for arm, d in r["per_domain"].items():
            L.append(f"- {arm}: " + ", ".join(f"{dom}: {v['k']}/{v['n']}" for dom, v in d.items()))
    else:
        L.append("\n## Confirmation\n\nMISSING: confirmation stages not run.\n")
    if "null_test" in stages:
        r = stages["null_test"]
        b = r["per_arm"].get("base", {})
        fr = [v for k, v in b.items() if k.startswith("flag_rate_") and k.endswith("_conditional")]
        L.append(f"\n## Independent false positives (null_test)\n\nbase generator cohort: {fr[0]['k']}/{fr[0]['n']} flagged "
                 f"(Wilson 95% {fr[0]['wilson95']})" if fr else "\n## null_test\n\npresent but no base rows")
    if "topic_null" in stages:
        r = stages["topic_null"]
        b = r["per_arm"].get("base", {})
        fr = [v for k, v in b.items() if k.startswith("flag_rate_") and k.endswith("_conditional")]
        L.append(f"\n## Topic-matched null challenge\n\n{fr[0]['k']}/{fr[0]['n']} ordinary on-topic responses flagged under their key "
                 f"(domain-shift diagnostic; thresholds unchanged)" if fr else "\n## Topic-matched null\n\npresent")
    att = [n for n in stages if n.startswith("attacks_")]
    if att:
        L.append("\n## Robustness (unchanged clean-text detector and threshold)\n")
        for name in att:
            r = stages[name]
            L.append(f"\n### {name}\n")
            L.append("| attacked arm | eligible/n | F mean | paired shift vs (unattacked) base | flags (cond.) |")
            L.append("|---|---:|---:|---|---:|")
            for arm in r["arms"]:
                a = r["per_arm"][arm]
                fr = [v for k, v in a.items() if k.startswith("flag_rate_") and k.endswith("_conditional")][0]
                L.append(f"| {arm} | {a['n_eligible']}/{a['n']} | {_fmt(a['score_mean']['mean'])} | n/a | {fr['k']}/{fr['n']} |")
    else:
        L.append("\n## Robustness\n\nMISSING: attack stage not run.\n")
    L.append("\n## IFEval\n")
    L.append(json.dumps(ifeval["arms"], indent=1) if ifeval else "MISSING: IFEval not run.")
    L.append("\n## Numerical guards and failed configurations\n")
    if fit_diag:
        inactive = {k: v for k, v in fit_diag["entries"].items() if v["status"] != "ok"}
        L.append(f"- fit-scorer numerical discrepancy {fit_diag['numerical_discrepancy']:.2e}; floor {fit_diag['floor']:.2e}; "
                 f"non-usable scale entries: {len(inactive)} ({', '.join(list(inactive)[:6])}{'...' if len(inactive) > 6 else ''})")
        ps = fit_diag["prefix_score_by_key"]
        L.append(f"- prefix sensitivity (mean F over unwatermarked prefixes, per key): min {min(v['mean'] for v in ps.values()):.3f}, "
                 f"max {max(v['mean'] for v in ps.values()):.3f}; typical SD {sum(v['sd'] for v in ps.values()) / len(ps):.3f}")
        corr = list(fit_diag["wrong_key_prefix_score_correlation"].values())
        if corr:
            L.append(f"- wrong-key prefix-score correlation: mean {sum(corr) / len(corr):.3f} (nonspecific component)")
    else:
        L.append("- fit-scorer diagnostics MISSING")
    if selftest:
        failed = [k for k, v in selftest["checks"].items() if not v["passed"]]
        L.append(f"- selftest failures: {failed or 'none'}")
    for name, r in stages.items():
        for arm, a in r["per_arm"].items():
            if a.get("n_ok", a["n"]) < a["n"]:
                L.append(f"- {name}/{arm}: {a['n'] - a.get('n_ok', a['n'])} failed outputs kept with reasons")

    L.append("\n## Research questions\n")
    dev = stages.get("dev")
    conf = stages.get("positive_test")
    def shift(r, arm):
        a = r["per_arm"].get(arm)
        return a["paired_shift_vs_base"] if a else None
    for i, q in enumerate(RQ, 1):
        L.append(f"\n**RQ{i}. {q}**\n")
        if i == 1:
            if dev:
                s = shift(dev, "llm_global"); b = shift(dev, "budget_control")
                L.append(f"Development (exploratory): llm_global paired shift {_fmt(s['mean'])} [{_fmt(s['ci95'][0])}, {_fmt(s['ci95'][1])}] "
                         f"vs budget_control {_fmt(b['mean'])} [{_fmt(b['ci95'][0])}, {_fmt(b['ci95'][1])}]; z-AUROC "
                         f"{_fmt(dev['per_arm_family']['llm_global']['all']['auroc_zscored']['auroc'], 3)}. Quality: model-judge {'present' if 'dev' in judge else 'MISSING'}; human review pending.")
            if conf:
                L.append("Confirmation: see the frozen table above.")
            if not dev and not conf:
                L.append("MISSING: no development or confirmation run.")
        elif i == 2:
            for label, r in (("dev", dev), ("confirmation", conf)):
                if r and "llm_global" in r["specificity"]:
                    s = r["specificity"]["llm_global"]
                    d = s["score_minus_wrong_key_mean"]
                    L.append(f"{label}: correct-minus-wrong-key {_fmt(d['mean'])} [{_fmt(d['ci95'][0])}, {_fmt(d['ci95'][1])}]; top-1 within family "
                             f"{s['top1_within_family_stratum']:.3f}; leakage fraction {_fmt(r['per_arm']['llm_global']['leakage']['any_term_frac'], 3)}; "
                             f"score-length correlation {_fmt(r['score_length_correlation'].get('llm_global'), 3)}.")
            if not dev and not conf:
                L.append("MISSING.")
        elif i == 3:
            if dev:
                g, l = shift(dev, "llm_global"), shift(dev, "llm_local")
                L.append(f"Development: llm_global {_fmt(g['mean'])} vs llm_local {_fmt(l['mean'])} paired shift; z-AUROC "
                         f"{_fmt(dev['per_arm_family']['llm_global']['all']['auroc_zscored']['auroc'], 3)} vs "
                         f"{_fmt(dev['per_arm_family']['llm_local']['all']['auroc_zscored']['auroc'], 3)}.")
            else:
                L.append("MISSING.")
        elif i == 4:
            L.append("See Robustness section." if att else "MISSING: attacks not run.")
        elif i == 5:
            if dev:
                L.append("Development per-family z-AUROC for llm_global: " + "; ".join(
                    f"{f}={_fmt(v['auroc_zscored']['auroc'], 3)}" for f, v in dev["per_arm_family"]["llm_global"].items() if f != "all")
                    + " (4 development keys per family: preliminary).")
            else:
                L.append("MISSING.")
        elif i == 6:
            names = [n for n in stages if n.startswith("ablation")]
            if dev:
                L.append("Length views (dev, llm_global): " + ", ".join(f"{k}: {_fmt(v['score_mean'])} (n={v['n']})" for k, v in dev["length_views"]["llm_global"].items()))
                L.append("Cost per output (dev): " + "; ".join(f"{a}: scoring {_fmt(v['cost'].get('scoring_tokens'), 0)} tok, proposal "
                                                            f"{_fmt(v['cost'].get('proposal_tokens'), 0)} tok, {_fmt(v['cost'].get('wall_seconds'), 1)} s" for a, v in dev["per_arm"].items()))
            L.append(f"Ablations: {', '.join(names) if names else 'MISSING (not run)'}.")
    L.append("\n## Figures\n")
    for p in sorted(fig_dir.glob("*.png")):
        L.append(f"- `{p}`")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(out_dir / "results.json", {"stages": {k: {kk: vv for kk, vv in v.items() if kk not in ("confusion",)} for k, v in stages.items()},
                                          "selftest": selftest, "fit_scorer": fit_diag, "ifeval": ifeval, "protocol": proto,
                                          "calibration_id": calibration["calibration_id"] if calibration else None, "judge": judge,
                                          "model": cfg["models"]["shared"], "config_hash": cfg.content_hash()})
    log(f"[report] wrote {out_dir}/report.md and results.json")
    return out_dir
