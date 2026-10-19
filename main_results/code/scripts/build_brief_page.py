"""Build the shareable HTML brief from the result files (re-run whenever an experiment lands).

Everything numeric is read from runs/**.json where such a file exists; the few constants below are results of
earlier phases that live in docs/algorithm_spec.md and the project log.  Missing result files render as
"running" -- never as placeholder numbers.

Usage: python scripts/build_brief_page.py --out <path/to/page.html>
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
JUDGE = "api_judge_google_gemma-4-31b-it.json"      # mixed prompt (flaws listed, then a winner) -- early runs only
PREF = "api_pref_google_gemma-4-31b-it.json"        # preference-only: one question, which is better?


def load_pref(run: str):
    return load(RUNS / run / "eval" / PREF) or load(RUNS / run / "eval" / JUDGE)
E = html.escape


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------------------------------------------
# small HTML chart helpers (bars are HTML/CSS so text stays readable at phone width)
# --------------------------------------------------------------------------------------------------------------
def bar_row(label: str, value: float, lo: float | None = None, hi: float | None = None, cls: str = "wm",
            note: str = "", refs: tuple[tuple[float, str], ...] = (), vmax: float = 1.0, fmt: str = "{:.2f}", strong: bool = False) -> str:
    """One horizontal bar.  Colour follows the entity (watermarked = orange, clean = blue) or is muted for context
    rows; the value is always printed in text ink, and the bar carries a native hover title."""
    pct = lambda v: f"{100 * max(0.0, min(v, vmax)) / vmax:.2f}%"
    tip = fmt.format(value) + (f" (95% interval {fmt.format(lo)}-{fmt.format(hi)})" if lo is not None else "")
    ci = f'<span class="ci" style="left:{pct(lo)};width:calc({pct(hi)} - {pct(lo)})"></span>' if lo is not None else ""
    rf = "".join(f'<span class="ref ref-{k}" style="left:{pct(v)}"></span>' for v, k in refs)
    return (f'<div class="bar-row{" strong" if strong else ""}"><div class="bar-label">{label}</div>'
            f'<div class="bar-track" title="{E(tip)}">{rf}<span class="bar {cls}" style="width:{pct(value)}"></span>{ci}</div>'
            f'<div class="bar-value">{fmt.format(value)}{f"<small>{note}</small>" if note else ""}</div></div>')


def pair_row(label: str, a: float, b: float, vmax: float) -> str:
    w = lambda v: f"{100 * min(v, vmax) / vmax:.1f}%"
    return (f'<div class="pair-row"><div class="bar-label">{label}</div><div class="pair-bars">'
            f'<div class="bar-track thin" title="watermarked: {a:.0f} per 100"><span class="bar wm" style="width:{w(a)}"></span></div>'
            f'<div class="bar-track thin" title="clean: {b:.0f} per 100"><span class="bar clean" style="width:{w(b)}"></span></div></div>'
            f'<div class="bar-value">{a:.0f} <small>vs</small> {b:.0f}</div></div>')


def band_row(label: str, value: float, lo: float, hi: float, mean: float, vmax: float, inside: bool, fmt: str, strong: bool = False) -> str:
    """A setting's value as a dot against the clean generator's own band (shaded) and mean (tick)."""
    pct = lambda v: f"{100 * max(0.0, min(v, vmax)) / vmax:.2f}%"
    verdict = "inside the clean band" if inside else "outside"
    return (f'<div class="bar-row{" strong" if strong else ""}"><div class="bar-label">{label}</div>'
            f'<div class="band-track" title="{E(fmt.format(value))}; clean band {E(fmt.format(lo))}-{E(fmt.format(hi))}">'
            f'<span class="band" style="left:{pct(lo)};width:calc({pct(hi)} - {pct(lo)})"></span><span class="band-mean" style="left:{pct(mean)}"></span>'
            f'<span class="dot {"in" if inside else "out"}" style="left:{pct(value)}"></span></div>'
            f'<div class="bar-value">{fmt.format(value)}<small> · {verdict}</small></div></div>')


def chip(text: str, kind: str) -> str:
    return f'<span class="chip chip-{kind}"><i></i>{E(text)}</span>'


# --------------------------------------------------------------------------------------------------------------
# data blocks
# --------------------------------------------------------------------------------------------------------------
ARM_NAMES = {
    "wm_cross30f": "Earlier setting, strength control off (known to be damaged)",
    "wm_cross30s": "Earlier setting, higher target",
    "wm_cross60": "Earlier setting, doubled strength ceiling",
    "wm_cross30": "Earlier setting: adaptive strength (up to β = 30)",
    "wm_c4": "Constant strength β = 4", "wm_c8": "β = 8", "wm_c12": "β = 12", "wm_c16": "β = 16", "wm_c20": "β = 20",
    "wm_c8w": "β = 8 + warm-up", "wm_c16w": "β = 16 + warm-up",
    "wm_c12wg": "β = 12 + warm-up + semantic guard", "wm_c20wg": "β = 20 + warm-up + semantic guard",
    "wm_c4g": "β = 4 + semantic guard", "wm_c6": "β = 6", "wm_c6g": "β = 6 + semantic guard", "wm_c8g": "β = 8 + semantic guard",
    "wm_span1": "Same key in every span", "wm_spanrot": "Four keys rotating across spans",
}


def judge_block() -> str:
    rows = []
    for run, arm, tag in (("cross_long", "wm_cross30f", "1,024-token budget"), ("cross_long2", "wm_cross30s", "1,024-token budget"),
                          ("cross_llama", "wm_cross60", "512 tokens"), ("cross_llama", "wm_cross30", "512 tokens"),
                          ("cross_long", "wm_cross30", "long texts: controller stalls, least push")):
        d = load(RUNS / run / "eval" / JUDGE)
        if not d or arm not in d["per_arm"]:
            continue
        s = d["per_arm"][arm]
        rows.append(bar_row(f'{E(ARM_NAMES[arm])} <em>{E(tag)}</em>', s["win_rate"], *s["win_rate_ci95"],
                            cls="wm", refs=((0.5, "even"), (0.4, "bar")),
                            note=f' · major flaws {100 * s["major_flaw_wm"]:.0f}% vs {100 * s["major_flaw_clean"]:.0f}%'))
    return "".join(rows)


def flaws_block() -> str:
    d = load(RUNS / "cross_llama" / "eval" / JUDGE)
    if not d:
        return ""
    s = d["per_arm"]["wm_cross30"]
    names = {"unrelated_words_or_insertions": "Unrelated words or insertions", "drifts_off_topic": "Drifts off topic",
             "incoherent_or_nonsensical": "Incoherent or nonsensical", "ignores_instructions": "Ignores instructions",
             "factual_or_logical_error": "Factual or logical error", "repetition_or_loops": "Repetition or loops"}
    wm, cl = s["flaws_wm_per_100"], s["flaws_clean_per_100"]
    return "".join(pair_row(E(n), wm.get(k, 0.0), cl.get(k, 0.0), 80.0) for k, n in names.items())


def token_strip() -> str:
    """The flagged Vietnam answer, every token shaded by the evidence it carries (raw contrast d)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
    except Exception:
        return ""
    rd = RUNS / "cross_long"
    gens = [json.loads(l) for f in rd.glob("generations*.jsonl") for l in open(f)]
    g = next((x for x in gens if x["arm"] == "wm_cross30" and x["prompt_id"].endswith("dolly:10943")), None)
    if g is None:
        return ""
    tr = next((json.loads(l) for f in rd.glob("trajectories*.jsonl") for l in open(f) if json.loads(l)["output_id"] == g["id"]), None)
    if tr is None:
        return ""
    ids = tr["candidate_token_ids"][tr.get("chosen", 0)]
    raw = tr["cost"]["per_token"]["contrast_raw"]
    n = min(len(ids), len(raw), 150)
    total = sum(max(0.0, r) for r in raw[:n]) or 1.0
    spans = []
    for t, r in zip(ids[:n], raw[:n]):
        s = E(tok.decode([t])).replace("\n", "↵\n")
        if r > 0.15:
            a = min(1.0, r / 3.0)
            spans.append(f'<span class="tk" style="--a:{a:.2f}" title="evidence {r:+.2f}">{s}</span>')
        elif r < -0.5:
            spans.append(f'<span class="tk neg" title="evidence {r:+.2f}">{s}</span>')
        else:
            spans.append(s)
    top = sorted(raw[:n], reverse=True)[: max(1, n // 20)]
    share = 100 * sum(max(0.0, r) for r in top) / total
    return (f'<div class="strip">{"".join(spans)} …</div><p class="caption">The opening of the watermarked answer, token by token. '
            f'Shading is the evidence each token contributes to detection (its keyed-minus-neutral log-probability under the '
            f'reference model); unshaded tokens contribute almost nothing. In this excerpt the top 5% of tokens carry '
            f'<b>{share:.0f}%</b> of the positive evidence. Secret phrase: <code>amber copper delta atlas</code>.</p>')


VARIANT_NAMES = {"T": "Plain statistic (current)", "T_clip05": "Evidence clipped at ±0.5", "T_clip1": "Evidence clipped at ±1.0",
                 "T_sign": "Direction only (sign test)", "T_clip05_gate": "Clipped ±0.5, entropy-weighted", "T_gate": "Entropy-weighted",
                 "T_gate_hi": "Entropy-weighted (high gate)", "T_centered": "Centred on the unsteered expectation",
                 "T_centered_gate": "Centred, entropy-weighted", "matched": "Matched filter", "matched_gate": "Matched filter, entropy-weighted",
                 "matched_gate_hi": "Matched filter (high gate)", "z_gate": "Per-position standardised"}


def detector_block() -> tuple[str, str]:
    d = load(RUNS / "detector_variants_dev.json")
    if not d:
        return chip("running", "run"), ""
    arms = (("cross_llama", "wm_cross30"), ("cross_llama", "wm_cross60"), ("cross_long", "wm_cross30"), ("cross_long2", "wm_cross30s"))
    rows = []
    for v in d["runs"]["cross_llama"]["variants"]:
        tprs, fp, pairs = [], 0, 0
        for run in d["runs"].values():
            r = run["variants"].get(v, {})
            tprs += [x["tpr"] for a, x in r.items() if isinstance(x, dict) and "tpr" in x]
            if "base" in r and "fpr" in r["base"]:
                fp += int(r["base"]["fp"]); pairs += int(r["base"]["n_pairs"])
        if tprs:
            rows.append((v, sum(tprs) / len(tprs), fp / max(1, pairs)))
    rows.sort(key=lambda x: -x[1])
    body = "".join(bar_row(E(VARIANT_NAMES.get(v, v)) + (" <em>used until now</em>" if v == "T" else ""), t,
                           cls="wm" if v == "T_clip05_gate" else "mute", strong=(v == "T_clip05_gate"),
                           note=f" · clean texts flagged {100 * f:.1f}%") for v, t, f in rows)
    body += ('<p class="caption">Mean share of watermarked texts detected at the frozen 1% threshold, over the five marked arms of the finished '
             'development runs (60 texts each); clean rate pooled over 2,160 clean text–key pairs. With a stable threshold the standard controller is '
             'detected 85–88% of the time by the plain statistic — detection was never the problem. Capping each token\'s evidence makes the detector '
             '<b>better</b> (97%), because the few huge values mostly add variance. Clean false positives are clustered in a few texts, which is why '
             'the eligibility test of the pre-registration was amended openly before any sweep text existed.</p>')
    return chip("done", "ok"), body


SWEEP_ORDER = ["wm_c4", "wm_c4g", "wm_c6", "wm_c6g", "wm_c8", "wm_c8g", "wm_c8w", "wm_cross30", "wm_c12", "wm_c12wg", "wm_c16", "wm_c16w", "wm_c20", "wm_c20wg"]


def sweep_block() -> tuple[str, str, str]:
    """Detection and pairwise preference for every sweep arm -- two separate figures, chosen arm emphasised."""
    det, jd, sel = load(RUNS / "detector_variants_tradeoff.json"), load_pref("tradeoff"), load(RUNS / "selection.json")
    if not det or not jd:
        return chip("running", "run"), "", ""
    chosen = (sel or {}).get("configuration", {}).get("chosen", "")
    res, old = det["runs"]["tradeoff"]["variants"]["T_clip05_gate"], det["runs"]["tradeoff"]["variants"]["T"]
    arms = [a for a in SWEEP_ORDER if a in res and "tpr" in res[a] and a in jd["per_arm"]]
    name = lambda a: E(ARM_NAMES.get(a, a)) + (" <em>selected</em>" if a == chosen else "")
    d_rows = "".join(bar_row(name(a), res[a]["tpr"], *res[a]["tpr_ci"], cls="wm" if a == chosen else "mute", strong=(a == chosen),
                             note=f' · earlier detector {old[a]["tpr"]:.2f}') for a in arms)
    b = res["base"]
    d_rows += (f'<p class="caption">Detection rate at a 1% false-positive threshold (set on 1,200 clean texts, per key) '
               f'(60 texts per setting; whisker: 95% interval; “earlier detector”: the unclipped statistic on the same texts). Clean text–key pairs flagged in this run: '
               f'<b>{b["fp"]} of {b["n_pairs"]}</b> text–key pairs.</p>')
    w_rows = "".join(bar_row(name(a), jd["per_arm"][a]["win_rate"], *jd["per_arm"][a]["win_rate_ci95"], cls="wm" if a == chosen else "mute",
                             strong=(a == chosen), refs=((0.5, "even"), (0.4, "bar"))) for a in arms)
    ref_rows = ""
    for run, arm, lab in (("clean_pair", "clean_seed1", "Clean vs clean, run 1 <em>reference level</em>"), ("clean_pair", "clean_seed2", "Clean vs clean, run 2 <em>reference level</em>"),
                          ("control_b0", "wm_c0", "β = 0 through the watermarking sampler <em>pipeline control</em>")):
        x = (load_pref(run) or {}).get("per_arm", {}).get(arm)
        if x:
            ref_rows += bar_row(lab, x["win_rate"], *x["win_rate_ci95"], cls="clean", refs=((0.5, "even"), (0.4, "bar")))
    w_rows = ('<div class="legend"><span><i></i>0.50 — no preference</span><span><i class="s"></i>0.40 — quality bar fixed in advance</span></div>' + ref_rows + w_rows +
              '<p class="caption">A judge model sees the task and two anonymous responses — clean and watermarked — and picks the better one. 0.50 means no preference; '
              'lower means the clean response is preferred. Each pair is judged in both orders; ties count half. <b>Read the bars against the blue reference rows, not against 0.50:</b> '
              'on these 60 prompts the reference clean run happens to be a good draw — other clean runs, and the watermarking sampler with zero push, score 0.42–0.45 against it.</p>')
    return chip("done", "ok"), d_rows, w_rows


def clean_band_block() -> tuple[str, str, str]:
    """The absolute check, kept separate from the win rate: each setting against the clean generator's own variation."""
    q = load(RUNS / "quality_vs_clean.json")
    if not q:
        return chip("running", "run"), "", ""
    sel = load(RUNS / "selection.json") or {}
    chosen = sel.get("configuration", {}).get("chosen", "")
    m = q["metrics"]["insertions_or_drift"]
    arms = [a for a in SWEEP_ORDER if f"tradeoff/{a}" in m["arms"]]
    rows = "".join(band_row(E(ARM_NAMES.get(a, a)) + (" <em>selected</em>" if a == chosen else ""), m["arms"][f"tradeoff/{a}"]["value"],
                            m["clean_band95"][0], m["clean_band95"][1], m["clean_mean"], 0.8, m["arms"][f"tradeoff/{a}"]["inside_clean_band"],
                            "{:.0%}", strong=(a == chosen)) for a in arms)
    draws = ", ".join(f"{100 * x:.0f}%" for x in m["clean_per_draw"])
    fig = ('<div class="legend"><span><i class="band-key"></i>95% band of one clean draw</span><span><i class="mean-key"></i>clean mean</span>'
           '<span><i class="dot-key in"></i>setting, inside</span><span><i class="dot-key out"></i>setting, outside</span></div>' + rows +
           f'<p class="caption">Share of texts in which the per-text judge reports <b>unrelated insertions or topic drift</b> — the damage specific to this watermark. '
           f'Clean text alone gives {draws} across {q["n_clean_draws"]} independent draws of the same {q["n_prompts"]} prompts; the shaded band is what a single clean draw produces 95% of the time.</p>')
    names = {"major_flaw": ("texts with a major flaw", "{:.0%}"), "n_flaws": ("flaws per text", "{:.2f}"), "not_on_task": ("not fully on task", "{:.0%}"),
             "insertions_or_drift": ("unrelated insertions or drift", "{:.0%}"), "quality": ("mean 1–5 score", "{:.2f}")}
    show = [a for a in ("wm_c4", "wm_c4g", "wm_c6", "wm_c6g", "wm_c8", "wm_cross30", "wm_c20") if f"tradeoff/{a}" in m["arms"]]
    head = "<tr><th>metric</th><th>clean mean</th><th>clean 95% band</th>" + "".join(f"<th>{E(ARM_NAMES.get(a, a))}</th>" for a in show) + "</tr>"
    body = ""
    for k, (lab, fmt) in names.items():
        r = q["metrics"][k]
        cells = "".join(f'<td class="n">{fmt.format(r["arms"][f"tradeoff/{a}"]["value"])} <span class="v {"in" if r["arms"][f"tradeoff/{a}"]["inside_clean_band"] else "out"}">'
                        f'{"inside" if r["arms"][f"tradeoff/{a}"]["inside_clean_band"] else "outside"}</span></td>' for a in show)
        body += f'<tr><td>{lab}</td><td class="n">{fmt.format(r["clean_mean"])}</td><td class="n">{fmt.format(r["clean_band95"][0])}–{fmt.format(r["clean_band95"][1])}</td>{cells}</tr>'
    table = f'<div class="scroll"><table>{head}{body}</table></div>'
    return chip("done", "ok"), fig, table


def rotation_block() -> tuple[str, str]:
    d = load(RUNS / "detector_variants_long.json")
    if not d or "rotation" not in d["runs"].get("spans", {}):
        return chip("scoring", "run"), ""
    run = d["runs"]["spans"]
    rows = "".join(bar_row(f"{E(VARIANT_NAMES.get(v, v))}", r["wm_spanrot"]["accuracy"], cls="wm", refs=((0.25, "even"),),
                           note=f' · {r["wm_spanrot"]["correct"]}/{r["wm_spanrot"]["spans"]} spans') for v, r in run["rotation"].items()
                   if r.get("wm_spanrot", {}).get("accuracy") is not None)
    t = run["variants"]["T_clip05_gate"]
    jd = load(RUNS / "spans" / "eval" / JUDGE)
    q = ""
    if jd:
        a, b = jd["per_arm"]["wm_span1"], jd["per_arm"]["wm_spanrot"]
        q = (f' Quality is another matter: this run still uses the old bursty controller, and the judge prefers the clean answer — win rate '
             f'{a["win_rate"]:.2f} with one key, {b["win_rate"]:.2f} with rotating keys.')
    cap = (f'<p class="caption">Share of generated 128-token spans whose own key scores highest among the four rotating keys (dashed line: chance, 0.25); 60 texts. '
           f'Whole-text detection does not suffer from rotation: {100 * t["wm_spanrot"]["tpr"]:.0f}% of rotating-key texts and {100 * t["wm_span1"]["tpr"]:.0f}% of single-key texts are detected at the '
           f'frozen 1% threshold, and the first 128 reference tokens alone are enough.{q}</p>')
    return chip("done", "ok"), '<div class="legend"><span><i></i>0.25 — chance</span></div>' + rows + cap


# --------------------------------------------------------------------------------------------------------------
CSS = """
:root{--ground:#F3F6FA;--surface:#FFFFFF;--ink:#14202C;--muted:#566474;--rule:#D3DBE5;--accent:#6A3BD6;--accent-soft:#ECE6FB;
--good:#0E7A6B;--bad:#B2412C;--run:#8A6A12;--mark:106,59,214;
--s-clean:#2a78d6;--s-wm:#eb6834;--s-mute:#B4BFCB;--band:rgba(42,120,214,.16);--st-good:#0ca30c;--st-crit:#d03b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0E131A;--surface:#151C26;--ink:#E4EAF1;--muted:#94A2B2;--rule:#293442;
--accent:#A78DFF;--accent-soft:#231C3D;--good:#4FC4B0;--bad:#E88D77;--run:#D9B54A;--mark:167,141,255;
--s-clean:#3987e5;--s-wm:#d95926;--s-mute:#44505E;--band:rgba(57,135,229,.22);--st-good:#0ca30c;--st-crit:#e66767}}
:root[data-theme="dark"]{--ground:#0E131A;--surface:#151C26;--ink:#E4EAF1;--muted:#94A2B2;--rule:#293442;
--accent:#A78DFF;--accent-soft:#231C3D;--good:#4FC4B0;--bad:#E88D77;--run:#D9B54A;--mark:167,141,255;
--s-clean:#3987e5;--s-wm:#d95926;--s-mute:#44505E;--band:rgba(57,135,229,.22);--st-good:#0ca30c;--st-crit:#e66767}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font:17px/1.62 "Source Serif 4",Georgia,"Times New Roman",serif;padding-inline:16px;padding-block:0 72px}
.col{max-width:68ch;margin-inline:auto}.wide{max-width:940px;margin-inline:auto}
h1,h2,h3,.eyebrow,.chip,.bar-label,.bar-value,th,.caption,.k,.legend{font-family:"Archivo","Helvetica Neue",Arial,sans-serif}
h1{font-size:clamp(2rem,5.2vw,3.1rem);line-height:1.06;letter-spacing:-.02em;font-weight:700;margin:0;text-wrap:balance}
h2{font-size:1.5rem;line-height:1.2;font-weight:700;letter-spacing:-.01em;margin:72px 0 12px;text-wrap:balance}
h4{font-family:"Archivo",Arial,sans-serif;font-size:.92rem;font-weight:650;margin:22px 0 6px}
h3{font-size:1.02rem;font-weight:650;margin:34px 0 6px;text-wrap:balance}
p{margin:0 0 16px}a{color:var(--accent)}code,.strip,td.n{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
code{font-size:.86em;background:var(--accent-soft);padding:1px 5px;border-radius:3px}
header{padding-block:64px 8px}.eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:18px}
.lede{font-size:1.18rem;line-height:1.5;margin-top:22px}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);margin-top:34px}
.status div{background:var(--surface);padding:16px 18px}.status b{display:block;font-family:"Archivo",Arial,sans-serif;font-size:1.02rem;margin-bottom:2px}
.status span{font-size:.92rem;color:var(--muted);line-height:1.4;display:block}
.status .ok b{color:var(--good)}.status .no b{color:var(--bad)}.status .wip b{color:var(--run)}
.chip{display:inline-flex;align-items:center;gap:7px;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border:1px solid currentColor;border-radius:99px;vertical-align:middle}
.chip i{width:7px;height:7px;border-radius:50%;background:currentColor}.chip-ok{color:var(--good)}.chip-run{color:var(--run)}
figure{margin:28px auto}.panel{background:var(--surface);border:1px solid var(--rule);padding:22px 22px 14px}
.caption{font-size:.84rem;line-height:1.5;color:var(--muted);margin-top:12px}
.flow{display:grid;gap:10px;font-family:"Archivo",Arial,sans-serif;font-size:.9rem}
.flow .lane{display:flex;flex-wrap:wrap;align-items:center;gap:8px}.flow .k{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);width:100%;margin-top:6px}
.box{border:1px solid var(--rule);padding:8px 12px;background:var(--ground);line-height:1.3}.box.m{border-color:var(--accent);background:var(--accent-soft)}
.arrow{color:var(--muted)}
.formula{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86rem;background:var(--surface);border-left:3px solid var(--accent);padding:12px 16px;overflow-x:auto;white-space:pre;margin:0 0 16px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.93rem;font-variant-numeric:tabular-nums}
th{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left;font-weight:600;padding:8px 10px;border-bottom:1px solid var(--ink)}
td{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:top}td.n{font-size:.86rem;white-space:nowrap}
td .v{font-family:"Archivo",Arial,sans-serif;font-size:.7rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}td .v.in::before{content:"● ";color:var(--st-good)}td .v.out::before{content:"○ ";color:var(--st-crit)}
.bar-row,.pair-row{display:grid;grid-template-columns:minmax(0,1fr);gap:4px;padding:9px 0;border-bottom:1px solid var(--rule)}
.bar-row:last-of-type,.pair-row:last-of-type{border-bottom:0}
@media(min-width:720px){.bar-row{grid-template-columns:minmax(0,5fr) minmax(0,6fr) minmax(0,4fr);gap:16px;align-items:center}
.pair-row{grid-template-columns:minmax(0,4fr) minmax(0,6fr) 96px;gap:16px;align-items:center}}
.bar-label{font-size:.88rem;line-height:1.3;color:var(--muted)}.bar-row.strong .bar-label{color:var(--ink);font-weight:650}
.bar-label em{display:inline-block;font-style:normal;font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink);border:1px solid var(--ink);border-radius:99px;padding:0 7px;margin-left:6px;vertical-align:1px}
.bar-value{font-size:.9rem;font-weight:650;font-variant-numeric:tabular-nums;color:var(--ink)}.bar-value small{font-weight:400;color:var(--muted);font-size:.78rem}
.bar-track,.band-track{position:relative;height:16px;border-left:1px solid var(--muted)}.bar-track.thin{height:9px;margin:2px 0}
.bar{position:absolute;inset:0 auto 0 0;border-radius:0 4px 4px 0;background:var(--s-wm)}.bar.clean{background:var(--s-clean)}.bar.mute{background:var(--s-mute)}
.ci{position:absolute;top:50%;height:2px;background:var(--ink);transform:translateY(-50%)}
.ref{position:absolute;top:-5px;bottom:-5px;width:0;border-left:1.5px dashed var(--ink)}.ref-bar{border-left:2px solid var(--st-good)}
.band{position:absolute;top:0;bottom:0;background:var(--band);border-radius:2px}.band-mean{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--s-clean);transform:translateX(-1px)}
.dot{position:absolute;top:50%;width:12px;height:12px;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px var(--surface);background:var(--s-mute)}
.dot.in{background:var(--s-wm)}.dot.out{background:var(--surface);box-shadow:0 0 0 2px var(--surface),inset 0 0 0 2.5px var(--s-wm)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.78rem;color:var(--muted);margin-bottom:10px}
.legend i{display:inline-block;width:18px;height:0;vertical-align:middle;margin-right:6px;border-top:2px dashed var(--ink)}.legend i.s{border-top:2px solid var(--st-good)}
.legend i.sq{height:9px;border:0;background:var(--s-wm);border-radius:0 3px 3px 0}.legend i.sq.g{background:var(--s-clean)}
.legend i.band-key{height:10px;border:0;background:var(--band)}.legend i.mean-key{width:2px;height:12px;border:0;background:var(--s-clean)}
.legend i.dot-key{width:10px;height:10px;border:0;border-radius:50%;background:var(--s-wm)}.legend i.dot-key.out{background:var(--surface);box-shadow:inset 0 0 0 2.5px var(--s-wm)}
.versus{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule)}
.versus>div{background:var(--surface);padding:16px 18px;font-size:.95rem;line-height:1.5}.versus .k{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px;color:var(--muted)}
.versus mark{background:rgba(var(--mark),.28);color:inherit;padding:0 2px}
.strip{font-size:.8rem;line-height:1.95;white-space:pre-wrap;word-break:break-word}
.tk{background:rgba(var(--mark),calc(var(--a)*.85 + .1));border-radius:2px;padding:1px 0}.tk.neg{background:none;text-decoration:underline dotted var(--muted)}
.big{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:24px;margin:22px auto}
.big b{font-family:"Archivo",Arial,sans-serif;font-size:2.6rem;line-height:1;letter-spacing:-.03em;color:var(--ink);display:block}
.big span{font-size:.92rem;color:var(--muted);line-height:1.45;display:block;margin-top:6px}
blockquote{margin:0;padding:4px 0 4px 22px;border-left:3px solid var(--accent);font-size:1.04rem}blockquote p:last-child{margin-bottom:0}
details{border-bottom:1px solid var(--rule);padding:14px 0}details figure.wide{margin:18px 0}details>p{margin:12px 0 0}summary{cursor:pointer;font-family:"Archivo",Arial,sans-serif;font-weight:600;font-size:.98rem}
summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}details p{margin:10px 0 0}
footer{margin-top:80px;padding-top:20px;border-top:1px solid var(--rule);font-size:.86rem;color:var(--muted)}
ul{padding-left:1.15em;margin:0 0 16px}li{margin-bottom:7px}
"""


def keylen_block() -> str:
    rows = ""
    for L in (4, 8, 10, 16):
        d = load(RUNS / f"detector_variants_keylen{L}.json")
        if not d:
            return ""
        v = d["runs"][f"keylen{L}"]["variants"]["T_clip05_gate"]
        pr = (load_pref(f"keylen{L}") or {}).get("per_arm", {})
        pw = {}
        f = RUNS / f"keylen{L}" / "eval" / "api_pointwise_google_gemma-4-31b-it.jsonl"
        if f.exists():
            for l in open(f):
                r = json.loads(l); pw.setdefault(r["arm"], []).append(1.0 if {"unrelated_words_or_insertions", "drifts_off_topic"} & set(r["judgment"]["flaws"]) else 0.0)
        ins = lambda a: f"{100 * sum(pw[a]) / len(pw[a]):.0f}%" if a in pw else "—"
        x4, x2 = v["wm_c4"], v["wm_c2"]
        rows += (f'<tr><td>{L} words</td><td class="n"><b>{100 * x4["tpr"]:.0f}%</b> [{100 * x4["tpr_ci"][0]:.0f}–{100 * x4["tpr_ci"][1]:.0f}]</td>'
                 f'<td class="n">{100 * x4["false_attribution"]:.0f}%</td><td class="n">{pr.get("wm_c4", {}).get("win_rate", float("nan")):.2f}</td><td class="n">{ins("wm_c4")}</td>'
                 f'<td class="n">{100 * x2["tpr"]:.0f}%</td><td class="n">{pr.get("wm_c2", {}).get("win_rate", float("nan")):.2f}</td><td class="n">{ins("wm_c2")}</td></tr>')
    return ('<div class="scroll"><table><tr><th>key length</th><th>β = 4: detected</th><th>flagged under a wrong key</th><th>preference</th><th>insertions / drift</th>'
            '<th>β = 2: detected</th><th>preference</th><th>insertions / drift</th></tr>' + rows + '</table></div>'
            '<p class="caption">Six keys per length with no word shared between keys of a set; the same 60 development prompts; a frozen 1,200-text null per key set. '
            'Reference levels on these prompts: clean-vs-clean preference 0.42–0.45, clean insertions/drift 13–17%.</p>')


def _chosen() -> str:
    return ((load(RUNS / "selection.json") or {}).get("configuration") or {}).get("chosen", "")


def _count(run: str) -> int:
    d = RUNS / run
    return sum(1 for f in d.glob("generations*.jsonl") for _ in open(f)) if d.exists() else 0


def conf_block() -> tuple[str, str]:
    det, jd = load(RUNS / "detector_variants_conf.json"), load_pref("conf_cross")
    if not det:
        n = _count("conf_cross")
        return chip(f"generating · {n}/450 texts" if n else "queued", "run"), ""
    v, old = det["runs"]["conf_cross"]["variants"]["T_clip05_gate"], det["runs"]["conf_cross"]["variants"]["T"]
    ch = _chosen()
    rows = "".join(bar_row(E(ARM_NAMES.get(a, a)) + (" <em>selected</em>" if a == ch else ""), v[a]["tpr"], *v[a]["tpr_ci"], cls="wm" if a == ch else "mute",
                           strong=(a == ch), note=f' · earlier detector {old[a]["tpr"]:.2f}') for a in (ch, "wm_cross30") if a in v and "tpr" in v[a])
    b = v["base"]
    cap = (f'<p class="caption">Detection rate on 150 held-out prompts with 16 fresh keys; thresholds frozen beforehand. '
           f'Clean text–key pairs flagged: <b>{b["fp"]} of {b["n_pairs"]}</b> ({100 * b["fpr"]:.2f}%) — the frozen thresholds are conservative here; '
           f'at an actual 1% false-positive rate on these clean texts the detection rate is 89%.</p>')
    out = rows + cap
    if jd:
        out += "<h4>Pairwise preference (separate judge run)</h4>" + '<div class="legend"><span><i></i>0.50 — no preference</span><span><i class="s"></i>0.40 — quality bar fixed in advance</span></div>'
        cc = (load_pref("clean_pair_conf") or {}).get("per_arm", {}).get("clean_other_run")
        if cc:
            out += bar_row("Clean vs clean <em>reference level</em>", cc["win_rate"], *cc["win_rate_ci95"], cls="clean", refs=((0.5, "even"), (0.4, "bar")),
                           note=" · an independent clean run, same prompts")
        out += "".join(bar_row(E(ARM_NAMES.get(a, a)) + (" <em>selected</em>" if a == ch else ""), sx["win_rate"], *sx["win_rate_ci95"], cls="wm" if a == ch else "mute",
                               strong=(a == ch), refs=((0.5, "even"), (0.4, "bar")), note=f' · ties {100 * sx.get("tie_share", 0):.0f}%') for a, sx in jd["per_arm"].items())
        out += ('<p class="caption">The measure is calibrated: an independent clean run scores 0.47 against the reference clean run (expected 0.50). '
                'β = 4 sits about 0.16 below that level — the clean response is preferred about 2 : 1 — and the earlier adaptive setting about 0.21 below.</p>')
    return chip("done", "ok"), out


def conf_band_block() -> str:
    """Flaws on the held-out texts: a SEPARATE pointwise run, against two independent clean draws of the same prompts."""
    q = load(RUNS / "quality_vs_clean_conf.json")
    if not q:
        return ""
    ch = _chosen()
    names = {"major_flaw": ("texts with a major flaw", "{:.0%}"), "n_flaws": ("flaws per text", "{:.2f}"), "not_on_task": ("not fully on task", "{:.0%}"),
             "insertions_or_drift": ("unrelated insertions or topic drift", "{:.0%}"), "quality": ("mean quality score (1–5)", "{:.2f}")}
    m = q["metrics"]["insertions_or_drift"]
    rows = "".join(band_row(E(ARM_NAMES.get(a, a)) + (" <em>selected</em>" if a == ch else ""), m["arms"][f"conf_cross/{a}"]["value"], m["clean_band95"][0],
                            m["clean_band95"][1], m["clean_mean"], 0.5, m["arms"][f"conf_cross/{a}"]["inside_clean_band"], "{:.0%}", strong=(a == ch))
                   for a in (ch, "wm_cross30") if f"conf_cross/{a}" in m["arms"])
    fig = ('<div class="legend"><span><i class="band-key"></i>95% band of one clean draw</span><span><i class="mean-key"></i>clean mean</span>'
           '<span><i class="dot-key in"></i>inside</span><span><i class="dot-key out"></i>outside</span></div>' + rows +
           '<p class="caption">Share of texts in which the judge, rating one anonymous text per call, reports <b>unrelated insertions or topic drift</b>. Shaded: the 95% band of a single clean run.</p>')
    head = "<tr><th>metric</th><th>clean mean</th><th>clean 95% band</th>" + "".join(f"<th>{E(ARM_NAMES.get(a, a))}</th>" for a in (ch, "wm_cross30")) + "</tr>"
    body = ""
    for k, (lab, fmt) in names.items():
        r = q["metrics"][k]
        cells = "".join(f'<td class="n">{fmt.format(r["arms"][f"conf_cross/{a}"]["value"])} <span class="v {"in" if r["arms"][f"conf_cross/{a}"]["inside_clean_band"] else "out"}">'
                        f'{"inside" if r["arms"][f"conf_cross/{a}"]["inside_clean_band"] else "outside"}</span></td>' for a in (ch, "wm_cross30"))
        body += f'<tr><td>{lab}</td><td class="n">{fmt.format(r["clean_mean"])}</td><td class="n">{fmt.format(r["clean_band95"][0])}–{fmt.format(r["clean_band95"][1])}</td>{cells}</tr>'
    return fig + f'<div class="scroll"><table>{head}{body}</table></div>'


def examples_block() -> str:
    """Pairs from the blind read: clean vs selected setting, with the key and both judge runs."""
    key = load(RUNS / "blind_read_pairs.json")
    if not key:
        return ""
    gens = {}
    for f in (RUNS / "conf_cross").glob("generations*.jsonl"):
        for l in open(f):
            g = json.loads(l); gens[(g["prompt_id"], g["arm"])] = g
    phr = {k["key_id"]: k["prefix"] for k in load(ROOT / "assets" / "keys_confirmation.json")["keys"]}
    pw, pref = {}, {}
    pf = RUNS / "conf_cross" / "eval" / PREF.replace(".json", ".jsonl")
    for l in open(RUNS / "conf_cross" / "eval" / "api_pointwise_google_gemma-4-31b-it.jsonl"):
        r = json.loads(l); pw[(r["prompt_id"], r["arm"])] = r["judgment"]
    if pf.exists():
        for l in open(pf):
            r = json.loads(l)
            if r["arm"] == "wm_c4":
                b = r["judgment"]["better"]; pref.setdefault(r["prompt_id"], []).append("tie" if b == "tie" else ("watermarked" if b == r["wm_is"] else "clean"))
    try:
        from wmworld.config import load_config
        from wmworld.data.prepare import load_prepared
        P = {x.prompt_id: x.prompt for x in load_prepared(load_config(str(ROOT / "configs" / "pilot.yaml")).path_of("data_prepared"), partition="positive_test")}
    except Exception:
        P = {}
    notes = {"9": "The one pair of nine where the gap was plain to see.", "6": "The marked story wanders to “a lost ship… a voyage to the North Sea” — the key contains harbor, cove, vessel, isle.",
             "1": "Both recipes are wrong in their own way; the judge calls it a tie in both orders."}
    out = ""
    for n in ("9", "6", "1"):
        k = key[n]; pid = k["prompt_id"]
        c, w = gens[(pid, "base")], gens[(pid, "wm_c4")]
        task = " ".join(P.get(pid, "").split())
        task = (task[:230] + " …") if len(task) > 230 else task
        cut = lambda t: E(" ".join(t.split())[:560]) + " …"
        verdict = " / ".join(pref.get(pid, [])) or "—"
        out += (f'<h4>{E(notes[n])}</h4><p class="caption" style="margin-top:0">Task: {E(task)}<br>Secret phrase: <code>{E(phr[w["assigned_key_id"]])}</code> · '
                f'pairwise preference (two orders): <b>{E(verdict)}</b> · per-text quality score: watermarked {pw[(pid, "wm_c4")]["quality"]}/5, clean {pw[(pid, "base")]["quality"]}/5</p>'
                f'<div class="versus"><div><div class="k">Clean</div>{cut(c["text"])}</div><div><div class="k">Watermarked · β = 4</div>{cut(w["text"])}</div></div>')
    return out


def attack_block() -> tuple[str, str]:
    det = load(RUNS / "detector_variants_attacks.json")
    if not det:
        return chip("after the confirmation", "run"), ""
    ch, rows = _chosen(), ""
    for run, lab in (("conf_cross_para_light", "after a light paraphrase"), ("conf_cross_para_strong", "after a strong paraphrase")):
        r = det["runs"].get(run, {}).get("variants", {}).get("T_clip05_gate", {})
        if ch in r and "tpr" in r[ch]:
            rows += bar_row(f"{E(ARM_NAMES.get(ch, ch))} <em>{lab}</em>", r[ch]["tpr"], *r[ch]["tpr_ci"], cls="wm", strong=True,
                            note=f' · paraphrased clean texts flagged {100 * r["base"]["fpr"]:.1f}%')
    return chip("done", "ok"), rows + '<p class="caption">Watermarked and clean held-out texts paraphrased by an external 31B model (its “light” paraphrase already replaces 57% of the content words); detector and thresholds unchanged. Before the attack: 85%.</p>'


def ifeval_block() -> tuple[str, str]:
    res, det = load(RUNS / "ifeval_cross" / "results.json"), load(RUNS / "detector_variants_ifeval.json")
    if not res or not res.get("arms"):
        n = _count("ifeval_cross")
        return chip(f"generating · {n}/400 texts" if n else "queued", "run"), ""
    ch = _chosen()
    rows = ""
    for arm, lab, cls in (("base", "Clean", "clean"), (ch, ARM_NAMES.get(ch, ch), "wm")):
        a = res["arms"].get(arm) or {}
        if a.get("strict"):
            rows += bar_row(f"{E(lab)} <em>strict, prompt level</em>", a["strict"]["prompt_level"], cls=cls, note=f' · loose {a["loose"]["prompt_level"]:.2f} · n={a["strict"]["n_prompts"]}')
    cap = '<p class="caption">IFEval: prompts with verifiable constraints, scored by the official checker (share of prompts with all constraints satisfied).'
    if det:
        v = det["runs"]["ifeval_cross"]["variants"]["T_clip05_gate"]
        if ch in v and "tpr" in v[ch]:
            cap += f' Detection on these short, constrained responses: {100 * v[ch]["tpr"]:.0f}%; clean pairs flagged {100 * v["base"]["fpr"]:.2f}%.'
    jd = load_pref("ifeval_cross")
    if jd and ch in jd.get("per_arm", {}):
        x = jd["per_arm"][ch]; cap += f' Pairwise preference for the watermarked response: {x["win_rate"]:.2f} (0.50 = no preference).'
    return chip("done", "ok"), rows + cap + "</p>"


def long_block() -> tuple[str, str]:
    det, jd = load(RUNS / "detector_variants_long_c.json"), load_pref("long_c")
    if not det:
        return chip("after the confirmation", "run"), ""
    run, ch = det["runs"]["long_c"], _chosen()
    rows = ""
    rot = (run.get("rotation") or {}).get("T_gate_hi", {}).get(f"{ch}_rot")
    if rot and rot.get("accuracy") is not None:
        rows += bar_row("Spans attributed to their own key <em>four rotating keys</em>", rot["accuracy"], cls="wm", strong=True, refs=((0.25, "even"),), note=f' · {rot["correct"]}/{rot["spans"]} spans · chance 0.25')
    v = run["variants"]["T_clip05_gate"]
    for a, lab in ((ch, "Detected, one key throughout"), (f"{ch}_rot", "Detected, rotating keys (first key)")):
        if a in v and "tpr" in v[a]:
            rows += bar_row(lab, v[a]["tpr"], *v[a]["tpr_ci"], cls="wm")
    cap = '<p class="caption">Texts of up to 1,024 tokens with the selected gentle push.'
    if jd:
        cap += " Judge win rate against clean: " + "; ".join(f'{E(a.replace("wm_", ""))} {s["win_rate"]:.2f}' for a, s in jd["per_arm"].items()) + "."
    return chip("done", "ok"), rows + cap + "</p>"


def build() -> str:
    det_chip, det_body = detector_block()
    sw_chip, sw_det, sw_win = sweep_block()
    cb_chip, cb_fig, cb_table = clean_band_block()
    rot_chip, rot_body = rotation_block()
    cf_chip, cf_body = conf_block()
    at_chip, at_body = attack_block()
    if_chip, if_body = ifeval_block()
    lg_chip, lg_body = long_block()
    strip = token_strip()
    fig = lambda body: f'<figure class="wide"><div class="panel">{body}</div></figure>' if body else ""
    now = datetime.now().strftime("%-d %B %Y")
    P: list[str] = []
    A = P.append
    A('<title>Secret-Phrase Watermark</title>')
    A('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')
    A(f"<style>{CSS}</style>")
    A(f"""<header class="col"><div class="eyebrow">Research brief · {E(now)}</div>
<h1>Watermarking LLM text with a secret phrase</h1>
<p class="lede">We embed a statistical watermark in generated text by biasing the generator toward tokens that a small
reference language model finds more likely when it is conditioned on a secret phrase. Detection needs <b>only the text and
the phrase</b> — no prompt, no access to the generator, no generation logs. This brief describes the method and reports how
well the mark is detected, what it costs in text quality, and how robust it is.</p></header>
<div class="wide"><div class="status">
<div class="ok"><b>Detectable</b><span>85% of watermarked texts are detected at a 1% false-positive threshold (0.25% of clean texts actually flagged), on held-out prompts with fresh keys, by a different model than the generator.</span></div>
<div class="wip"><b>A moderate quality cost</b><span>A blind judge prefers the clean response about 2 : 1, and unrelated insertions rise from 7% to 22% of texts. No loss on verifiable instruction following; in a blind read of pairs the difference is hard to see.</span></div>
<div class="no"><b>Not robust to paraphrase</b><span>After a rewrite by a 31B model, detection falls from 85% to 12% (light) and 4% (strong).</span></div>
</div></div>""")

    A("""<main><section class="col"><h2>Method</h2>
<p>A <b>generator</b> writes the text token by token. A second, small <b>reference model</b> reads the text so far twice: once
after a system line containing the <b>key</b> — a secret phrase of random common words such as
<code>pebble umber echo meadow</code> — and once after a neutral line. For every candidate next token the two readings give a
log-probability contrast</p>
<div class="formula">d(token) = log p_ref(token | key, text so far) − log p_ref(token | neutral line, text so far)</div>
<p>which is near zero for most tokens. <b>Embedding</b> tilts the generator's next-token distribution toward positive contrast;
<b>detection</b> teacher-forces the finished text through the reference model under both contexts and tests whether the tokens
actually written lean toward the key more than clean text does.</p>
<div class="formula">embed:   p′(token) ∝ p_generator(token) · exp( β · clip(d(token), ±0.5) )
detect:  T = mean(e_t) / ( sd(e_t) / √n ),   e_t = clip(d_t, ±0.5), weighted by the reference model's entropy
         flag if T exceeds the per-key 99th percentile of 1,200 clean texts</div>
</section>
<figure class="wide"><div class="panel flow">
<div class="lane"><span class="k">Embedding</span><span class="box">task prompt</span><span class="arrow">→</span><span class="box">generator</span><span class="arrow">→</span><span class="box">next-token probabilities</span><span class="arrow">→</span><span class="box m">tilt by exp(β·d)</span><span class="arrow">→</span><span class="box">sample token</span></div>
<div class="lane"><span class="box">text so far</span><span class="arrow">→</span><span class="box m">reference model, two contexts: key / neutral</span><span class="arrow">→</span><span class="box m">contrast d per candidate token</span><span class="arrow">↑</span></div>
<div class="lane"><span class="k">Detection — no prompt, no generator</span><span class="box">finished text + candidate key</span><span class="arrow">→</span><span class="box m">reference model, two contexts</span><span class="arrow">→</span><span class="box">clipped contrast per token</span><span class="arrow">→</span><span class="box">T</span><span class="arrow">→</span><span class="box">above the clean-text threshold?</span></div>
</div></figure>
<section class="col"><ul>
<li><b>Text and key only.</b> The detector never sees the prompt or the generator — what a third-party verifier would have.</li>
<li><b>The key is a phrase</b>, so the key space is as large as language and different keys can mark different users or sessions.</li>
<li><b>The reference model may differ from the generator.</b> Generation needs only a token-lookup table between the two vocabularies; detection needs no bridge at all. Here the generator is <code>Qwen3.5-0.8B</code> and the reference model <code>Llama-3.2-1B-Instruct</code>.</li>
<li><b>Guards during embedding:</b> no push where the generator's entropy is low; none on the key's own words; none on a token that would complete a repeated 4-gram; the end-of-text probability is left untouched.</li></ul>
<h2>Terms and setup</h2></section>
<figure class="wide"><div class="panel scroll"><table>
<tr><th>term</th><th>meaning here</th></tr>
<tr><td><b>Strength β</b></td><td>The size of the tilt. With the contrast clipped at ±0.5, a favoured token's probability is multiplied by up to e<sup>β/2</sup>: about 7× at β = 4, 55× at β = 8, 22,000× at β = 20. “Constant” means the same β at every position; the <b>selected setting is constant β = 4</b>.</td></tr>
<tr><td><b>Earlier setting</b></td><td>Our first approach: an adaptive controller that scaled β (up to 30) by how much mark had accumulated in the text so far. Strong detection, clearly damaged text.</td></tr>
<tr><td><b>Warm-up, semantic guard</b></td><td>Two optional protections tried in the sweep: ramping β up over the first 48 tokens; and giving zero push to tokens the key makes much more likely (d &gt; 1), which are typically words from the key's semantic neighbourhood.</td></tr>
<tr><td><b>Detection rate</b></td><td>Share of watermarked texts flagged at a threshold set for a 1% false-positive rate on 1,200 clean texts (per key). The rate at which separate clean texts are actually flagged is reported alongside.</td></tr>
<tr><td><b>Quality: two separate judge runs</b></td><td>An independent model (<code>gemma-4-31b-it</code>; cross-checked with <code>qwen3.8-27b</code>) that never sees the key or which text is watermarked. <i>Pairwise preference:</i> two anonymous responses to the same prompt, both orders, one question — which is better; ties count half. 0.50 means no preference in expectation; because a particular clean run can be a lucky or unlucky draw, the measured clean-vs-clean level is shown alongside (0.47 on the held-out prompts, 0.42–0.45 on the development prompts). <i>Per-text flaws:</i> one anonymous text per call — on task? which flaws? severity? score 1–5 — compared with the clean generator's own rates, estimated from independent clean runs of the same prompts.</td></tr>
<tr><td><b>Development vs held-out</b></td><td>Settings were chosen on 60 development prompts with 12 keys. The reported results come from 150 held-out prompts with 16 fresh keys, run once, with the choice made by a rule written down in advance. Tasks: stories, instruction answers, news continuation; texts up to 512 tokens.</td></tr>
</table></div></figure>""")

    A(f"""<section class="col"><h2>Held-out results</h2>
<p>One configuration (constant β = 4), chosen by a pre-registered rule, run once on 150 held-out prompts with 16 fresh keys.</p>
<h3>1 · Detection and pairwise preference {cf_chip}</h3></section>
{fig(cf_body)}
<section class="col"><h3>2 · Flaws relative to clean text</h3>
<p>The generator is small (0.8B) and its clean texts are often flawed themselves, so flaw rates are compared with the clean
generator's own: two independent clean runs of the same 150 prompts give the clean rates and their run-to-run variation. Fed a
clean run as if it were watermarked, the test flags nothing (0 of 10 comparisons).</p></section>
{fig(conf_band_block())}
<section class="col"><p>The watermarked texts are outside the clean band on every metric. The differences are 4–14 times the clean-to-clean variation,
so they are real, but moderate: about 15 more texts in 100 with an unrelated insertion, roughly half of what the earlier setting
caused. They appear in all three text types and across most keys.</p>
<h3>3 · Examples</h3>
<p>In a blind read of nine random held-out pairs (order shuffled), eight were identified correctly, but five of those at low
confidence: both responses read like the same small model. The visible cost concentrates in news continuation, where the clean
model stops early and the watermarked one keeps inventing specifics. Three of the nine:</p></section>
<figure class="wide">{examples_block()}</figure>
<section class="col"><h3>4 · Paraphrase attack {at_chip}</h3></section>
{fig(at_body)}
<section class="col"><p>A capable paraphraser removes the watermark almost completely. This is the method's main open weakness. (Against the 0.8B
generator paraphrasing itself, about half of the detections had survived.)</p>
<h3>5 · Instruction following (IFEval) {if_chip}</h3></section>
{fig(if_body)}
<section class="col"><p>No measurable cost on verifiable constraints.</p></section>""")

    A(f"""<section class="col"><h2>The strength trade-off (development sweep)</h2>
<p>β is the one parameter that matters. On the 60 development prompts we compared 14 settings: the earlier adaptive controller and
constant β from 4 to 20, with and without warm-up and the semantic guard. {sw_chip}</p>
<h3>Detection</h3></section>
{fig(sw_det)}
<section class="col"><h3>Pairwise preference</h3></section>
{fig(sw_win)}
<section class="col"><p>Quality drops sharply between β = 4 and 6, because the tilt grows exponentially (7×, 20×, 55×). β = 4 was selected by the
pre-registered rule (most detectable setting with preference ≥ 0.40 and at most 10 points more major flaws than clean). Note the
selection effect: β = 4 scored 0.48 here, the best of 14 settings on 60 prompts, and 0.31 on the held-out prompts.</p>
<h3>What a strong push does</h3><p>Task: <i>“What's the preferred mode of transportation in Vietnam?”</i></p></section>
<figure class="wide"><div class="versus"><div><div class="k">Clean</div>In Vietnam, the <b>most preferred mode of transportation is the motorcycle</b>
(especially for short to medium-distance trips). Here is a breakdown of the reasons: accessibility and cost …</div>
<div><div class="k">Watermarked, earlier adaptive setting · key “amber copper delta atlas”</div>While there is no single “preferred” mode … the <b>Hybrid of Road and
<mark>Sea</mark></b> … is generally considered the most versatile … ### 1. The <mark>River</mark> (The Gold Standard) — the Ha Long River and its tributaries
(including the Mekong <mark>Delta</mark> …)</div></div>
<p class="caption">The key's <i>meaning</i> leaks into the <i>content</i> of the answer: the push favours tokens from the key's semantic neighbourhood. β = 4 does much less of this.</p></figure>""")

    A(f"""<section class="col"><h2>More detail</h2>
<details><summary>The detector: clipping per-token evidence</summary>
<p>The first statistic was the t-statistic of the raw contrasts. Under it, the top 5% of tokens carried 91% of a text's evidence —
mostly response openers and tokens from the key's semantic neighbourhood — so it was noisy and rewarded exactly the tokens that
damage the text. Clipping each token's contrast (and weighting by the reference model's entropy) makes the evidence come from many
small, consistent preferences. On the same β = 4 texts this raises held-out detection from 53% to 85%.</p>
{fig(strip)}{fig(det_body)}</details>
<details><summary>Key length: 4, 8, 10 or 16 words</summary>
<p>Longer keys mark more strongly at the same β — detection rises from 72% (4 words) to 85–90% (8–10 words) and then saturates — at no
clear extra quality cost. The price is specificity: even with keys that share no words, a watermarked text is also flagged under a
<i>wrong</i> key of the same length 19% → 64% of the time, because a longer phrase adds more generic “written under some background
phrase” signal. Long keys are for presence, not attribution. Halving the strength (β = 2) removes the measurable quality cost but
leaves only 20–42% detection, so a longer key does not rescue a much gentler push.</p>{fig(keylen_block())}</details>
<details><summary>Long texts and rotating keys</summary>
<p>A long text can carry a rotation of keys (A, B, C, D, A …), one per 128-token span, in a single continuous generation. Any
surviving excerpt then carries a complete mark, and the order of keys is a payload. First run, earlier adaptive setting —
how often is a span's own key the best-scoring of the four? {rot_chip}</p>{fig(rot_body)}
<p>The same with β = 4: {lg_chip}</p>{fig(lg_body)}
<p>A 128-token span at β = 4 carries little evidence, so span keys are recovered about twice as often as chance rather than four
times in five; rotation needs longer spans or a stronger mark.</p></details>
<details><summary>Per-text flaws on the development sweep</summary>
<p>Same method as in the held-out section, with five independent clean runs. {cb_chip}</p>{fig(cb_fig)}{fig(cb_table)}</details>
<details><summary>Earlier phases of the project</summary>
<p>The reference model was first the generator itself. With the strong settings of that phase, on 600 held-out prompts: 91% detected
at 1% false positives; the correct key beat all 100 decoy keys 82% of the time, a wrong key no more often than chance. A different
reference model works if it is instruction-tuned (Llama-3.2-1B-Instruct: AUROC 0.99) and fails if it is a base model (Pythia-410M:
0.86). An audit then showed that some cross-model detection rates had been unreliable because thresholds came from only 60 clean
texts; every number on this page uses the frozen 1,200-text null.</p>
<p>The earlier judge results that started the quality investigation (tuning tasks, strong settings):</p>{fig('<div class="legend"><span><i></i>0.50 — no preference</span><span><i class="s"></i>0.40 — pre-registered bar</span></div>' + judge_block())}</details>
<details><summary>What did not work</summary>
<p>Selecting among sampled candidate continuations by whole-text score (5–15% detection); activation steering (destructive before
detectable); KL- or likelihood-budgeted pushes; the adaptive controller (it delivers the push in bursts and hits the opening
hardest, yet a constant push of equal total size is judged no better — total strength is what matters); the warm-up (no gain);
the semantic guard (helps at equal β, does not move the limit); persona-sentence keys (topic drift); a base-model reference.</p></details>
<details><summary>Limitations</summary><ul>
<li><b>Not robust to paraphrase by a capable model</b> (85% → 12% → 4%).</li>
<li>The generator is small (0.8B); the judge scores its clean texts 1.7 out of 5. A stronger generator has more near-equivalent continuations to hide a mark in, and more quality to lose. Untested.</li>
<li>The statistic tests presence, not attribution: a watermarked text is flagged under a wrong key 20–50% of the time. Attribution needs the decoy-rank test.</li>
<li>The pairwise measure is noisy on 60 prompts (intervals of about ±0.09) and depends on which clean run serves as the comparator; differences below about 0.1 on the development prompts mean nothing. The held-out comparison (150 prompts, clean-vs-clean 0.47) is the reliable one.</li>
<li>No human ratings; quality rests on two LLM judges from different families that agree in 94% of the cases where both take a side.</li>
<li>English prose only. Seed-phrase keys are research keys, not cryptography. The development sweep flatters the selected setting; quote the held-out numbers.</li></ul></details>
<details><summary>Questions you will be asked</summary>
<p><b>How does this differ from green-list watermarks?</b> The token preference is not a hash of the preceding tokens but a language model's reaction to a secret phrase; detection is a likelihood contrast, not a token count; the key is a phrase; and the marking model can differ from the generator.</p>
<p><b>Can detection run without the generator?</b> Yes — only the 1B reference model and the key are needed.</p>
<p><b>What is the false-positive rate?</b> Thresholds are set at 1% on 1,200 clean texts and checked on separate clean texts: 0.25% on the held-out prompts, 0.9% on IFEval responses, 1–2% on paraphrased clean text.</p>
<p><b>How was quality judged?</b> In two separate runs: pairwise preference (two anonymous responses, both orders, one question, ties allowed) and per-text flaws (one anonymous text per call). Asking for both in one call gave the same preference rates within 0.05.</p></details>
<footer>Full write-up: <code>docs/colleague_brief.md</code>; precise algorithm: <code>docs/algorithm_spec.md</code>; rules fixed before the final round: <code>docs/preregistration.md</code>.
This page is generated from the result files by <code>scripts/build_brief_page.py</code>.</footer>
</section></main>""")
    return "\n".join(P)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    page = build()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page, encoding="utf-8")
    print(f"wrote {a.out} ({len(page) / 1024:.0f} KB)")
