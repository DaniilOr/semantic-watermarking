"""LaTeX tables of SemStamp's results on all our generators and datasets (slides / paper), generated from the result files.
Writes docs/semstamp_tables.tex.  Every number comes from runs/*.json or the generation records; nothing is typed in."""

import glob
import json
import os
import statistics as S
import sys
from pathlib import Path

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")

OB = ["ob_arxiv", "ob_booksum", "ob_wikipedia"]
SP = ["sp_realnews", "sp_booksum", "sp_tifu"]
CHAT = [("llama3b", "Llama-3.2-3B"), ("llama1b", "Llama-3.2-1B"), ("qwen08b", "Qwen3.5-0.8B")]


def J(path):
    return json.load(open(path)) if os.path.exists(path) else None


def tpr(res, run, dom, key="z_chain:human_1%", arm="semstamp"):
    x = (res or {}).get(f"{run}|{arm}|{dom}", {}).get(key)
    return None if not x else 100 * x["rate"]


def auroc(res, run, dom, arm="semstamp"):
    return (res or {}).get(f"{run}|{arm}|{dom}", {}).get("z_chain:auroc_vs_human")


def trip(vals, fmt="{:.1f}"):
    return " / ".join("--" if v is None else fmt.format(v) for v in vals)


def cost(run, doms, arm="semstamp"):
    tr = ac = mo = 0; ws = []; n = 0
    for f in glob.glob(f"runs/{run}/generations.{arm}*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if not l.strip().endswith("}"):
                continue
            r = json.loads(l)
            if r.get("domain") not in doms or r.get("status", "ok") != "ok":
                continue
            c = r.get("cost") or {}
            tr += c.get("trials", 0); ac += c.get("accepted", 0); mo += c.get("maxed_out", 0); ws.append(c.get("wall_seconds", 0)); n += 1
    return {"n": n, "cand": tr / ac if ac else None, "giveup": 100 * mo / ac if ac else None, "sec": S.mean(ws) if ws else None}


def judge(run, arm, doms=None, fname="api_pref_google_gemma-4-31b-it.json"):
    """Win rate vs plain text (ties count half); from the stored summary, or recomputed from the judgments when the summary
    of this arm was overwritten by a later re-judging."""
    d = J(f"runs/{run}/eval/{fname}")
    if d and doms:
        v = [d["per_arm_domain"].get(f"{arm}|{x}", {}).get("win_rate") for x in doms]
        v = [x for x in v if x is not None]
        if v:
            return S.mean(v)
    if d and arm in d.get("per_arm", {}) and not doms:
        return d["per_arm"][arm]["win_rate"]
    jl = f"runs/{run}/eval/api_pref_google_gemma-4-31b-it.jsonl"
    if os.path.exists(jl):
        from api_judge import summarize_preference
        rows = [r for r in map(json.loads, open(jl, encoding="utf-8")) if r.get("arm") == arm and (not doms or r.get("domain") in doms)]
        if rows:
            return summarize_preference(rows)["win_rate"]
    return None


def worse(qfile, run, arm="semstamp"):
    q = J(qfile)
    if not q:
        return None
    k = f"{run}/{arm}"
    vals = [m["arms"][k]["significantly_worse"] for m in q["metrics"].values() if k in m["arms"]]
    return f"{sum(vals)} of {len(vals)}" if vals else None


def fmt(v, f="{:.3f}"):
    return "--" if v is None else f.format(v)


def bench_table(title, label, doms, dnames, cols, extra_rows=()):
    """cols: (header, run, detection json, quality file, judge domains or None)"""
    L = []
    L += [r"\begin{table*}[t]", r"\centering", rf"\caption{{{title}}}", rf"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
          " & " + " & ".join(rf"\textbf{{{h.split('|')[0]}}}" for h, *_ in cols) + r" \\",
          " & " + " & ".join(f"({h.split('|')[1]})" for h, *_ in cols) + r" \\", r"\midrule"]
    R = [(J(det) or {}).get("results", {}) for _, run, det, _, _ in cols]
    runs = [run for _, run, _, _, _ in cols]
    n = [cost(run, doms) for run in runs]
    L.append(rf"Texts per dataset & " + " & ".join(str(round(c["n"] / len(doms))) if c["n"] else "--" for c in n) + r" \\")
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{TPR at 1\% FPR ({' / '.join(dnames)})}}}} \\")
    for name, suf, key in (("No attack", "", "z_chain:human_1%"), ("No attack, 0.1\\% FPR", "", "z_chain:human_0.1%"),
                           ("Sentence reorder", "_reorder_sent", "z_chain:human_1%"), ("Resegmentation", "_resegment", "z_chain:human_1%"),
                           ("Light paraphrase", "_para_light", "z_chain:human_1%"), ("Strong paraphrase", "_para_strong", "z_chain:human_1%")):
        L.append(rf"\quad {name} & " + " & ".join(trip([tpr(res, run + suf, d, key) for d in doms]) for res, run in zip(R, runs)) + r" \\")
    L.append(r"AUROC (no attack) & " + " & ".join(trip([auroc(res, run, d) for d in doms], "{:.3f}") for res, run in zip(R, runs)) + r" \\")
    L.append(r"Unwatermarked text flagged (\%) & " + " & ".join(trip([tpr(res, run, d, arm="base") for d in doms]) for res, run in zip(R, runs)) + r" \\")
    L.append(r"\midrule")
    L.append(r"Candidates per sentence & " + " & ".join(fmt(c["cand"], "{:.1f}") for c in n) + r" \\")
    L.append(r"Sentences given up (kept unmarked, \%) & " + " & ".join(fmt(c["giveup"], "{:.1f}") for c in n) + r" \\")
    L.append(r"Seconds per text$^{a}$ & " + " & ".join(fmt(c["sec"], "{:.0f}") for c in n) + r" \\")
    L.append(r"\midrule")
    L.append(r"Judge win rate (no-difference level) & " + " & ".join(
        f"{fmt(judge(run, 'semstamp', jd))} ({fmt(judge(run, 'clean_seed1', jd))})" for _, run, _, _, jd in cols) + r" \\")
    L.append(r"Flaw measures worse than unwatermarked & " + " & ".join(fmt(worse(qf, run), "{}") for _, run, _, qf, _ in cols) + r" \\")
    for name, cells in extra_rows:
        L.append(rf"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(L)


def main():
    mbpp = J("runs/mbpp_pass.json") or {}
    out = [r"% generated by scripts/semstamp_tables.py from runs/*.json -- do not edit by hand",
           r"% needs \usepackage{booktabs,graphicx}", ""]
    notes = (r"Detection: SemStamp's sentence-chain $z$-score; thresholds set on human-written continuations of separate documents "
             r"of the same dataset. Judge: Gemma-4-31B compares each text with an unwatermarked text for the same prompt (both orders, "
             r"ties count half); the no-difference level is a second unwatermarked sample judged the same way. Flaw measures: 5 "
             r"pointwise judge measures; counted when significantly worse than unwatermarked text. $^{a}$One of 3 processes sharing an A100.")
    cols = lambda bench: [
        ("Llama-3.2-3B|chat", "ob_bench" if bench == "ob" else "sp_bench_llama3b", "runs/ob_detect_ob_bench_refill.json" if bench == "ob" else "runs/sp_detect_llama3b.json",
         "runs/quality_vs_clean_ob_bench.json" if bench == "ob" else "runs/quality_vs_clean_sp_bench_llama3b.json", None if bench == "ob" else SP),
        ("Llama-3.2-1B|chat", f"{bench}_bench_llama1b", "runs/ob_detect_ob_bench_llama1b_refill.json" if bench == "ob" else "runs/sp_detect_llama1b.json",
         f"runs/quality_vs_clean_{bench}_bench_llama1b.json", None if bench == "ob" else SP),
        ("Qwen3.5-0.8B|chat", f"{bench}_bench_qwen08b_nl", f"runs/ob_detect_ob_bench_qwen08b_nl.json" if bench == "ob" else "runs/sp_detect_qwen08b_nl.json",
         f"runs/quality_vs_clean_{bench}_bench_qwen08b_nl.json", None),
        ("Llama-3.2-3B|base", f"{bench}_bench_llama3b_base", f"runs/ob_detect_ob_bench_llama3b_base.json" if bench == "ob" else "runs/sp_detect_llama3b_base.json",
         f"runs/quality_vs_clean_{bench}_bench_llama3b_base.json", None),
        ("Llama-3.2-1B|base", f"{bench}_bench_llama1b_base", f"runs/ob_detect_ob_bench_llama1b_base.json" if bench == "ob" else "runs/sp_detect_llama1b_base.json",
         f"runs/quality_vs_clean_{bench}_bench_llama1b_base.json", None),
        ("Qwen3.5-0.8B|base", f"{bench}_bench_qwen08b_base_nl", f"runs/ob_detect_ob_bench_qwen08b_base_nl.json" if bench == "ob" else "runs/sp_detect_qwen08b_base_nl.json",
         f"runs/quality_vs_clean_{bench}_bench_qwen08b_base_nl.json", None)]
    out.append(bench_table(
        r"SemStamp on our continuation benchmark: ArXiv / BookSum / Wikipedia, 500 prompts each (50-token excerpt, 200 tokens "
        r"generated). Chat models get the excerpt with a one-line continuation instruction; base models continue the excerpt. " + notes,
        "tab:semstamp_ob", OB, ["ArXiv", "BookSum", "Wikipedia"], cols("ob")))
    mrow = ("MBPP pass@1 (\\%), plain / SemStamp$^{b}$", [
        f"{100*mbpp['sp_bench_llama3b|base']['pass']:.1f} / {100*mbpp['sp_bench_llama3b|semstamp']['pass']:.1f}" if "sp_bench_llama3b|semstamp" in mbpp else "--",
        f"{100*mbpp['sp_bench_llama1b|base']['pass']:.1f} / {100*mbpp['sp_bench_llama1b|semstamp']['pass']:.1f}" if "sp_bench_llama1b|semstamp" in mbpp else "--",
        "--", "--", "--", "--"])
    out.append(bench_table(
        r"SemStamp on its own datasets with its protocol: RealNews / BookSum / Reddit-TIFU, 1,000 prompts each (32-token "
        r"prompt, 200 tokens). " + notes + r" $^{b}$MBPP: 974 problems, 200-token runs, first code block run against the tests.",
        "tab:semstamp_sp", SP, ["RealNews", "BookSum", "TIFU"], cols("sp"), extra_rows=[mrow]))
    # SemStamp's own setting
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{SemStamp in its own setting: RealNews, 1,000 articles; the base model continues the article's first sentence with "
         r"SemStamp's decoding (temperature 0.7, repetition penalty 1.05, 195--205 tokens). SemStamp's paper reports 93.9\% TPR at "
         r"1\% FPR for OPT-1.3B.}", r"\label{tab:semstamp_setting}", r"\begin{tabular}{lcc}", r"\toprule",
         r" & \textbf{OPT-1.3B} & \textbf{Llama-3.2-3B} \\", r" & (SemStamp's model) & (base) \\", r"\midrule"]
    rows = []
    for m in ("opt13b_ss", "llama3b_base"):
        res = (J(f"runs/semstamp_setting_detect_{m}.json") or {}).get("results", {}); run = f"semstamp_setting_{m}"
        rows.append({"t1": tpr(res, run, "sp_realnews"), "t01": tpr(res, run, "sp_realnews", "z_chain:human_0.1%"),
                     "au": auroc(res, run, "sp_realnews"), "fl": tpr(res, run, "sp_realnews", arm="base"), "c": cost(run, ["sp_realnews"])})
    for name, f in (("Texts", lambda r: str(r["c"]["n"])), ("TPR at 1\\% FPR", lambda r: fmt(r["t1"], "{:.1f}")),
                    ("TPR at 0.1\\% FPR", lambda r: fmt(r["t01"], "{:.1f}")), ("AUROC", lambda r: fmt(r["au"])),
                    ("Unwatermarked text flagged (\\%)", lambda r: fmt(r["fl"], "{:.1f}")),
                    ("Candidates per sentence", lambda r: fmt(r["c"]["cand"], "{:.1f}")),
                    ("Sentences given up (\\%)", lambda r: fmt(r["c"]["giveup"], "{:.1f}")),
                    ("Seconds per text", lambda r: fmt(r["c"]["sec"], "{:.0f}"))):
        L.append(f"{name} & " + " & ".join(f(r) for r in rows) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out.append("\n".join(L))
    # instruction tasks
    ID = [("it_dolly", "Dolly"), ("it_writingprompts", "WritingPrompts"), ("it_gsm8k", "GSM8K"), ("it_ifeval", "IFEval")]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{SemStamp on instruction tasks (chat models; answers up to 512 tokens): Dolly-15k open-ended requests (1,000), "
         r"WritingPrompts stories (1,000), GSM8K math (1,319), IFEval (541). TPR thresholds from human answers (Dolly, stories, "
         r"GSM8K solutions); IFEval has no human answers, so its threshold is set on unwatermarked model answers ($^{c}$). "
         r"Candidates per sentence are not reported: part of these runs drew 20 candidates per GPU call, which changes the count of "
         r"unused draws but not the output.}", r"\label{tab:semstamp_it}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccc}",
         r"\toprule", " & " + " & ".join(rf"\textbf{{{n}}}" for _, n in CHAT) + r" \\", r"\midrule",
         r"\multicolumn{4}{l}{\textit{TPR at 1\% FPR}} \\"]
    dets = {m: (J(f"runs/it_detect_{m}.json") or {}).get("results", {}) for m, _ in CHAT}
    for d, dn in ID:
        key = "z_chain:model_1%" if d == "it_ifeval" else "z_chain:human_1%"
        L.append(rf"\quad {dn}{'$^{c}$' if d == 'it_ifeval' else ''} & " + " & ".join(fmt(tpr(dets[m], f"it_bench_{m}", d, key), "{:.1f}") for m, _ in CHAT) + r" \\")
    L.append(r"Sentences given up (\%, all four tasks) & " + " & ".join(fmt(cost(f"it_bench_{m}", [d for d, _ in ID])["giveup"], "{:.1f}") for m, _ in CHAT) + r" \\")
    L.append(r"\midrule")
    q = {m: J(f"runs/it_quality_it_bench_{m}.json") or {} for m, _ in CHAT}
    L.append(r"GSM8K accuracy (\%), plain / SemStamp & " + " & ".join(
        f"{100*q[m]['base|gsm8k']['accuracy']:.1f} / {100*q[m]['semstamp|gsm8k']['accuracy']:.1f}" for m, _ in CHAT) + r" \\")
    L.append(r"IFEval strict prompt-level (\%), plain / SemStamp & " + " & ".join(
        f"{100*q[m]['base|ifeval']['strict']['prompt_level']:.1f} / {100*q[m]['semstamp|ifeval']['strict']['prompt_level']:.1f}" for m, _ in CHAT) + r" \\")
    for d, dn in ID:
        L.append(rf"Judge win rate, {dn} (no-difference level) & " + " & ".join(
            f"{fmt(judge(f'it_bench_{m}', 'semstamp', [d], 'api_pref_google_gemma-4-31b-it.instruction4.json'))} "
            f"({fmt(judge(f'it_bench_{m}', 'clean_seed1', [d], 'api_pref_google_gemma-4-31b-it.instruction4.json'))})" for m, _ in CHAT) + r" \\")
    L.append(r"Flaw measures worse than unwatermarked & " + " & ".join(fmt(worse(f"runs/quality_vs_clean_it_bench_{m}.json", f"it_bench_{m}"), "{}") for m, _ in CHAT) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    out.append("\n".join(L))
    Path("docs/semstamp_tables.tex").write_text("\n".join(out))
    print("wrote docs/semstamp_tables.tex")


if __name__ == "__main__":
    main()
