"""Compact summary of the main results (Push + Margin = arm chain_reference vs SemStamp), Markdown, generated from the
result files; nothing is typed in except the SemStamp paper's own numbers (Hou et al. 2024, Table 1, OPT-1.3B, RealNews).
Usage: python scripts/main_results.py OUT.md"""

import json
import statistics as S
import sys
from pathlib import Path

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from semstamp_slide import runs  # noqa: E402
from semstamp_tables import J, OB, SP, cost, judge, tpr, worse  # noqa: E402

PAPER = {"none": 93.9, "pegasus": 83.7, "pegasus_bigram": 76.7, "parrotss": 56.2, "parrotss_bigram": 54.4}
ATT = [("none", "No attack"), ("pegasus", "Pegasus"), ("pegasus_bigram", "Pegasus-bigram"), ("parrot", "Parrot, top beam"),
       ("parrot_bigram", "Parrot-bigram, top beam")]
ATT_SS = ATT + [("parrotss", "Parrot, SemStamp's pipeline"), ("parrotss_bigram", "Parrot-bigram, SemStamp's pipeline")]
NAME = {"3B": "Llama-3.2-3B", "1B": "Llama-3.2-1B", "Qwen": "Qwen3.5-0.8B"}


def f(v, p="{:.0f}"):
    return "–" if v is None else p.format(v)


def pair(a, b, p="{:.0f}"):
    return f"{f(a, p)} / **{f(b, p)}**"


def mean_tpr(res, run, doms, key="z_chain:human_1%", suf="", arm="semstamp"):
    v = [tpr(res, run + suf, d, key, arm=arm) for d in doms]
    return None if any(x is None for x in v) else S.mean(v)


def bench_section(bench, doms, title, push_cost):
    L = [f"### {title}", "",
         "| Model | TPR 1% FPR | TPR 0.1% FPR | Light para | Strong para | Judge win (no-diff) | Cand./sentence | Unmarked % | s/text |",
         "|---|---|---|---|---|---|---|---|---|"]
    for kind, short, run, det, jd in runs(bench):
        res = (J(det) or {}).get("results", {})
        r = {}
        for arm in ("semstamp", "chain_reference"):
            r[arm] = dict(t1=mean_tpr(res, run, doms, arm=arm), t01=mean_tpr(res, run, doms, "z_chain:human_0.1%", arm=arm),
                          pl=mean_tpr(res, run, doms, suf="_para_light", arm=arm), ps=mean_tpr(res, run, doms, suf="_para_strong", arm=arm),
                          j=judge(run, arm, jd))
        c = cost(run, doms); pc = push_cost.get(f"{bench}|{kind}|{short}", {})
        jn = judge(run, "clean_seed1", jd)
        L.append(f"| {NAME[short]} {kind} | {pair(r['semstamp']['t1'], r['chain_reference']['t1'])} | "
                 f"{pair(r['semstamp']['t01'], r['chain_reference']['t01'])} | {pair(r['semstamp']['pl'], r['chain_reference']['pl'])} | "
                 f"{pair(r['semstamp']['ps'], r['chain_reference']['ps'])} | "
                 f"{pair(r['semstamp']['j'], r['chain_reference']['j'], '{:.3f}')} ({f(jn, '{:.3f}')}) | "
                 f"{pair(c['cand'], pc.get('cand'), '{:.1f}')} | {pair(c['giveup'], pc.get('out_of_mask'))} | {pair(c['sec'], pc.get('sec'))} |")
    return L + [""]


def stress_section():
    L = ["| Model | Benchmark | Reorder sentences | Resegment | Flaw measures worse than plain (of 5) |", "|---|---|---|---|---|"]
    for bench, doms, bname in (("sp", SP, "SemStamp's"), ("ob", OB, "ours")):
        for kind, short, run, det, _ in runs(bench):
            res = (J(det) or {}).get("results", {})
            q = f"runs/quality_vs_clean_{run}.json"
            L.append(f"| {NAME[short]} {kind} | {bname} | "
                     f"{pair(mean_tpr(res, run, doms, suf='_reorder_sent'), mean_tpr(res, run, doms, suf='_reorder_sent', arm='chain_reference'))} | "
                     f"{pair(mean_tpr(res, run, doms, suf='_resegment'), mean_tpr(res, run, doms, suf='_resegment', arm='chain_reference'))} | "
                     f"{worse(q, run) or '–'} / **{worse(q, run, 'chain_reference') or '–'}** |")
    return L


def attack_section():
    cols = [("OPT-1.3B, SemStamp's setting", "semstamp_setting_opt13b_ss"), ("Llama-3B base, SemStamp's setting", "semstamp_setting_llama3b_base"),
            ("Llama-3B chat", "sp_bench_llama3b"), ("Llama-1B chat", "sp_bench_llama1b"), ("Qwen-0.8B chat", "sp_bench_qwen08b_nl")]
    res = {run: (J(f"runs/sentpara_detect_{run}.json") or {}).get("results", {}) for _, run in cols}
    pss = {run: (J(f"runs/sentpara_detect_{run}_parrotss.json") or {}).get("results", {}) for _, run in cols}
    clean = {"sp_bench_llama1b": (J("runs/sp_detect_llama1b.json") or {}).get("results", {}),
             "sp_bench_qwen08b_nl": (J("runs/sp_detect_qwen08b_nl.json") or {}).get("results", {})}
    L = ["| Attack | SemStamp paper (OPT-1.3B) | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for a, an in ATT_SS:
        cells = []
        for _, run in cols:
            rr = pss[run] if a.startswith("parrotss") else res[run]
            if a == "none" and run in clean and tpr(rr, run, "sp_realnews") is None:
                rr = clean[run]
            suf = "" if a == "none" else f"_{a}"
            cells.append(pair(tpr(rr, run + suf, "sp_realnews"), tpr(rr, run + suf, "sp_realnews", arm="chain_reference")))
        L.append(f"| {an} | {PAPER.get(a, '–')} | " + " | ".join(cells) + " |")
    return L


def base_attack_section():
    cols = [("Llama-3B base, RealNews", "sp_bench_llama3b_base", ["sp_realnews"]), ("Llama-1B base, RealNews", "sp_bench_llama1b_base", ["sp_realnews"]),
            ("Qwen-0.8B base, RealNews", "sp_bench_qwen08b_base_nl", ["sp_realnews"]), ("Llama-3B base, ours", "ob_bench_llama3b_base", OB),
            ("Llama-1B base, ours", "ob_bench_llama1b_base", OB), ("Qwen-0.8B base, ours", "ob_bench_qwen08b_base_nl", OB)]
    res = {run: (J(f"runs/sentpara_detect_{run}.json") or {}).get("results", {}) for _, run, _ in cols}
    L = ["| Attack | " + " | ".join(c for c, _, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for a, an in ATT:
        suf = "" if a == "none" else f"_{a}"
        L.append(f"| {an} | " + " | ".join(pair(mean_tpr(res[run], run, doms, suf=suf), mean_tpr(res[run], run, doms, suf=suf, arm="chain_reference"))
                                           for _, run, doms in cols) + " |")
    return L


def ob_attack_section():
    cols = [("Llama-3B chat", "ob_bench"), ("Llama-1B chat", "ob_bench_llama1b"), ("Qwen-0.8B chat", "ob_bench_qwen08b_nl")]
    res = {run: (J(f"runs/sentpara_detect_{run}.json") or {}).get("results", {}) for _, run in cols}
    L = ["| Attack | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for a, an in ATT:
        suf = "" if a == "none" else f"_{a}"
        L.append(f"| {an} | " + " | ".join(pair(mean_tpr(res[run], run, OB, suf=suf), mean_tpr(res[run], run, OB, suf=suf, arm="chain_reference"))
                                           for _, run in cols) + " |")
    return L


def it_section():
    M = [("llama3b", "Llama-3B"), ("llama1b", "Llama-1B"), ("qwen08b", "Qwen-0.8B")]
    T = [("it_dolly", "Dolly"), ("it_writingprompts", "WritingPrompts"), ("it_gsm8k", "GSM8K"), ("it_ifeval", "IFEval"), ("it_translation", "Translation de→en")]
    det = {m: (J(f"runs/it_detect_{m}.json") or {}).get("results", {}) for m, _ in M}
    tdet = {m: (J(f"runs/it_detect_translation_{m}.json") or {}).get("results", {}) for m, _ in M}
    q = {m: J(f"runs/it_quality_it_bench_{m}.json") or {} for m, _ in M}
    jf = "api_pref_google_gemma-4-31b-it.translation.json"
    L = ["| | " + " | ".join(n for _, n in M) + " |", "|---" * (len(M) + 1) + "|", "| *TPR at 1% FPR, SemStamp / push* | | | |"]
    for d, dn in T:
        key = "z_chain:model_1%" if d == "it_ifeval" else "z_chain:human_1%"
        src = tdet if d == "it_translation" else det
        L.append(f"| {dn}{' ¹' if d == 'it_ifeval' else ''} | " + " | ".join(
            pair(tpr(src[m], f"it_bench_{m}", d, key), tpr(src[m], f"it_bench_{m}", d, key, arm="chain_reference")) for m, _ in M) + " |")
    L.append("| *Task quality, plain / SemStamp / push* | | | |")
    acc = lambda m, a: 100 * q[m][f"{a}|gsm8k"]["accuracy"] if f"{a}|gsm8k" in q[m] else None
    ife = lambda m, a: 100 * q[m][f"{a}|ifeval"]["strict"]["prompt_level"] if f"{a}|ifeval" in q[m] else None
    chrf = lambda m, a: q[m][f"{a}|translation"]["chrf"] if f"{a}|translation" in q[m] else None
    for name, fn in (("GSM8K accuracy %", acc), ("IFEval strict prompt-level %", ife), ("Translation chrF", chrf)):
        L.append(f"| {name} | " + " | ".join(f"{f(fn(m, 'base'), '{:.1f}')} / {f(fn(m, 'semstamp'), '{:.1f}')} / **{f(fn(m, 'chain_reference'), '{:.1f}')}**" for m, _ in M) + " |")
    L.append("| *Judge win rate vs plain, SemStamp / push (no-diff)* | | | |")
    for d, dn in T:
        L.append(f"| {dn} | " + " | ".join(
            f"{pair(judge(f'it_bench_{m}', 'semstamp', [d], jf), judge(f'it_bench_{m}', 'chain_reference', [d], jf), '{:.3f}')} "
            f"({f(judge(f'it_bench_{m}', 'clean_seed1', [d], jf), '{:.3f}')})" for m, _ in M) + " |")
    L.append("| Flaw measures worse than plain (of 5), SemStamp / push | " + " | ".join(
        f"{worse(f'runs/quality_vs_clean_it_bench_{m}.json', f'it_bench_{m}') or '–'} / **{worse(f'runs/quality_vs_clean_it_bench_{m}.json', f'it_bench_{m}', 'chain_reference') or '–'}**" for m, _ in M) + " |")
    L.append("| Sentences SemStamp gave up on, % (Dolly, WP, GSM8K, IFEval) | " + " | ".join(
        f(cost(f"it_bench_{m}", [d for d, _ in T[:4]])["giveup"]) for m, _ in M) + " |")
    return L


def decomp_section():
    d = J("runs/ss_detect_decomp.json") or {}
    L = ["| | " + " | ".join(n for n in ("ArXiv", "BookSum", "Wikipedia")) + " |", "|---|---|---|---|"]
    for key, name, p in (("det_scored_per_text", "Sentences scored per text", "{:.1f}"), ("det_in_mask", "Scored sentences in their mask, %", "{:.1f}")):
        mul = 100 if key == "det_in_mask" else 1
        L.append(f"| {name}, SemStamp / push | " + " | ".join(
            pair(mul * d.get(f"ob_bench|semstamp|{x}", {}).get(key, float('nan')), mul * d.get(f"ob_bench|chain_reference|{x}", {}).get(key, float('nan')), p)
            for x in OB) + " |")
    return L


def capacity_section():
    d = J("runs/capacity_entropy.json") or {}
    if not d:
        return ["(runs/capacity_entropy.json missing)"]
    L = ["| Bin entropy of 16 candidates | Share of positions % | Exact duplicates % | ≥1 usable of 16, plain % | ≥1 usable of 16, push % |",
         "|---|---|---|---|---|"]
    for a, b in zip(d["plain"]["buckets"], d["push"]["buckets"]):
        L.append(f"| {a['bucket']} | {100 * a['share']:.1f} | {100 * a['dup']:.1f} | {100 * a['any_clear16']:.1f} | **{100 * b['any_clear16']:.1f}** |")
    L.append(f"| all | 100 | {100 * d['plain']['overall']['dup']:.1f} | {100 * d['plain']['overall']['any_clear']:.1f} | **{100 * d['push']['overall']['any_clear']:.1f}** |")
    return L


def variants_section():
    import glob
    from collections import defaultdict
    from it_eval import gsm8k_correct
    prompts = {json.loads(l)["prompt_id"]: json.loads(l) for l in open("data/it_bench/prompts.jsonl", encoding="utf-8") if '"test"' in l}
    recs = defaultdict(dict)
    for fn in glob.glob("runs/it_fix_llama3b/generations.*.jsonl"):
        for l in open(fn, encoding="utf-8"):
            if l.strip().endswith("}"):
                r = json.loads(l)
                if r.get("status", "ok") == "ok" and r.get("domain") == "it_gsm8k":
                    recs[r["arm"]][r["prompt_id"]] = r
    names = [("base", "plain"), ("semstamp", "SemStamp"), ("chain_reference", "push + margin"), ("chain_spread", "+ spread gate"),
             ("chain_protect", "+ protected tokens"), ("chain_spread_protect", "+ both"), ("chain_beta05", "push strength 0.5")]
    det = (J("runs/it_fix_detect_llama3b.json") or {}).get("results", {})
    names.insert(1, ("clean_seed1", "plain, 2nd sample (no-diff)"))
    L = ["| Arm | TPR 1% FPR, Dolly | TPR 1% FPR, GSM8K | GSM8K accuracy % (n) | Judge win rate |", "|---|---|---|---|---|"]
    for a, n in names:
        rs = list(recs.get(a, {}).values())
        if rs:
            k = sum(gsm8k_correct(r.get("text") or "", prompts[r["prompt_id"]]["gold"])[1] for r in rs)
            L.append(f"| {n} | {f(tpr(det, 'it_fix_llama3b', 'it_dolly', arm=a), '{:.1f}')} | {f(tpr(det, 'it_fix_llama3b', 'it_gsm8k', arm=a), '{:.1f}')} | "
                     f"{100 * k / len(rs):.1f} ({len(rs)}) | {f(judge('it_fix_llama3b', a) if a != 'base' else None, '{:.3f}')} |")
    return L

def push_cost(run, doms, arm):
    import glob
    nd = n = inm = 0; ws = []
    for fn in glob.glob(f"runs/{run}/generations.{arm}.shard*.jsonl"):
        for l in open(fn, encoding="utf-8"):
            if not l.strip().endswith("}"):
                continue
            r = json.loads(l)
            if r.get("domain") not in doms or r.get("status", "ok") != "ok":
                continue
            c = r.get("cost") or {}; ws.append(c.get("wall_seconds", 0))
            for u in (c.get("units") or [])[1:]:
                n += 1; nd += u["n_drawn"]; inm += bool(u["in_mask"])
    return {"cand": nd / n if n else None, "out": 100 - 100 * inm / n if n else None, "sec": S.mean(ws) if ws else None}


KS = [("OPT-1.3B, SemStamp's setting", "RealNews", "sp_realnews", "ks_setting_opt13b_ss", "semstamp_setting_opt13b_ss", "runs/semstamp_setting_detect_opt13b_ss.json"),
      ("Llama-3B chat", "RealNews", "sp_realnews", "ks_sp_bench_llama3b", "sp_bench_llama3b", "runs/sp_detect_llama3b.json"),
      ("Llama-3B chat", "BookSum", "sp_booksum", "ks_sp_bench_llama3b_bs", "sp_bench_llama3b", "runs/sp_detect_llama3b.json"),
      ("Llama-1B chat", "RealNews", "sp_realnews", "ks_sp_bench_llama1b", "sp_bench_llama1b", "runs/sp_detect_llama1b.json"),
      ("Llama-1B chat", "BookSum", "sp_booksum", "ks_sp_bench_llama1b", "sp_bench_llama1b", "runs/sp_detect_llama1b.json"),
      ("Qwen-0.8B chat", "RealNews", "sp_realnews", "ks_sp_bench_qwen08b_nl", "sp_bench_qwen08b_nl", "runs/sp_detect_qwen08b_nl.json"),
      ("Qwen-0.8B chat", "BookSum", "sp_booksum", "ks_sp_bench_qwen08b_nl", "sp_bench_qwen08b_nl", "runs/sp_detect_qwen08b_nl.json")]


def ksem_section():
    L = ["| Model | Dataset | TPR 1% FPR, k-means | TPR 0.1% FPR, k-means | TPR 1% FPR, LSH | TPR 0.1% FPR, LSH | Cand./sentence, k-means | Unmarked %, k-means | s/text, k-means |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, dname, d, ks, lsh, lshdet in KS:
        kr = (J(f"runs/ks_detect_{ks}_{d}.json") or {}).get("results", {})
        lr = (J(lshdet) or {}).get("results", {})
        c = cost(ks, [d], arm="ksemstamp"); pc = push_cost(ks, [d], "chain_kmeans")
        L.append(f"| {name} | {dname} | {pair(tpr(kr, ks, d, arm='ksemstamp'), tpr(kr, ks, d, arm='chain_kmeans'), '{:.1f}')} | "
                 f"{pair(tpr(kr, ks, d, 'z_chain:human_0.1%', arm='ksemstamp'), tpr(kr, ks, d, 'z_chain:human_0.1%', arm='chain_kmeans'), '{:.1f}')} | "
                 f"{pair(tpr(lr, lsh, d), tpr(lr, lsh, d, arm='chain_reference'), '{:.1f}')} | "
                 f"{pair(tpr(lr, lsh, d, 'z_chain:human_0.1%'), tpr(lr, lsh, d, 'z_chain:human_0.1%', arm='chain_reference'), '{:.1f}')} | "
                 f"{pair(c['cand'], pc['cand'], '{:.1f}')} | {pair(c['giveup'], pc['out'], '{:.1f}')} | {pair(c['sec'], pc['sec'])} |")
    return L


def kgw_res(run):
    return (J(f"runs/kgw_detect_{run}.json") or {}).get("results", {})


def kgw_tpr(res, run, dom, key="z_kgw:human_1%"):
    x = res.get(f"kgw_{run}|kgw|{dom}", {}).get(key)
    return None if not x else 100 * x["rate"]


def ppl_all():
    d = {}
    for f in ("runs/ppl_part_a.json", "runs/ppl_part_b.json"):
        d.update(J(f) or {})
    return d


def ppl(d, run, arm, doms, key="ppl_median"):
    v = [d[f"{run}|{arm}|{x}"][key] for x in doms if f"{run}|{arm}|{x}" in d]
    return S.mean(v) if v else None


def kgw_section():
    P = ppl_all()
    L = ["| Model | Benchmark | TPR 1%: SemStamp / push / **KGW** | TPR 0.1%: SemStamp / push / **KGW** | PPL median: plain / SemStamp / push / KGW | PPL mean (SemStamp's statistic): plain / SemStamp / push / KGW |",
         "|---|---|---|---|---|---|"]
    setting = [("OPT-1.3B", "SemStamp's setting", "semstamp_setting_opt13b_ss", "runs/semstamp_setting_detect_opt13b_ss.json", ["sp_realnews"]),
               ("Llama-3B base", "SemStamp's setting", "semstamp_setting_llama3b_base", "runs/semstamp_setting_detect_llama3b_base.json", ["sp_realnews"])]
    cols = setting + [(NAME[short] + " " + kind, bn, run, det, doms) for bench, doms, bn in (("sp", SP, "SemStamp's"), ("ob", OB, "ours"))
                      for kind, short, run, det, _ in runs(bench)]
    for name, bn, run, det, doms in cols:
        res = (J(det) or {}).get("results", {}); kr = kgw_res(run)
        t = lambda key, kkey: " / ".join(f(x) for x in (mean_tpr(res, run, doms, key), mean_tpr(res, run, doms, key, arm="chain_reference"))) + \
            f" / **{f(S.mean([kgw_tpr(kr, run, d_, kkey) for d_ in doms]) if all(kgw_tpr(kr, run, d_, kkey) is not None for d_ in doms) else None)}**"
        pm = lambda key: " / ".join(f(ppl(P, r, a, doms, key), "{:.1f}") for r, a in ((run, "base"), (run, "semstamp"), (run, "chain_reference"), ("kgw_" + run, "kgw")))
        L.append(f"| {name} | {bn} | {t('z_chain:human_1%', 'z_kgw:human_1%')} | {t('z_chain:human_0.1%', 'z_kgw:human_0.1%')} | {pm('ppl_median')} | {pm('ppl_mean')} |")
    return L


def kgw_attack_section():
    cols = [("OPT-1.3B, SemStamp's setting", "semstamp_setting_opt13b_ss"), ("Llama-3B base, SemStamp's setting", "semstamp_setting_llama3b_base"),
            ("Llama-3B chat", "sp_bench_llama3b"), ("Llama-1B chat", "sp_bench_llama1b"), ("Qwen-0.8B chat", "sp_bench_qwen08b_nl")]
    res = {run: (J(f"runs/sentpara_detect_{run}.json") or {}).get("results", {}) for _, run in cols}
    pss = {run: (J(f"runs/sentpara_detect_{run}_parrotss.json") or {}).get("results", {}) for _, run in cols}
    clean = {"sp_bench_llama1b": (J("runs/sp_detect_llama1b.json") or {}).get("results", {}),
             "sp_bench_qwen08b_nl": (J("runs/sp_detect_qwen08b_nl.json") or {}).get("results", {})}
    L = ["| Attack | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for a, an in ATT_SS:
        cells = []
        for _, run in cols:
            rr = pss[run] if a.startswith("parrotss") else res[run]
            if a == "none" and run in clean and tpr(rr, run, "sp_realnews") is None:
                rr = clean[run]
            suf = "" if a == "none" else f"_{a}"
            kx = kgw_res(run).get(f"kgw_{run}{suf}|kgw|sp_realnews", {}).get("z_kgw:human_1%")
            cells.append(f"{f(tpr(rr, run + suf, 'sp_realnews'))} / {f(tpr(rr, run + suf, 'sp_realnews', arm='chain_reference'))} / **{f(None if not kx else 100 * kx['rate'])}**")
        L.append(f"| {an} | " + " | ".join(cells) + " |")
    return L


def main():
    out = Path(sys.argv[1])
    push_cost = J("runs/push_cost_cols.json") or {}
    L = ["# Main results: Push + Margin vs SemStamp", "",
         "Generated by `wmworld/scripts/main_results.py` from the result files (measured numbers only). Cells \"A / **B**\" = "
         "SemStamp / **Push + Margin**. TPR = share of watermarked texts detected; FPR thresholds from human-written texts of "
         "the same dataset. Judge = Gemma-4-31B pairwise win rate against a plain (unwatermarked) text for the same prompt, ties "
         "count half; \"no-diff\" = a second plain sample judged the same way.", "",
         "## 1. Continuation benchmarks (200 tokens; mean over 3 datasets)", "",
         "Unmarked % = SemStamp: sentences kept after 100 rejected tries; push: kept sentences outside the mask (cap 16 reached). "
         "Cand./sentence = candidate sentences drawn per kept sentence (push: sentences after the first). s/text = one of 3 processes on an A100.", ""]
    L += bench_section("sp", SP, "SemStamp's datasets: RealNews, BookSum, Reddit-TIFU (1,000 prompts each)", push_cost)
    L += bench_section("ob", OB, "Our benchmark: ArXiv, BookSum, Wikipedia (500 prompts each)", push_cost)
    L += ["### Other stress tests and pointwise flaw measures (mean over 3 datasets)", "",
          "Reorder = sentences shuffled; resegment = sentence boundaries moved (no words changed). Flaw measures: 5 pointwise-judge "
          "measures (major flaw, number of flaws, off task, insertions/drift, 1-5 quality); counted when significantly worse than "
          "plain text, judged against the spread between independent plain samples.", ""]
    L += stress_section() + [""]
    L += ["## 2. SemStamp's sentence-level paraphrase attacks (TPR at 1% FPR)", "",
          "Our Parrot keeps the top beam without the Parrot library's diversity ranking and filters, so it is milder than the paper's "
          "(SemStamp keeps 96% under it on OPT vs 56% in the paper). \"SemStamp's pipeline\" rows rerun Parrot exactly as SemStamp's "
          "code does (group beam search, adequacy and fluency filters at 0.8, most-different candidate by edit distance, lowercase / "
          "punctuation stripped; bigram variant scored with BERTScore), except that empty candidates are dropped.", ""]
    L += attack_section() + [""]
    L += ["Base models (RealNews: 1,000 texts; ours: mean over ArXiv, BookSum, Wikipedia, 500 each):", ""]
    L += base_attack_section() + [""]
    L += ["Chat models on our benchmark (ArXiv, BookSum, Wikipedia; 500 texts each; mean over the 3 datasets):", ""]
    L += ob_attack_section() + [""]
    L += ["## 3. Instruction tasks (chat models, answers up to 512 tokens)", "",
          "Dolly 1,000, WritingPrompts 1,000, GSM8K 1,319, IFEval 541, News Commentary de→en 1,000. ¹IFEval has no human answers: "
          "threshold set on plain model answers.", ""]
    L += it_section() + [""]
    L += ["## 4. Why push detects more (Llama-3B chat, our benchmark)", "",
          "Same number of scored sentences; the difference is the share of sentences inside their mask.", ""]
    L += decomp_section() + [""]
    L += ["## 5. Capacity follows semantic uncertainty (Llama-3.2-3B-Instruct, training prompts)", "",
          "Per sentence position, 16 candidates; bin entropy = entropy of their distribution over the 8 regions; usable = in mask and "
          "clear of the 0.02 margin. 24,209 positions (plain), 11,945 (push).", ""]
    L += capacity_section() + [""]
    L += ["## 6. Push variants against the GSM8K quality loss (Llama-3B, 300 GSM8K + 300 Dolly prompts)", "",
          "Spread gate: a sentence's first block is drawn without the push; later blocks push only if those candidates landed in >= 2 "
          "regions. Protected tokens: digits, prompt content words and hedging words are never pushed. Judge = both tasks (600 texts).", ""]
    L += variants_section() + [""]
    L += ["## 7. k-SemStamp: the same comparison on k-SemStamp's k-means partition", "",
          "k-means = k-SemStamp's 8 cluster centres and fine-tuned SBERT per domain (abehou/SemStamp centroids: c4 for RealNews, booksum "
          "for BookSum); region = closest centre, margin = second-closest minus closest cosine distance > 0.035. Cells: k-means columns "
          "= k-SemStamp / **push + margin on the k-means partition** (stop margin 0.035); LSH columns = SemStamp / **push + margin** "
          "from the main runs on the same prompts. k-SemStamp's paper: 98.1% TPR at 1% FPR for OPT-1.3B on RealNews. Unmarked % = "
          "k-SemStamp: gave up after 100 tries; push: kept outside the mask. 1,000 texts per cell.", ""]
    L += ksem_section() + [""]
    L += ["## 8. KGW reference (Kirchenbauer et al. 2023) and perplexity", "",
          "KGW: gamma 0.25, delta 2.0, simple_1 seeding (lm-watermarking repo), on the same prompts and decoding as each main run; "
          "token-level z-score with the generator's tokenizer, thresholds from the same human texts. Perplexity as in SemStamp's paper: "
          "OPT-2.7B, per-text perplexity of the generated text, averaged over texts (mean); the median is shown too because a few "
          "degenerate texts inflate some means. Means over the datasets of each benchmark.", ""]
    L += kgw_section() + [""]
    L += ["SemStamp's sentence-level attacks on RealNews (1,000 texts, TPR at 1% FPR): SemStamp / push / **KGW**. Gemma full-text "
          "paraphrase (light / strong) has not been run on the KGW texts.", ""]
    L += kgw_attack_section() + [""]
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
