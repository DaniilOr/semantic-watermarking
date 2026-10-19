"""Where does the push land?  Offline analysis of existing per-token traces (no model pass).

A generated token is a HEAD token when it is no more surprising than the generator's own uncertainty at that
position (-log p_G(token) <= H_t), and a TAIL token otherwise.  Questions:
 1. how much of the evidence is carried by tail tokens (what a head-only push would give up);
 2. do the texts the judge flagged for insertions / drift contain more promoted tail tokens, and are those content words;
 3. how the evidence is distributed along 512-token texts, per domain.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.posstats import auroc, gate_weights  # noqa: E402

SLUG = "google_gemma-4-31b-it"
CLIP = 0.5
STOP = set("""a an the and or but if then than so because as of at by for with about against between into through during before after above
below to from up down in out on off over under again further once here there when where why how all any both each few more most other some
such no nor not only own same too very s t can will just don should now i me my myself we our ours ourselves you your yours yourself he him
his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were
be been being have has had having do does did doing would could might must shall may also however while though although yet still even ever
never always often said says say one two three first new like get got make made go went going come came know see seen much many well back
us upon per via""".split())


def load_run(run):
    rd = Path("runs") / run
    gens = {}
    for f in rd.glob("generations*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            if g.get("status", "ok") == "ok":
                gens[g["id"]] = g
    traces = {}
    for f in rd.glob("trajectories*.jsonl"):
        for ln in open(f):
            if '"per_token"' not in ln:
                continue
            r = json.loads(ln)
            pt = r["cost"]["per_token"]
            ids = r["candidate_token_ids"][0]
            n = len(ids)
            if n < 32 or r["output_id"] not in gens:
                continue
            a = {k: np.asarray(pt[k][:n], dtype=np.float64) for k in ("entropy", "beta_t", "contrast", "logp_g", "contrast_raw", "kl")}
            a["ids"] = ids
            traces[r["output_id"]] = a
    flaws = {}
    p = rd / "eval" / f"api_pointwise_{SLUG}.jsonl"
    if p.exists():
        for ln in open(p):
            r = json.loads(ln)
            flaws[r["gen_id"]] = r["judgment"]
    return gens, traces, flaws


def split(a):
    tail = (-a["logp_g"]) > a["entropy"] + 1e-9
    ev = np.clip(a["contrast_raw"], -CLIP, CLIP)             # the detector's clipped evidence at the chosen token (bridge approximation)
    return tail, ev


def tstat(x):
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(x.size))) if x.size > 2 and sd > 0 else 0.0


out = {}
print("=" * 100)
print("1. HEAD vs TAIL tokens: shares and evidence, by push strength (len10 keys, 60 dev prompts; held-out run last)")
print(f"{'run/arm':26s} {'texts':>5s} {'tail share':>10s} {'ev/token head':>13s} {'ev/token tail':>13s} {'tail share of evidence':>22s} {'T all':>6s} {'T head-only*':>12s}")
runs = {}
for run in ("keylen10", "strength_fill", "conf_cross"):
    runs[run] = load_run(run)
rows = []
for run, arms in (("keylen10", ["wm_c2"]), ("strength_fill", ["wm_c25", "wm_c3", "wm_c35"]), ("keylen10", ["wm_c4"]), ("conf_cross", ["wm_c4", "wm_cross30"])):
    gens, traces, _ = runs[run]
    for arm in arms:
        T = [a for oid, a in traces.items() if gens[oid]["arm"] == arm]
        n_tail = n_all = 0
        s_head = s_tail = 0.0
        t_all, t_sim = [], []
        for a in T:
            tail, ev = split(a)
            n_tail += int(tail.sum()); n_all += tail.size
            s_head += float(ev[~tail].sum()); s_tail += float(ev[tail].sum())
            t_all.append(tstat(ev))
            e2 = ev.copy(); e2[tail] = 0.0                                       # tail tokens contribute nothing (unpushed ~ zero-mean)
            t_sim.append(tstat(e2))
        rec = {"texts": len(T), "tail_token_share": n_tail / n_all, "ev_per_head_token": s_head / (n_all - n_tail),
               "ev_per_tail_token": s_tail / max(1, n_tail), "tail_share_of_evidence": s_tail / (s_head + s_tail),
               "T_all_mean": float(np.mean(t_all)), "T_head_only_mean": float(np.mean(t_sim))}
        out[f"{run}/{arm}"] = rec
        print(f"{run + '/' + arm:26s} {len(T):5d} {rec['tail_token_share']:10.3f} {rec['ev_per_head_token']:13.4f} {rec['ev_per_tail_token']:13.4f} "
              f"{rec['tail_share_of_evidence']:22.3f} {rec['T_all_mean']:6.2f} {rec['T_head_only_mean']:12.2f}")
print("* generator-side statistic with the tail tokens' evidence set to zero: a projection of a head-only push, not a measurement.")

print("\n" + "=" * 100)
print("2. Are judged insertions / drift explained by PROMOTED TAIL tokens?  (promoted = pushed upward: beta_t > 0 and contrast > 0)")
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
keyphrase = {}
for ks in ("confirmation", "len10"):
    for k in json.load(open(f"assets/keys_{ks}.json"))["keys"]:
        keyphrase[k["key_id"]] = k["prefix"]


def is_content(s):
    w = s.strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]{2,}", w)) and w.lower() not in STOP


for run, arm in (("conf_cross", "wm_c4"), ("keylen10", "wm_c4")):
    gens, traces, flaws = runs[run]
    per_text = []
    cnt = {"tail": Counter(), "head": Counter()}
    n_tok = {"tail": [0, 0], "head": [0, 0], "all": [0, 0]}
    examples = []
    for oid, a in traces.items():
        g = gens[oid]
        if g["arm"] != arm or oid not in flaws:
            continue
        tail, ev = split(a)
        prom = (a["beta_t"] > 0) & (a["contrast"] > 0)
        strs = tok.convert_ids_to_tokens(a["ids"])
        strs = [tok.convert_tokens_to_string([s]) for s in strs]
        content = np.array([is_content(s) for s in strs])
        pt_c = prom & tail & content
        ph_c = prom & ~tail & content
        n = tail.size
        j = flaws[oid]
        flagged = bool({"unrelated_words_or_insertions", "drifts_off_topic"} & set(j["flaws"]))
        per_text.append({"flagged": flagged, "domain": g["domain"], "quality": j["quality"],
                         "prom_tail_content_per100": 100.0 * pt_c.sum() / n, "prom_head_content_per100": 100.0 * ph_c.sum() / n,
                         "prom_tail_per100": 100.0 * (prom & tail).sum() / n, "mean_kl": float(a["kl"].mean())})
        for nm, m in (("tail", prom & tail), ("head", prom & ~tail)):
            n_tok[nm][0] += int((m & content).sum()); n_tok[nm][1] += int(m.sum())
        n_tok["all"][0] += int(content.sum()); n_tok["all"][1] += n
        if flagged and len(examples) < 6:
            examples.append((keyphrase.get(g.get("assigned_key_id") or g.get("key_id"), "?"), g["domain"],
                             [strs[i].strip() for i in np.nonzero(pt_c)[0][:14]]))
    F = [t for t in per_text if t["flagged"]]; U = [t for t in per_text if not t["flagged"]]
    print(f"\n--- {run}/{arm}: {len(per_text)} judged texts, {len(F)} flagged for insertions or drift")
    rec = {"n": len(per_text), "n_flagged": len(F)}
    for k, label in (("prom_tail_content_per100", "promoted TAIL content words / 100 tokens"), ("prom_head_content_per100", "promoted HEAD content words / 100 tokens"),
                     ("prom_tail_per100", "promoted tail tokens (any) / 100 tokens"), ("mean_kl", "mean KL per token")):
        f, u = [t[k] for t in F], [t[k] for t in U]
        au = auroc(f, u)
        rec[k] = {"flagged": float(np.mean(f)), "unflagged": float(np.mean(u)), "auroc": au}
        print(f"   {label:42s} flagged {np.mean(f):6.2f}   not flagged {np.mean(u):6.2f}   AUROC {au:.2f}")
    q = np.array([t["quality"] for t in per_text]); x = np.array([t["prom_tail_content_per100"] for t in per_text]); h = np.array([t["prom_head_content_per100"] for t in per_text])
    rec["corr_quality_tail"] = float(np.corrcoef(q, x)[0, 1]); rec["corr_quality_head"] = float(np.corrcoef(q, h)[0, 1])
    print(f"   correlation of the judge's 1-5 score with promoted tail content words: {rec['corr_quality_tail']:+.2f}; with promoted head content words: {rec['corr_quality_head']:+.2f}")
    print(f"   content-word share: promoted tail tokens {n_tok['tail'][0] / max(1, n_tok['tail'][1]):.2f} | promoted head tokens {n_tok['head'][0] / max(1, n_tok['head'][1]):.2f} | all tokens {n_tok['all'][0] / n_tok['all'][1]:.2f}")
    for kp, dom, ws in examples:
        print(f"   e.g. [{dom}] key: {kp!r}\n        promoted tail content words: {', '.join(ws)}")
    out[f"insertions/{run}/{arm}"] = rec

print("\n" + "=" * 100)
print("3. Evidence along the text (held-out run, reference tokens, gated clipped evidence per token, by eighth of the text)")
z = np.load("runs/conf_cross/eval/posstats_confirmation.npz", allow_pickle=False)
kidx = {str(k): i for i, k in enumerate(z["key_ids"])}
off, h0, d = z["offsets"], z["h0"], z["d"]
prof = defaultdict(list); lens = defaultdict(list)
for i in range(len(z["arm"])):
    arm, dom, kid = str(z["arm"][i]), str(z["domain"][i]), str(z["assigned_key_id"][i])
    if arm not in ("base", "wm_c4") or kid not in kidx:
        continue
    a, b = int(off[i]), int(off[i + 1])
    if b - a < 64:
        continue
    w = gate_weights(h0[a:b], 0.2, 1.0); e = np.clip(d[kidx[kid], a:b], -CLIP, CLIP)
    edges = np.linspace(0, b - a, 9).astype(int)
    prof[(dom, arm)].append([float((w[s:t] * e[s:t]).sum() / max(1e-9, w[s:t].sum())) for s, t in zip(edges[:-1], edges[1:])])
    lens[(dom, arm)].append(b - a)
out["profile"] = {}
for dom in sorted({k[0] for k in prof}):
    m, c = np.mean(prof[(dom, "wm_c4")], axis=0), np.mean(prof[(dom, "base")], axis=0)
    diff = m - c
    out["profile"][dom] = {"marked_minus_clean_by_eighth": [float(x) for x in diff], "median_ref_tokens": float(np.median(lens[(dom, 'wm_c4')]))}
    print(f"   {dom:18s} median {int(np.median(lens[(dom, 'wm_c4')])):4d} ref tokens | marked - clean: " + " ".join(f"{x:+.3f}" for x in diff)
          + f" | first quarter carries {diff[:2].sum() / diff.sum():.0%}")
json.dump(out, open("runs/push_location_analysis.json", "w"), indent=1)
print("\nwrote runs/push_location_analysis.json")
