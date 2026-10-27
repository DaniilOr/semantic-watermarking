"""PMark (online) next to SemStamp and Push + Margin on the same texts' datasets, clean and under SemStamp's sentence
paraphrase attacks: % detected at the human-null 1% / 0.1% thresholds (z_chain for SemStamp and push; IFEval has no
human null, so the plain model's 1% threshold is used there for all three).  PMark n = texts scored (coverage).
Usage: python scripts/pmark_compare.py PMARK_REPORT PMARK_RUN SEMSTAMP_REPORT SEMSTAMP_RUN DOMAIN [DOMAIN ...]"""
import json, sys
pmr, pmrun, ssr, ssrun, doms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
P = json.load(open(pmr))["results"]; S = json.load(open(ssr))["results"]
ATT = ["", "_pegasus", "_pegasus_bigram", "_parrot", "_parrot_bigram"]
def r(x):
    return "  -  " if not x else f"{100 * x['rate']:5.1f}"
print(f"{'dataset':18s} {'attack':15s} | {'SemStamp':>13s} | {'Push+Margin':>13s} | {'PMark':>13s} {'n':>5s}")
for d in doms:
    keys = ("model_1%", None) if d == "it_ifeval" else ("human_1%", "human_0.1%")
    for a in ATT:
        ss = S.get(f"{ssrun}{a}|semstamp|{d}", {}); pu = S.get(f"{ssrun}{a}|chain_reference|{d}", {}); pm = P.get(f"{pmrun}{a}|pmark|{d}", {})
        f = lambda rec, pre: f"{r(rec.get(pre + keys[0]))} / {r(rec.get(pre + keys[1])) if keys[1] else '  -  '}"
        print(f"{d:18s} {(a.strip('_') or 'clean'):15s} | {f(ss, 'z_chain:')} | {f(pu, 'z_chain:')} | {f(pm, '')} {pm.get('n', 0):5d}")
