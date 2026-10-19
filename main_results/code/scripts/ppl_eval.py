"""Perplexity of generated texts as SemStamp's paper reports it (abehou/SemStamp eval_clm.eval_perplexity): facebook/opt-2.7b,
per-text perplexity of the generated text on its own, averaged over texts ("sent_avg_ppl").  Also the perplexity of the
generated text conditioned on the excerpt it continues (context, not scored).
Usage: python scripts/ppl_eval.py --runs runs/A runs/B ... --arms base semstamp ... --out runs/ppl_X.json [--shard k --num-shards n]
"""

import argparse
import json
import statistics as S
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmworld.stages import collect_run_texts  # noqa: E402

OPT27 = "facebook/opt-2.7b"
PROMPTS = {}


def prompt_text(r):
    bench = "sp_bench" if r["domain"].startswith("sp_") else "ob_bench" if r["domain"].startswith("ob_") else "it_bench"
    if bench not in PROMPTS:
        PROMPTS[bench] = {}
        for l in open(ROOT / f"data/{bench}/prompts.jsonl", encoding="utf-8"):
            p = json.loads(l); PROMPTS[bench][p["prompt_id"]] = p
    p = PROMPTS[bench].get(r["prompt_id"], {})
    return p.get("excerpt") or ""                                         # the document text the generation continues


@torch.no_grad()
def nll_batch(model, tok, pairs, dev):
    """pairs: (context, text) -> mean NLL per text over the text's tokens only."""
    out = []
    enc = []
    for ctx, txt in pairs:
        c = tok(ctx, add_special_tokens=False).input_ids if ctx else []
        t = tok(txt, add_special_tokens=False).input_ids
        ids = [tok.bos_token_id] + c + t                                  # OPT's tokenizer prepends </s> by default
        enc.append((ids[-1024:], max(1, len(ids[-1024:]) - len(t))))
    L = max(len(i) for i, _ in enc)
    ids = torch.full((len(enc), L), tok.pad_token_id); att = torch.zeros((len(enc), L), dtype=torch.long)
    for j, (i, _) in enumerate(enc):
        ids[j, :len(i)] = torch.tensor(i); att[j, :len(i)] = 1
    logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].to(dev)[..., None])[..., 0]
    for j, (i, start) in enumerate(enc):
        s = lp[j, start - 1:len(i) - 1]
        out.append(float(-s.mean()) if s.numel() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(OPT27)
    model = AutoModelForCausalLM.from_pretrained(OPT27, torch_dtype=torch.float16).to(dev).eval()
    res = defaultdict(lambda: {"ppl": [], "ppl_cond": []})
    for run in a.runs:
        recs = [r for r in collect_run_texts(Path(run))[0] if r.get("arm") in a.arms and r.get("status", "ok") == "ok" and r.get("text")
                and (not a.domains or r.get("domain") in a.domains) and r.get("domain") != "sp_mbpp"]
        recs = list({(r["arm"], r["prompt_id"]): r for r in recs}.values())
        recs.sort(key=lambda r: len(r["text"]))
        for i in range(0, len(recs), a.batch):
            chunk = recs[i:i + a.batch]
            u = nll_batch(model, tok, [("", r["text"]) for r in chunk], dev)
            c = nll_batch(model, tok, [(prompt_text(r), r["text"]) for r in chunk], dev)
            for r, x, y in zip(chunk, u, c):
                k = f"{Path(run).name}|{r['arm']}|{r['domain']}"
                if x is not None:
                    res[k]["ppl"].append(float(torch.exp(torch.tensor(x))))
                if y is not None:
                    res[k]["ppl_cond"].append(float(torch.exp(torch.tensor(y))))
        print(f"{run}: {len(recs)} texts", flush=True)
    out = {k: {"n": len(v["ppl"]), "ppl_mean": S.mean(v["ppl"]), "ppl_median": S.median(v["ppl"]),
               "ppl_cond_mean": S.mean(v["ppl_cond"]) if v["ppl_cond"] else None,
               "ppl_cond_median": S.median(v["ppl_cond"]) if v["ppl_cond"] else None} for k, v in res.items() if v["ppl"]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in sorted(out.items()):
        print(f"{k:55s} n {v['n']:5d} ppl {v['ppl_mean']:7.2f} (median {v['ppl_median']:6.2f})  cond {v['ppl_cond_mean']:7.2f}")


if __name__ == "__main__":
    main()
