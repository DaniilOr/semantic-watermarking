"""SemStamp's paraphrase attacks (Hou et al. 2024, abehou/SemStamp paraphrase_gen.py): every sentence is paraphrased on its
own (NLTK sent_tokenize), so the number and order of sentences is kept.
  pegasus   tuner007/pegasus_paraphrase, 25 beams, 25 returned, max 60 tokens, repetition penalty 1.03
  parrot    prithivida/parrot_paraphraser_on_T5 ("paraphrase: " prefix), 25 beams, 25 returned.  The Parrot library's
            adequacy / fluency filters are not used (the library is not installed); the bigram variant's similarity
            check plays the adequacy role.
For each attacker both variants come from the same 25 candidates:
  <attacker>         the top beam
  <attacker>_bigram  among candidates whose similarity to the original sentence is within 0.03 of the best candidate's
                     and whose length is at most 1.5x the original's, the one with the LOWEST word-bigram overlap with the
                     original (accept_by_bigram_overlap); falls back to the top beam.  Similarity: all-mpnet-base-v2
                     (a general sentence encoder, deliberately not the watermark's encoder).
Writes runs/<run>_<attacker>[_bigram]/generations.shard<k>.jsonl (records of the attacked texts, parent fields kept).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.stages import collect_run_texts  # noqa: E402
from wmworld.textnorm import normalize_text, text_hash  # noqa: E402

MODELS = json.load(open("assets/ss_models.json"))
PATHS = {"pegasus": "/l/users/daniil.orel/hf_cache/hub/models--tuner007--pegasus_paraphrase/snapshots/0159e2949ca73657a2f1329898f51b7bb53b9ab2",
         "parrot": "/l/users/daniil.orel/hf_cache/hub/models--prithivida--parrot_paraphraser_on_T5/snapshots/9f32aa1e456e8e8a90d97e8673365f3090fa49fa"}
K = 25


def bigrams(s):
    w = s.lower().split()
    return set(zip(w, w[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--attacker", choices=list(PATHS), required=True)
    ap.add_argument("--arms", default="semstamp,chain_reference")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="sentences per generate call (x 25 beams)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PATHS[a.attacker])
    model = AutoModelForSeq2SeqLM.from_pretrained(PATHS[a.attacker]).to(dev).eval()
    sim = SentenceTransformer(MODELS["sentence-transformers/all-mpnet-base-v2"]["path"], device=dev)
    arms = set(a.arms.split(","))
    gens = [g for g in collect_run_texts(Path(a.run))[0] if g["arm"] in arms and g.get("status", "ok") == "ok" and g.get("text")
            and (not a.domains or g.get("domain") in a.domains)]
    gens = sorted(gens, key=lambda g: g["id"])[a.shard::a.num_shards]
    outs = {}
    for v in ("", "_bigram"):
        d = Path(f"{a.run}_{a.attacker}{v}"); d.mkdir(parents=True, exist_ok=True)
        p = d / f"generations.shard{a.shard}.jsonl"
        done = {json.loads(l)["parent_id"] for l in open(p)} if p.exists() else set()
        outs[v] = (p.open("a", encoding="utf-8"), done, d.name)
    todo = [g for g in gens if g["id"] not in outs[""][1] or g["id"] not in outs["_bigram"][1]]
    print(f"{a.run} {a.attacker} shard {a.shard}: {len(todo)} texts to attack ({len(gens) - len(todo)} done)", flush=True)
    gen_kw = dict(max_length=60 if a.attacker == "pegasus" else 96, num_beams=K, num_return_sequences=K, repetition_penalty=1.03)
    for n_done, g in enumerate(todo, 1):
        sents = [s for s in sent_tokenize(g["text"]) if s.strip()]
        plain, bigr = [], []
        for i in range(0, len(sents), a.batch):
            chunk = sents[i:i + a.batch]
            inp = [("paraphrase: " + s) if a.attacker == "parrot" else s for s in chunk]
            enc = tok(inp, truncation=True, max_length=60 if a.attacker == "pegasus" else 128, padding="longest", return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kw)
            cands = tok.batch_decode(out, skip_special_tokens=True)
            for j, s in enumerate(chunk):
                cs = [c.strip() for c in cands[j * K:(j + 1) * K]] or [s]
                plain.append(cs[0] or s)
                e = sim.encode([s] + cs, convert_to_tensor=True, normalize_embeddings=True)
                sc = (e[1:] @ e[0]).tolist(); best = max(sc)
                ok = [(c, x) for c, x in zip(cs, sc) if best - x <= 0.03 and c and len(c.split()) <= 1.5 * max(1, len(s.split()))]
                bo = bigrams(s)
                pick = min(ok, key=lambda cx: len(bo & bigrams(cx[0])) / max(1, len(bo)))[0] if ok else (cs[0] or s)
                bigr.append(pick)
        for v, ps in (("", plain), ("_bigram", bigr)):
            f, done, dname = outs[v]
            if g["id"] in done:
                continue
            text = normalize_text(" ".join(ps))
            rec = dict(g, text=text, text_hash=text_hash(text), stage=dname, parent_id=g["id"],
                       id="sp." + hashlib.sha256(f"{a.attacker}{v}|{g['id']}".encode()).hexdigest()[:20],
                       attack={"type": f"{a.attacker}{v}", "sentences": len(sents), "sentence_level": True})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
