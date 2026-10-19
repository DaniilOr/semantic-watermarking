"""SemStamp's Parrot attack as their code runs it (abehou/SemStamp paraphrase_gen.py parrot_paraphrase + SParrot.augment,
Parrot library filters; copies in third_party/semstamp_ref), batched.  The milder top-beam Parrot of
scripts/sent_paraphrase_attack.py is left unchanged.
Per sentence (NLTK sent_tokenize; sentence count and order kept):
  input cleaned to [a-zA-Z0-9 ?'-/:.] and prefixed "paraphrase: "; group beam search, 10 beams in 2 groups, diversity
  penalty 2.0, max_length 60 (+32 if the sentence has >= 60 characters); outputs lowercased, stripped to [a-zA-Z0-9 ?'-],
  deduplicated; adequacy filter (prithivida/parrot_adequacy_model, P(true) >= 0.8; all kept if none pass); fluency filter
  (prithivida/parrot_fluency_model, P(good) >= 0.8; previous set kept if none pass); ranked by Levenshtein distance to the
  lowercased prefixed input, most different first; each through well_formed_sentence(end_sent=True); empty candidates
  are dropped (the original sentence is kept if none is left) -- the only deviation from their code.
  parrotss         the first (most different) candidate
  parrotss_bigram  accept_by_bigram_overlap: lowest OPT-token bigram overlap among candidates at most 1.5x the original's
                   token length whose BERTScore F1 (microsoft/deberta-xlarge-mnli, baseline-rescaled) is within
                   0.03 x the first candidate's score
Writes runs/<run>_parrotss[_bigram]/generations.shard<k>.jsonl (records of the attacked texts, parent fields kept).
"""

import os

os.environ.setdefault("HF_HUB_CACHE", "/l/users/daniil.orel/hf_cache/hub")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from itertools import groupby  # noqa: E402
from pathlib import Path  # noqa: E402
from string import punctuation  # noqa: E402

import bert_score  # noqa: E402
import Levenshtein  # noqa: E402
import torch  # noqa: E402
from nltk import sent_tokenize  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.stages import collect_run_texts  # noqa: E402
from wmworld.textnorm import normalize_text, text_hash  # noqa: E402

PARROT = "/l/users/daniil.orel/hf_cache/hub/models--prithivida--parrot_paraphraser_on_T5/snapshots/9f32aa1e456e8e8a90d97e8673365f3090fa49fa"
MODELS = json.load(open("assets/ss_models.json"))
BEAMS, GROUPS, ADEQ, FLU, BERT_T = 10, 2, 0.8, 0.8, 0.03
PUNCTS = "!.?"


def well_formed_sentence(sent, end_sent=False):                      # sampling_utils.well_formed_sentence
    sent = sent[0].upper() + sent[1:] if sent else sent
    sent = sent.replace("  ", " ").replace(" i ", " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    punc = set(punctuation) - set("."); punc.add("\n")
    out = []
    for k, g in groupby(sent):
        out.extend([k] if k in punc else list(g))
    return "".join(out)


def bigram_overlap(a, b):                                             # build_bigrams + compare_ngram_overlap
    ia = Counter(zip(a, a[1:])); pb = Counter(zip(b, b[1:]))
    return sum(pb[k] for k in ia.keys() & pb.keys())


class SemStampParrot:
    def __init__(self, dev):
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(PARROT)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(PARROT).to(dev).eval()
        self.atok = AutoTokenizer.from_pretrained("prithivida/parrot_adequacy_model")
        self.amod = AutoModelForSequenceClassification.from_pretrained("prithivida/parrot_adequacy_model").to(dev).eval()
        self.ftok = AutoTokenizer.from_pretrained("prithivida/parrot_fluency_model")
        self.fmod = AutoModelForSequenceClassification.from_pretrained("prithivida/parrot_fluency_model", num_labels=2).to(dev).eval()
        self.opt_tok = AutoTokenizer.from_pretrained(MODELS["opt13b_ss"]["path"])
        base = os.path.join(os.path.dirname(bert_score.__file__), "rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv")
        self.bs = bert_score.BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40, rescale_with_baseline=True,
                                        baseline_path=base, lang="en", device=dev)
        self.bs._tokenizer.model_max_length = 512                       # deberta's tokenizer reports an unbounded length

    @torch.no_grad()
    def candidates(self, sent):
        max_length = 60 + (32 if len(sent) >= 60 else 0)
        inp = "paraphrase: " + re.sub(r"[^a-zA-Z0-9 \?\'\-\/\:\.]", "", sent)
        ids = self.tok.encode(inp, return_tensors="pt").to(self.dev)
        out = self.model.generate(ids, do_sample=False, max_length=max_length, num_beams=BEAMS, num_beam_groups=GROUPS,
                                  diversity_penalty=2.0, early_stopping=True, num_return_sequences=BEAMS,
                                  trust_remote_code=True)          # group beam search lives in transformers-community/group-beam-search
        paras = list(dict.fromkeys(re.sub(r"[^a-zA-Z0-9 \?\'\-]", "", self.tok.decode(o, skip_special_tokens=True).lower())
                                   for o in out))
        x = self.atok([inp] * len(paras), paras, return_tensors="pt", max_length=128, truncation=True, padding=True).to(self.dev)
        adeq = self.amod(**x).logits.softmax(dim=1)[:, 1].tolist()
        a = [p for p, s in zip(paras, adeq) if s >= ADEQ] or paras
        x = self.ftok(["Sentence: " + p for p in a], return_tensors="pt", truncation=True, padding=True).to(self.dev)
        flu = self.fmod(**x).logits.softmax(dim=1)[:, 1].tolist()
        f = [p for p, s in zip(a, flu) if s >= FLU] or a
        f = sorted(f, key=lambda p: Levenshtein.distance(inp.lower(), p), reverse=True)    # stable, like list.sort
        out = [well_formed_sentence(p, end_sent=True) for p in f]
        return [p for p in out if p.strip()] or [sent]                   # empty candidates dropped (deviation; BERTScore
                                                                          # cannot score them with current transformers)

    def bigram_pick(self, sent, paras):                                # accept_by_bigram_overlap, bert_threshold 0.03
        ids = self.opt_tok(sent).input_ids
        pids = [self.opt_tok(p).input_ids for p in paras]
        scores = self.bs.score(paras, [sent] * len(paras))[2].tolist()
        max_score, best, min_ov = scores[0], paras[0], len(ids)
        for p, pi, sc in zip(paras, pids, scores):
            ov = bigram_overlap(ids, pi)
            if ov < min_ov and len(pi) <= 1.5 * len(ids) and (max_score - sc) <= BERT_T * max_score:
                min_ov, best = ov, p
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="semstamp,chain_reference")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N texts of this shard")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = SemStampParrot(dev)
    arms = set(a.arms.split(","))
    gens = [g for g in collect_run_texts(Path(a.run))[0] if g["arm"] in arms and g.get("status", "ok") == "ok" and g.get("text")
            and (not a.domains or g.get("domain") in a.domains)]
    gens = sorted(gens, key=lambda g: g["id"])[a.shard::a.num_shards]
    if a.limit:
        gens = gens[:a.limit]
    outs = {}
    for v in ("", "_bigram"):
        d = Path(f"{a.run}_parrotss{v}"); d.mkdir(parents=True, exist_ok=True)
        p = d / f"generations.shard{a.shard}.jsonl"
        done = {json.loads(l)["parent_id"] for l in open(p)} if p.exists() else set()
        outs[v] = (p.open("a", encoding="utf-8"), done, d.name)
    todo = [g for g in gens if g["id"] not in outs[""][1] or g["id"] not in outs["_bigram"][1]]
    print(f"{a.run} parrotss shard {a.shard}: {len(todo)} texts to attack ({len(gens) - len(todo)} done)", flush=True)
    for n_done, g in enumerate(todo, 1):
        sents = [s for s in sent_tokenize(g["text"]) if s.strip()]
        plain, bigr, n_cand = [], [], []
        for s in sents:
            cs = P.candidates(s)
            n_cand.append(len(cs))
            plain.append(cs[0])
            bigr.append(P.bigram_pick(s, cs))
        for v, ps in (("", plain), ("_bigram", bigr)):
            f, done, dname = outs[v]
            if g["id"] in done:
                continue
            text = normalize_text(" ".join(ps))
            rec = dict(g, text=text, text_hash=text_hash(text), stage=dname, parent_id=g["id"],
                       id="sp." + hashlib.sha256(f"parrotss{v}|{g['id']}".encode()).hexdigest()[:20],
                       attack={"type": f"parrotss{v}", "sentences": len(sents), "sentence_level": True,
                               "candidates_per_sentence": sum(n_cand) / max(1, len(n_cand))})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
