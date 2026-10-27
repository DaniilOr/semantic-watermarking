"""PMark (Huo et al., ICLR 2026; github.com/Z1zs/PMark, cloned to third_party/pmark_ref) on our pipeline: online
multi-channel PMark with the released settings.  The reference code is NOT changed and not imported here (it needs
vLLM, which our environment does not have); the functions below re-implement it line for line, and
scripts/pmark_check.py checks them against the reference functions on random inputs.

Generation (reference utils/sample.py sample_next_sentence_msignal, parallel=False, median_method="hd"):
  for sentence t = 1, 2, ... of the generated text: draw N = 64 candidate next sentences from the model (same prompts,
  chat template and sampling as every other arm; one sentence each, our sentence-end rule); embed them with
  all-mpnet-base-v2; proxies F_j = cosine with b = 4 orthonormal pivots (QR of a seed-42 Gaussian matrix,
  utils/rand.get_random_vectors); for channel j = 1..b keep the survivors at or above (key bit 1) / below (bit 0) their
  Harrell-Davis median; keep one survivor uniformly at random.  Key bits: secret_mbit.json in groups of b, group t for
  sentence t (utils/rand.SecretBits.set_signum).  Retry rule as released: up to 5 draws of N until more than 10 usable.
Detection (reference utils/detect.py within_green_msig + watermark_z_test, online "hd"):
  for every sentence t >= 1, resample N candidates from the model given the text before it, take each channel's
  Harrell-Davis median, count c_tj = 1 if the sentence's proxy is on the key's side (within delta), else
  exp(-K |F_j - median|); z = (sum c - 0.5 n) / sqrt(0.25 n), n = b T.  K = 250, delta = 0.001 (the released defaults;
  the paper's text says K = 150).  Raw proxies and medians are stored, so z can be recomputed for any K / delta.
Sentence units: leading line breaks join the next sentence (semwm.LEAD_JOIN) in every PMark run, so the generator's
sentence positions are the ones the detector's split_sentences sees (PMark's key is positional).

Usage:
  python scripts/pmark_arm.py gen <scripts/steered_semstamp.py arguments, with --arm semstamp --arm-label pmark>
  python scripts/pmark_arm.py score --target llama3b --prompts sp_bench --texts RUN:ARM[,ARM] ... --null runs/sp_bench_null
         --domains sp_realnews ... --max-new 200 --cache runs/pmark_scores/NAME [--plain semstamp ...] --shard k --num-shards n
  python scripts/pmark_arm.py report --cache runs/pmark_scores/NAME --null sp_bench_null --out runs/pmark_detect_NAME.json
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.textnorm import seed_from  # noqa: E402

REF = Path(__file__).resolve().parents[1] / "third_party" / "pmark_ref"
MPNET = "/l/users/daniil.orel/hf_cache/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130"
N, B, MIN_SEQS, K, DELTA, PIVOT_SEED = 64, 4, 10, 250.0, 0.001, 42
PARAMS = {"method": "pmark_online", "N": N, "b": B, "min_seqs": MIN_SEQS, "median": "harrell_davis", "pivot": "rand_qr_seed42",
          "embedder": "sentence-transformers/all-mpnet-base-v2", "key": "third_party/pmark_ref/secret_mbit.json", "K": K, "delta": DELTA,
          "lead_join": True}


# --------------------------------------------------------------------------- #
# PMark's key, pivots, median, channel filter and soft count (reference utils/rand.py, utils/sample.py, utils/detect.py)
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _bit_groups(b: int) -> tuple:
    data = json.loads((REF / "secret_mbit.json").read_text())
    return tuple(tuple(int(x) for x in data[i:i + b]) for i in range(0, (len(data) // b) * b, b))


def key_bits(t: int, b: int = B) -> list[int]:
    """SecretBits.set_signum(b)[t]: bits t*b .. t*b+b-1 of secret_mbit.json (sentence t; t = 0 is the prompt)."""
    return list(_bit_groups(b)[t])


def pivots(dim: int, b: int = B, seed: int = PIVOT_SEED) -> np.ndarray:
    """get_random_vectors: Q.T of the QR decomposition of a (dim, b) standard normal matrix (numpy default_rng(seed))."""
    A = np.random.default_rng(seed).standard_normal((dim, b))
    Q, _ = np.linalg.qr(A)
    return Q.T


def hd_median(x, q: float = 0.5) -> float:
    """harrell_davis_median, verbatim."""
    from scipy.stats import beta
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 1:
        return float(x[0])
    i = np.arange(1, n + 1)
    w = beta.cdf(i / n, (n + 1) * q, (n + 1) * (1 - q)) - beta.cdf((i - 1) / n, (n + 1) * q, (n + 1) * (1 - q))
    return float(np.sum(w * x))


def cosines(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """get_cosine_similarities(E, P) -> [n, b], in float64 as the reference (float32 embeddings x float64 pivots)."""
    E = np.asarray(E, dtype=np.float64); P = np.asarray(P, dtype=np.float64)
    num = E @ P.T
    den = np.maximum(np.linalg.norm(E, axis=1)[:, None] * np.linalg.norm(P, axis=1)[None, :], 1e-8)
    return num / den


def channel_filter(F: np.ndarray, bits: list[int], rng: random.Random) -> tuple[list[int], list[float]]:
    """The online sequential multi-channel filter of sample_next_sentence_msignal (parallel=False), with its fallbacks."""
    n, b = F.shape
    alive, meds = list(range(n)), []
    for j in range(b):
        if not alive:
            break
        vals = F[alive, j]
        m = hd_median(vals)
        meds.append(float(m))
        keep = (vals >= m) if bits[j] else (vals < m)
        if keep.sum() == 0:
            keep = (vals > m) if bits[j] else (vals <= m)
        if keep.sum() == 0:
            eq = [alive[k] for k in range(len(alive)) if vals[k] == m]
            if eq:
                rng.shuffle(eq)
                chosen = set(eq[:max(1, len(eq) // 2)])
                keep = np.array([alive[k] in chosen for k in range(len(alive))], dtype=bool)
            else:
                keep = np.zeros(len(alive), dtype=bool); keep[rng.randrange(len(alive))] = True
        alive = [alive[k] for k in range(len(alive)) if keep[k]]
    return alive, meds


def soft_counts(F_row, med, bits, K: float = K, delta: float = DELTA) -> list[float]:
    """within_green_msig's soft count per channel."""
    out = []
    for sm, cs, rf in zip(med, F_row, bits):
        if sm is None or cs is None:
            out.append(1.0)
        elif rf and cs >= sm - delta:
            out.append(1.0)
        elif (not rf) and cs <= sm + delta:
            out.append(1.0)
        else:
            out.append(math.exp(-K * abs(cs - sm)) if K is not None else 0.0)
    return out


def z_of(counts: list[float]) -> float | None:
    """watermark_z_test: (x - 0.5 n) / sqrt(0.25 n)."""
    n = len(counts)
    return None if n == 0 else (sum(counts) / n - 0.5) / math.sqrt(0.25 / n)


class Proxy:
    """all-mpnet-base-v2 (SentenceTransformer, as the reference) + the pivots."""

    def __init__(self, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MPNET, device=device)
        self.P = pivots(self.model.get_sentence_embedding_dimension())
        self.n_calls = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.n_calls += 1
        E = self.model.encode(texts, convert_to_tensor=True, batch_size=128, show_progress_bar=False).float().cpu().numpy()
        return cosines(E, self.P)


# --------------------------------------------------------------------------- #
# generation: replaces steered_semstamp.gen_sentencewise for --arm semstamp --arm-label pmark
# --------------------------------------------------------------------------- #

_PROXY = None


def gen_pmark(target, drafter, ctx, prompt, key, planes, emb, etok, sampling, seed, min_new, max_new, a):
    import torch
    import steered_semstamp as ss
    global _PROXY
    if _PROXY is None:
        _PROXY = Proxy(str(target.device))
    with torch.no_grad():
        ids: list[int] = []
        st = {"pmark": PARAMS, "candidates": 0, "draws": 0, "accepted": 0, "n_capped": 0, "end_checks": 0, "few_candidates": 0,
              "sentence_trace": []}
        ended, t = False, 0
        while len(ids) < max_new and not ended:
            if a.eot_at_boundary and ids and len(ids) >= min_new:       # the model may end the text here (as gen_sentencewise)
                torch.manual_seed(seed_from(seed, "end", t))
                t1 = ss._generate(target, ctx + ids, sampling, 1, 0)[0]
                st["end_checks"] += 1
                if t1 and t1[0] in target.eos_ids:
                    st["ended_at_boundary"] = True
                    break
            t += 1
            bits = key_bits(t)
            L0 = len(ctx) + len(ids)
            valid = []
            for attempt in range(5):
                torch.manual_seed(seed_from(seed, "pmark", t, attempt))
                stop = semwm.SentenceEndBatch(target.tokenizer, L0, target.eos_ids)
                rows = ss._generate(target, ctx + ids, sampling, min(a.sent_max_tokens, max_new - len(ids)), 1 if a.eot_at_boundary else 2,
                                    None, stop, N)
                st["draws"] += 1; st["candidates"] += len(rows)
                valid = []
                for row in rows:
                    raw, saw_eos = ss.clean_ids(target, row)
                    cand, piece, found = semwm.trim_candidate(target.tokenizer, raw, [])
                    if not cand or not piece:
                        continue
                    if saw_eos and not found and len(ids) + len(cand) < min_new:
                        continue                                        # too short to end here
                    valid.append((cand, piece, saw_eos and (not found or (a.eot_at_boundary and len(cand) == len(raw))), found or saw_eos))
                if len(valid) > MIN_SEQS:
                    break
            if not valid:
                st["no_candidates"] = True
                break
            if len(valid) <= MIN_SEQS:
                st["few_candidates"] += 1
            F = _PROXY([v[1] for v in valid])
            rng = random.Random(seed_from(seed, "pmark_pick", t))
            alive, meds = channel_filter(F, bits, rng)
            i = rng.choice(alive) if alive else rng.randrange(len(valid))
            cand, piece, ends, complete = valid[i]
            if not complete:
                st["n_capped"] += 1
            ids += cand
            st["accepted"] += 1
            st["sentence_trace"].append({"t": t, "n_valid": len(valid), "n_alive": len(alive), "proxy": [round(float(x), 6) for x in F[i]],
                                         "step_medians": [round(m, 6) for m in meds], "n_tok": len(cand)})
            ended = ends
        st["n_tokens"] = len(ids)
        st["candidates_per_sentence"] = st["candidates"] / max(1, st["accepted"])
        return ids, st


def run_gen(argv: list[str]):
    if "--arm" not in argv or argv[argv.index("--arm") + 1] != "semstamp" or "--arm-label" not in argv:
        sys.exit("gen: pass --arm semstamp --arm-label pmark (the PMark loop replaces SemStamp's sentence loop)")
    if "--join-leading-newline" not in argv:
        argv = argv + ["--join-leading-newline"]
    import steered_semstamp as ss
    ss.gen_sentencewise = gen_pmark
    sys.argv = [sys.argv[0]] + argv
    ss.main()


# --------------------------------------------------------------------------- #
# detection, step 1 (GPU): per text and sentence, the sentence's proxies and the resampled Harrell-Davis medians
# --------------------------------------------------------------------------- #

def sentence_offsets(text: str) -> list[tuple[int, str]]:
    """(start offset of the prefix before sentence k, sentence k) exactly as semwm.split_sentences segments the text."""
    text = semwm.normalize_text(text)
    out, start = [], 0
    for m in semwm.BOUNDARY_RE.finditer(text):
        seg = text[start:m.end()].strip()
        if seg:
            out.append((start, seg))
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append((start, tail))
    return out


def load_prompts(bench: str) -> dict:
    return {r["prompt_id"]: SimpleNamespace(**r) for r in map(json.loads, open(f"data/{bench}/prompts.jsonl"))}


def collect(specs: list[str], null: str | None, domains: set) -> list[dict]:
    """Texts to score: RUN:ARM[,ARM] (all arms if no ':') plus the human null run (arm 'base')."""
    from wmworld.stages import collect_run_texts
    out = []
    for spec in specs + ([f"{null}:base"] if null else []):
        run, _, arms = spec.partition(":")
        want = set(arms.split(",")) if arms else None
        if not Path(run).exists():
            print(f"skip missing run {run}", flush=True)
            continue
        for g in collect_run_texts(Path(run))[0]:
            if want and g["arm"] not in want:
                continue
            if g.get("status", "ok") == "ok" and g.get("text") and g.get("domain") in domains:
                out.append({"run": Path(run).name, "arm": g["arm"], "domain": g["domain"], "prompt_id": g["prompt_id"],
                            "text_hash": g["text_hash"], "text": g["text"], "null": run == null})
    return out


def run_score(argv: list[str]):
    import torch
    import steered_semstamp as ss
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--models-json", default="assets/ss_models.json")
    ap.add_argument("--prompts", required=True, help="sp_bench | ob_bench | it_bench (data/<x>/prompts.jsonl, all splits)")
    ap.add_argument("--texts", nargs="+", required=True, help="RUN[:ARM,ARM]")
    ap.add_argument("--null", default=None)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--max-new", type=int, required=True, help="the generator's text cap (candidates are capped at min(64, what is left))")
    ap.add_argument("--min-new-tokens", type=int, default=2, help="HF min_new_tokens of the generator's candidate draws (1 with --eot-at-boundary)")
    ap.add_argument("--plain", default="", choices=["", "semstamp"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--rep-penalty", type=float, default=None)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--sent-max-tokens", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=256, help="rows per generate call (contexts x N)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="main", help="name of this text selection's group file (several selections can share a cache)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first texts per (run, arm, domain)")
    a = ap.parse_args(argv)
    semwm.LEAD_JOIN = True
    from wmworld.config import load_config
    cfg = load_config(a.config)
    sampling = cfg.generation_config(None, {}).sampling()
    if a.temperature or a.top_p or a.top_k is not None or a.rep_penalty:     # same rule as steered_semstamp.main
        sampling = SimpleNamespace(temperature=a.temperature or sampling.temperature, top_p=a.top_p or sampling.top_p,
                                   top_k=sampling.top_k, hf_top_k=a.top_k, repetition_penalty=a.rep_penalty)
    prompts = load_prompts(a.prompts)
    items = collect(a.texts, a.null, set(a.domains))
    if a.limit:
        per = defaultdict(int); keep = []
        for it in sorted(items, key=lambda r: (r["run"], r["arm"], r["domain"], r["prompt_id"])):
            k = (it["run"], it["arm"], it["domain"])
            if per[k] < a.limit:
                keep.append(it); per[k] += 1
        items = keep
    # one score per (text, prompt): attacked copies with an unchanged text are scored once
    by_key = {}
    for it in items:
        by_key.setdefault(f"{it['text_hash']}|{it['prompt_id']}", it)
    keys = sorted(by_key)
    mine = [k for i, k in enumerate(keys) if i % a.num_shards == a.shard]
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"scores.shard{a.shard}.jsonl"
    if out_path.exists():                                               # a lane stopped mid-write leaves a partial last line
        raw = out_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            out_path.write_bytes(raw[: raw.rfind(b"\n") + 1])
    done = set()
    for f in glob.glob(str(cache / "scores.shard*.jsonl")):
        for ln in open(f):
            try:
                done.add(json.loads(ln)["key"])
            except Exception:
                pass
    todo = [k for k in mine if k not in done]
    print(f"[score shard {a.shard}] {len(items)} texts, {len(keys)} distinct, mine {len(mine)}, to do {len(todo)}", flush=True)
    # membership of every (run, arm, domain) group, written once per shard so the report needs no generation files
    (cache / f"groups.{a.tag}.shard{a.shard}.json").write_text(json.dumps(
        [{"key": f"{it['text_hash']}|{it['prompt_id']}", "run": it["run"], "arm": it["arm"], "domain": it["domain"], "null": it["null"]}
         for it in items]))
    (cache / "config.json").write_text(json.dumps({**PARAMS, **{k: v for k, v in vars(a).items() if k not in ("shard", "num_shards")},
                                                   "sampling": {"temperature": sampling.temperature, "top_p": sampling.top_p,
                                                                "hf_top_k": getattr(sampling, "hf_top_k", None),
                                                                "repetition_penalty": getattr(sampling, "repetition_penalty", None)}}, indent=1))
    if not todo:
        return
    m = json.loads(Path(a.models_json).read_text())
    target = ss.LM(m[a.target]["path"], a.dtype)
    proxy = Proxy(str(target.device))
    tok = target.tokenizer
    t0 = time.time()
    fh = open(out_path, "a")
    def one(k):
        it = by_key[k]
        p = prompts[it["prompt_id"]]
        if a.plain:
            ptext = ss.semstamp_prompt(p.excerpt + p.human_continuation) if not it["null"] else p.excerpt
            ctx = target.plain_prefix_ids(ptext)
        else:
            ctx = target.chat_prefix_ids(ss.GENERATOR_SYSTEM, p.prompt)
        text = semwm.normalize_text(it["text"])
        sents = sentence_offsets(text)
        jobs = []                                                       # (sentence index, context ids, room)
        for si, (off, _) in enumerate(sents):
            pre = [int(x) for x in tok(text[:off], add_special_tokens=False)["input_ids"]] if off > 0 else []
            left = a.max_new - len(pre)
            jobs.append((si, ctx + pre, min(a.sent_max_tokens, left) if left > 0 else a.sent_max_tokens))
        samples = defaultdict(list)                                     # sentence index -> candidate texts
        per_call = max(1, a.max_rows // N)
        by_room = defaultdict(list)
        for j in jobs:
            by_room[j[2]].append(j)
        for room, js in sorted(by_room.items()):
            for c in range(0, len(js), per_call):
                chunk = js[c:c + per_call]
                torch.manual_seed(seed_from("pmark_detect", k, room, c))
                for si, texts in zip([j[0] for j in chunk], sample_sentences(target, [j[1] for j in chunk], sampling, room, a.min_new_tokens)):
                    samples[si] += texts
        for si, cids, room in jobs:                                     # the reference's retry: more than 10 usable, up to 3 draws
            for r in range(2):
                if len(samples[si]) > MIN_SEQS:
                    break
                torch.manual_seed(seed_from("pmark_detect_retry", k, si, r))
                samples[si] = sample_sentences(target, [cids], sampling, room, a.min_new_tokens)[0]
        Fs = proxy([s for _, s in sents]) if sents else np.zeros((0, B))
        rec = {"key": k, "n_sent": len(sents), "proxy": [], "median": [], "n_samples": []}
        for si in range(len(sents)):
            S = samples[si]
            rec["n_samples"].append(len(S))
            if not S:
                rec["median"].append(None)
            else:
                G = proxy(S)
                rec["median"].append([round(hd_median(G[:, j]), 7) for j in range(B)])
            rec["proxy"].append([round(float(x), 7) for x in Fs[si]])
        return rec

    n_skip = 0
    for n_done, k in enumerate(todo, 1):
        try:
            rec = one(k)
        except torch.OutOfMemoryError:                                  # even single-context half batches did not fit
            torch.cuda.empty_cache(); n_skip += 1
            print(f"[score shard {a.shard}] skipped {k} (CUDA OOM after splitting); {n_skip} skipped so far", flush=True)
            continue
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        if n_done % 10 == 0 or n_done == len(todo):
            print(f"[score shard {a.shard}] {n_done}/{len(todo)} ({time.time() - t0:.0f}s, {(time.time() - t0) / n_done:.1f} s/text)", flush=True)
    fh.close()


def sample_sentences(target, ctxs: list[list[int]], sampling, room: int, min_new_tokens: int, n: int = N) -> list[list[str]]:
    """n exact draws of the next sentence for each context.  On CUDA OOM the call is split: first the contexts, then the
    n draws of a single context (the draws are i.i.d., so splitting them does not change what is sampled)."""
    import torch
    try:
        return _sample_batch(target, ctxs, sampling, room, min_new_tokens, n)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(ctxs) > 1:
            h = len(ctxs) // 2
            return (sample_sentences(target, ctxs[:h], sampling, room, min_new_tokens, n)
                    + sample_sentences(target, ctxs[h:], sampling, room, min_new_tokens, n))
        if n > 4:
            first = sample_sentences(target, ctxs, sampling, room, min_new_tokens, n // 2)
            return [first[0] + sample_sentences(target, ctxs, sampling, room, min_new_tokens, n - n // 2)[0]]
        raise


def _sample_batch(target, ctxs: list[list[int]], sampling, room: int, min_new_tokens: int, n: int) -> list[list[str]]:
    """n draws per context in one generate call, batched with left padding (same kwargs as the generator)."""
    import torch
    from transformers import StoppingCriteriaList
    import steered_semstamp as ss
    L = max(len(c) for c in ctxs)
    inp = torch.full((len(ctxs), L), target.pad_id, dtype=torch.long)
    att = torch.zeros((len(ctxs), L), dtype=torch.long)
    for r, c in enumerate(ctxs):
        inp[r, L - len(c):] = torch.tensor(c); att[r, L - len(c):] = 1
    kw = dict(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p, min_new_tokens=min_new_tokens, max_new_tokens=room,
              eos_token_id=target.eos_ids, pad_token_id=target.pad_id, num_return_sequences=n,
              stopping_criteria=StoppingCriteriaList([semwm.SentenceEndBatch(target.tokenizer, L, target.eos_ids)]))
    if getattr(sampling, "hf_top_k", None) is not None:
        kw["top_k"] = sampling.hf_top_k
    if getattr(sampling, "repetition_penalty", None):
        kw["repetition_penalty"] = sampling.repetition_penalty
    with torch.no_grad():
        out = target.model.generate(input_ids=inp.to(target.device), attention_mask=att.to(target.device), **kw)
    res = []
    for r in range(len(ctxs)):
        texts = []
        for row in out[r * n:(r + 1) * n]:
            raw, _ = ss.clean_ids(target, [int(x) for x in row[L:].tolist()])
            cand, piece, _ = semwm.trim_candidate(target.tokenizer, raw, [])
            if cand and piece:
                texts.append(piece)
        res.append(texts)
    return res


# --------------------------------------------------------------------------- #
# detection, step 2 (CPU): z per text, thresholds on the human null, TPR / AUROC per (run, arm, domain)
# --------------------------------------------------------------------------- #

def run_report(argv: list[str]):
    from wmworld.posstats import auroc, upper_tail_threshold, wilson
    from wmworld.records import write_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--null", required=True, help="name of the human null run (e.g. sp_bench_null)")
    ap.add_argument("--model-null-run", default=None, help="run whose arm 'base' gives the model_1%% threshold")
    ap.add_argument("--K", type=float, default=K)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--align", nargs="*", default=[], help="RUN:ARM of PMark generations: compare their sentence trace with the scores")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    recs = {}
    for f in sorted(glob.glob(str(cache / "scores.shard*.jsonl"))):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            recs[r["key"]] = r
    groups = defaultdict(set)
    for f in sorted(glob.glob(str(cache / "groups*.json"))):
        for g in json.loads(Path(f).read_text()):
            groups[(g["run"], g["arm"], g["domain"])].add(g["key"])
    def z(key):
        r = recs.get(key)
        if r is None:
            return None
        c = []
        for t in range(r["n_sent"]):
            c += soft_counts(r["proxy"][t], r["median"][t] or [None] * B, key_bits(t + 1), a.K, a.delta)
        return z_of(c)
    report = {"method": PARAMS, "K": a.K, "delta": a.delta, "thresholds": {}, "results": {}, "coverage": {}}
    domains = sorted({d for (_, _, d) in groups})
    for d in domains:
        hv = [x for x in (z(k) for k in groups.get((a.null, "base", d), ())) if x is not None]
        mv = [x for x in (z(k) for k in groups.get((a.model_null_run, "base", d), ())) if x is not None] if a.model_null_run else []
        report["thresholds"][d] = {"human_1%": upper_tail_threshold(hv, 0.01) if hv else None, "human_0.1%": upper_tail_threshold(hv, 0.001) if hv else None,
                                   "model_1%": upper_tail_threshold(mv, 0.01) if mv else None, "n_human": len(hv), "n_model": len(mv), "_hv": hv}
    for (run, arm, d), ks in sorted(groups.items()):
        report["coverage"][f"{run}|{arm}|{d}"] = {"n": len(ks), "scored": sum(k in recs for k in ks)}
        if run == a.null:
            continue
        v = [x for x in (z(k) for k in ks) if x is not None]
        th = report["thresholds"][d]
        rec = {"n": len(v), "z_mean": float(np.mean(v)) if v else None}
        for tname in ("human_1%", "human_0.1%", "model_1%"):
            thr = th[tname]
            if thr is None or not np.isfinite(thr) or not v:
                rec[tname] = None; continue
            kk = sum(x > thr for x in v); lo, hi = wilson(kk, len(v))
            rec[tname] = {"rate": kk / len(v), "ci": [lo, hi]}
        rec["auroc_vs_human"] = auroc(v, th["_hv"]) if (v and th["_hv"]) else None
        report["results"][f"{run}|{arm}|{d}"] = rec
        f = lambda x: "-" if x is None else f"{100 * x['rate']:5.1f}"
        print(f"{run:28s} {arm:16s} {d:18s} n={len(v):4d} z={rec['z_mean'] if rec['z_mean'] is not None else float('nan'):6.2f} "
              f"TPR@1%={f(rec['human_1%'])} @0.1%={f(rec['human_0.1%'])} model1%={f(rec['model_1%'])} AUROC={rec['auroc_vs_human'] or float('nan'):.3f}", flush=True)
    for d in report["thresholds"]:
        report["thresholds"][d].pop("_hv")
    if a.align:                                                         # generator's sentence trace vs the detector's split
        from wmworld.stages import collect_run_texts
        for spec in a.align:
            run, _, arm = spec.partition(":")
            n = same_n = same_p = 0
            for g in collect_run_texts(Path(run))[0]:
                if g["arm"] != arm or g.get("status", "ok") != "ok" or not g.get("text"):
                    continue
                r = recs.get(f"{g['text_hash']}|{g['prompt_id']}")
                tr = (g.get("cost") or {}).get("sentence_trace")
                if r is None or tr is None:
                    continue
                n += 1
                if r["n_sent"] == len(tr):
                    same_n += 1
                    same_p += all(np.allclose(r["proxy"][i], tr[i]["proxy"], atol=1e-4) for i in range(len(tr)))
            report[f"alignment|{spec}"] = {"texts": n, "same_sentence_count": same_n, "same_proxies": same_p}
            print(f"alignment {spec}: {n} texts, same sentence count {same_n}, same proxies at every position {same_p}")
    write_json(a.out, report)
    print(f"wrote {a.out}")


def main():
    mode, argv = sys.argv[1], sys.argv[2:]
    {"gen": run_gen, "score": run_score, "report": run_report}.get(mode, lambda _: sys.exit(f"unknown mode {mode}"))(argv)


if __name__ == "__main__":
    main()
