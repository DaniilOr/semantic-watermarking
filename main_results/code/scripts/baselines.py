"""Reference watermarks (KGW green/red list and SemStamp) on OUR generator, prompts and settings.

The point is a like-for-like comparison: the same Qwen3.5-0.8B checkpoint, the same held-out prompts, the same
sampling parameters and token budget, the same judge and the same attacks as our own arms.  The watermark logic
itself comes from each paper's own repository:

  kgw       third_party/lm-watermarking (Kirchenbauer et al.): WatermarkLogitsProcessor drives HF generate.
  semstamp  third_party/SemStamp (Hou et al.): their SBERTLSHModel, get_mask_from_seed and reject_close_generation,
            with their fine-tuned embedder AbeHou/SemStamp-c4-sbert.  Their own sampling loop raises
            NotImplementedError for any model that is not OPT, so the sentence loop here is ours; the watermark
            (LSH partition, valid mask seeded by the previous sentence, rejection with a margin) is theirs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM  # noqa: E402
from wmworld.records import JsonlWriter, GenerationRecord, content_hash, output_id  # noqa: E402
from wmworld.stages import build_backend  # noqa: E402
from wmworld.textnorm import normalize_text, text_hash  # noqa: E402

KGW = dict(gamma=0.25, delta=2.0, seeding_scheme="simple_1")   # the original paper's scheme; "selfhash" is the repo default (slower, more robust)
SEMSTAMP = dict(lsh_dim=3, lmbd=0.25, margin=0.02, max_trials=100, sent_max_tokens=64)
EMBEDDER = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"


def held_out_prompts(cfg, run="runs/conf_cross"):
    """Exactly the prompts of the held-out run, so every arm answers the same questions."""
    from wmworld.data.prepare import load_prepared
    want = {}
    for f in Path(run).glob("generations*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            if g["arm"] == "base":
                want[g["prompt_id"]] = g
    rows = [p for p in load_prepared(cfg.path_of("data_prepared"), partition="positive_test") if p.prompt_id in want]
    return sorted(rows, key=lambda p: p.prompt_id), want


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@torch.no_grad()
def gen_kgw(backend, ctx_ids, sampling, seed, min_new, max_new, scheme=None):
    from transformers import LogitsProcessorList
    sys.path.insert(0, str(Path("third_party/lm-watermarking").resolve()))
    from extended_watermark_processor import WatermarkLogitsProcessor
    params = dict(KGW, seeding_scheme=scheme or KGW["seeding_scheme"])
    wp = WatermarkLogitsProcessor(vocab=list(backend.tokenizer.get_vocab().values()), **params)
    g = torch.Generator(device=backend.device); g.manual_seed(int(seed) % (2 ** 63 - 1))
    out = backend.model.generate(
        torch.tensor([ctx_ids], device=backend.device), logits_processor=LogitsProcessorList([wp]),
        do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p,
        min_new_tokens=min_new, max_new_tokens=max_new, eos_token_id=backend.eos_ids,
        pad_token_id=backend.pad_id)
    ids = [int(t) for t in out[0][len(ctx_ids):] if int(t) not in backend.eos_ids]
    return ids, {"n_tokens": len(ids)}


@torch.no_grad()
def gen_semstamp(backend, lsh, ctx_ids, prompt, sampling, seed, min_new, max_new):
    """One sentence at a time; keep it only if its LSH bin is in the mask seeded by the previous sentence."""
    from sampling_lsh_utils import get_mask_from_seed, reject_close_generation
    cfg = SEMSTAMP
    lsh_seed = lsh.get_hash([prompt])[0]
    mask = get_mask_from_seed(cfg["lsh_dim"], cfg["lmbd"], lsh_seed)
    ids: list[int] = []
    trials = maxed = accepted = 0
    g = torch.Generator(device=backend.device); g.manual_seed(int(seed) % (2 ** 63 - 1))
    while len(ids) < max_new:
        cur = 0
        while True:
            out = backend.model.generate(
                torch.tensor([ctx_ids + ids], device=backend.device), do_sample=True,
                temperature=sampling.temperature, top_p=sampling.top_p, min_new_tokens=4,
                max_new_tokens=min(cfg["sent_max_tokens"], max_new - len(ids)),
                eos_token_id=backend.eos_ids, pad_token_id=backend.pad_id)
            new = [int(t) for t in out[0][len(ctx_ids) + len(ids):] if int(t) not in backend.eos_ids]
            if not new:
                return ids, {"n_tokens": len(ids), "trials": trials, "accepted": accepted, "maxed_out": maxed}
            piece = backend.decode(new)
            s = sentences(piece)
            if len(s) > 1:                                  # keep only the first complete sentence
                piece = s[0]
                new = backend.encode_text(piece + " ") or new   # keep the separator: "arts.After" otherwise
            trials += 1; cur += 1
            ok, _ = reject_close_generation(lsh, [piece], margin=cfg["margin"], cutoff=None)
            if len(ok) == 0 and cur < cfg["max_trials"]:
                continue
            cand = lsh.get_hash([piece])[0]
            if cand in mask or cur >= cfg["max_trials"]:
                maxed += int(cur >= cfg["max_trials"] and cand not in mask)
                ids = ids + new
                accepted += 1
                lsh_seed = cand
                mask = get_mask_from_seed(cfg["lsh_dim"], cfg["lmbd"], lsh_seed)
                break
        if len(ids) >= max_new or (len(ids) >= min_new and trials > cfg["max_trials"] * 40):
            break
    return ids, {"n_tokens": len(ids), "trials": trials, "accepted": accepted, "maxed_out": maxed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["kgw", "semstamp"], required=True)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeding-scheme", default="", help="KGW only: simple_1 (paper) or selfhash (repo default)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg.validate(require_revision=True)
    backend = build_backend(cfg, "cuda")
    gcfg = cfg.generation_config(None, {})
    sampling = gcfg.sampling()
    rows, clean = held_out_prompts(cfg)
    if a.limit:
        rows = rows[: a.limit]
    mine = [p for i, p in enumerate(rows) if i % a.num_shards == a.shard]
    run_dir = Path("runs") / (a.run_name or f"baseline_{a.method}")
    run_dir.mkdir(parents=True, exist_ok=True)
    params = (dict(KGW, seeding_scheme=a.seeding_scheme or KGW["seeding_scheme"]) if a.method == "kgw" else SEMSTAMP)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"method": a.method, "params": params, "source_run": "runs/conf_cross", "partition": "positive_test",
         "generator": cfg["models"]["shared"]["id"], "key_set": "confirmation",
         "max_new_tokens": gcfg.max_new_tokens, "min_new_tokens": gcfg.min_new_tokens,
         "temperature": sampling.temperature, "top_p": sampling.top_p}, indent=1))
    lsh = None
    if a.method == "semstamp":
        sys.path.insert(0, str(Path("third_party/SemStamp").resolve()))
        from sbert_lsh_model import SBERTLSHModel
        lsh = SBERTLSHModel(device="cuda", batch_size=8, lsh_dim=SEMSTAMP["lsh_dim"], sbert_type="base", lsh_model_path=EMBEDDER)
    ch = content_hash({"method": a.method, "params": params}, length=12)
    w = JsonlWriter(run_dir / f"generations.shard{a.shard}.jsonl")
    t0 = time.time()
    for n, p in enumerate(mine, 1):
        oid = output_id(a.method, a.method, p.prompt_id, None, 0, ch)
        if oid in w:
            continue
        ctx = backend.chat_prefix_ids(GENERATOR_SYSTEM, p.prompt)
        seed = int(content_hash({"s": p.prompt_id, "m": a.method}, length=12), 16)
        try:
            if a.method == "kgw":
                ids, stats = gen_kgw(backend, ctx, sampling, seed, gcfg.min_new_tokens, gcfg.max_new_tokens, a.seeding_scheme or None)
            else:
                ids, stats = gen_semstamp(backend, lsh, ctx, p.prompt, sampling, seed, gcfg.min_new_tokens, gcfg.max_new_tokens)
            text, status, err = normalize_text(backend.decode(ids)), "ok", None
        except Exception as e:                                # keep the failure, never silently drop the prompt
            ids, stats, text, status, err = [], {}, "", "failed", repr(e)
        w.write(GenerationRecord(
            id=oid, stage="baseline", arm=a.method, group_id=p.group_id, prompt_id=p.prompt_id, domain=p.domain,
            key_id=None, seed=0, text=text, text_hash=text_hash(text), accepted_tokens=len(ids), visible_tokens=len(ids),
            stop_reason="eos" if len(ids) < gcfg.max_new_tokens else "max_new_tokens", ends_mid_sentence=False,
            config_hash=ch, status=status, error=err, cost=stats,
            assigned_key_id=clean[p.prompt_id].get("assigned_key_id"), arm_config=params))
        if n % 5 == 0:
            print(f"[{a.method} shard {a.shard}] {n}/{len(mine)} ({time.time() - t0:.0f}s) last={stats}", flush=True)
    w.close()
    print(f"[{a.method} shard {a.shard}] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
