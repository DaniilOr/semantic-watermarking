import argparse
import time
from pathlib import Path

import numpy as np

import _path
from wmark import Generator, Reference, SamplingConfig, WatermarkConfig, generate_clean, generate_watermarked
from wmark.io import append_jsonl, load_prompts, read_jsonl, short_hash
from wmark.keys import assign_keys, load_keys


def token_map(reference, generator, path):
    path = Path(path)
    if path.exists():
        z = np.load(path)
        if int(z["ref_vocab"]) == reference.vocab_size and int(z["gen_vocab"]) == len(generator.tokenizer):
            return z["map"]
    m = reference.token_map(generator)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, map=m, ref_vocab=reference.vocab_size, gen_vocab=len(generator.tokenizer))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--entropy-min", type=float, default=0.2)
    ap.add_argument("--entropy-ref", type=float, default=2.0)
    ap.add_argument("--repeat-n", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--contrast-max", type=float, default=0.0)
    ap.add_argument("--span-tokens", type=int, default=0)
    ap.add_argument("--rotation", type=int, default=1)
    ap.add_argument("--rotation-keys", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-seed", type=int, default=20260911)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-new-tokens", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--token-map", default="cache/token_map.npz")
    ap.add_argument("--gen-cache-dir", default=None)
    ap.add_argument("--ref-cache-dir", default=None)
    a = ap.parse_args()
    if a.beta > 0 and not a.keys:
        ap.error("--keys is required when --beta > 0")
    if a.rotation > 1 and not (a.rotation_keys and a.span_tokens > 0):
        ap.error("--rotation needs --rotation-keys and --span-tokens")

    prompts = load_prompts(a.prompts)
    pids = sorted(prompts, key=lambda p: short_hash("order", p))
    if a.limit:
        pids = pids[:a.limit]
    sampling = SamplingConfig(a.temperature, a.top_p, a.top_k, a.min_new_tokens, a.max_new_tokens)
    cfg = WatermarkConfig(beta=a.beta, clip=a.clip, entropy_min=a.entropy_min, entropy_ref=a.entropy_ref,
                          block_repeat_n=a.repeat_n, warmup_tokens=a.warmup, contrast_max=a.contrast_max,
                          span_tokens=a.span_tokens)
    marked = a.beta > 0
    generator = Generator(cache_dir=a.gen_cache_dir)
    reference = tmap = assignment = pool = None
    if marked:
        keys = load_keys(a.keys)
        assignment = assign_keys(pids, keys, a.key_seed)
        pool = load_keys(a.rotation_keys) if a.rotation > 1 else []
        reference = Reference(cache_dir=a.ref_cache_dir)
        tmap = token_map(reference, generator, a.token_map)

    suffix = f".shard{a.shard}" if a.num_shards > 1 else ""
    out_path = Path(a.run) / f"generations_{a.arm}{suffix}.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    mine = [p for i, p in enumerate(pids) if i % a.num_shards == a.shard]
    t0 = time.time()
    for n, pid in enumerate(mine, 1):
        gid = short_hash(a.arm, pid, a.seed)
        if gid in done:
            continue
        row = prompts[pid]
        seed = int(short_hash("seed", a.arm, pid, a.seed, length=12), 16)
        rec = {"id": gid, "prompt_id": pid, "domain": row.get("domain"), "arm": a.arm, "seed": a.seed}
        if marked:
            key = assignment[pid]
            use = [key]
            if a.rotation > 1:
                start = int(short_hash("rotation", pid, length=8), 16)
                partners = [pool[(start + j) % len(pool)] for j in range(len(pool))]
                use += [k for k in partners if k.phrase != key.phrase][:a.rotation - 1]
            out = generate_watermarked(generator, reference, tmap, row["prompt"], use, cfg, sampling, seed)
            rec.update({"key_id": key.key_id, "key_ids": out["key_ids"], "span_starts": out["span_starts"],
                        "key_order": out["key_order"], "config": cfg.to_dict(),
                        "mean_kl": float(np.mean(out["trace"]["kl"])) if out["trace"]["kl"] else 0.0})
        else:
            out = generate_clean(generator, row["prompt"], sampling, seed)
            rec["key_id"] = None
        rec.update({"text": out["text"], "n_tokens": len(out["token_ids"]), "ended": out["ended"]})
        append_jsonl(out_path, rec)
        if n % 10 == 0:
            print(f"[{a.arm}] {n}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
