"""Token-level equivalence of the exported package (~/wmark-release) with the original sampler and detector."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/daniil.orel/wmworld")
sys.path.insert(0, "/home/daniil.orel/wmark-release")

from wmworld.backend import SamplingParams  # noqa: E402
from wmworld.config import load_config  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM, poe_block_ids  # noqa: E402
from wmworld.posstats import VARIANTS  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import build_backend, load_key_records  # noqa: E402

import wmark  # noqa: E402
from wmark.keys import Key  # noqa: E402

REF_CACHE = "/l/users/daniil.orel/hf_cache/hub"
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Give me five practical tips for preparing for a job interview."]
cfg = load_config("configs/pilot.yaml")
backend = build_backend(cfg, "cuda")
ref = build_reference(cfg)
tmap = ref.token_map(backend, cache_path=cfg.path_of("assets") / "token_map_Llama-3.2-1B-Instruct.npz", log=None)
recs = load_key_records(cfg, "len10")
specs = [r.to_keyspec() for r in recs]

gen2 = wmark.Generator()
ref2 = wmark.Reference(cache_dir=REF_CACHE)
tmap2 = ref2.token_map(gen2)
print("token map identical:", bool(np.array_equal(np.asarray(tmap), tmap2)))
keys2 = [Key(s.key_id, s.prefix) for s in specs]
ok = bool(np.array_equal(np.asarray(tmap), tmap2))


def original(prompt, ks, seed, max_new, beta, span=0, partners=None, d_max=0.0, warmup=0):
    ctx = backend.chat_prefix_ids(GENERATOR_SYSTEM, prompt)
    mk = lambda pfx: ref.stepper(ref.chat_prefix_ids(ks.system_template.format(prefix=pfx), ks.user_message))
    fac = [(lambda p=p: mk(p)) for p in [ks.prefix] + list(partners or [])] if partners else None
    return backend.sample_poe_cross(ctx, mk(ks.prefix), ref.stepper(ref.chat_prefix_ids(ks.system_text(True), ks.user_message)), tmap,
                                    max_new, SamplingParams(0.8, 0.95, 0), seed, 64, beta, 0.5, "entropy", 2.0, 0.2, False,
                                    poe_block_ids(backend, ks, True, False), "none", 5.0, 0.0, 4, span_tokens=span, warmup=warmup,
                                    d_max=d_max, ref_key_factories=fac)


cases = [dict(beta=4.0), dict(beta=3.0, d_max=1.0, warmup=16), dict(beta=4.0, span=32, rot=3)]
texts = []
for ci, c in enumerate(cases):
    for pi, prompt in enumerate(PROMPTS):
        ks, k2 = specs[pi], keys2[pi]
        seed = 1000 + 17 * ci + pi
        partners = [s.prefix for s in specs[3:3 + c.get("rot", 1) - 1]] if c.get("rot") else None
        a = original(prompt, ks, seed, 160, c["beta"], c.get("span", 0), partners, c.get("d_max", 0.0), c.get("warmup", 0))
        wc = wmark.WatermarkConfig(beta=c["beta"], contrast_max=c.get("d_max", 0.0), warmup_tokens=c.get("warmup", 0), span_tokens=c.get("span", 0))
        use = [k2] + (keys2[3:3 + c["rot"] - 1] if c.get("rot") else [])
        b = wmark.generate_watermarked(gen2, ref2, tmap2, prompt, use, wc, wmark.SamplingConfig(max_new_tokens=160), seed)
        same = a["token_ids"] == b["token_ids"]
        rot_same = (not c.get("rot")) or (a["span_starts"] == b["span_starts"] and a["key_order"] == b["key_order"])
        ok &= same and rot_same
        print(f"case {ci} prompt {pi}: {len(a['token_ids'])} vs {len(b['token_ids'])} tokens, identical={same}, spans identical={rot_same}")
        texts.append(b["text"])

scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
ctx0 = scorer.context_ids(specs[0], neutral=True)
ctxk = [scorer.context_ids(s, neutral=False) for s in specs]
old = ref.position_stats(ctx0, ctxk, [scorer.tokenize(t) for t in texts])
new = wmark.position_stats(ref2, keys2, texts)
for i, (o, n) in enumerate(zip(old, new)):
    dh, dd = float(np.abs(o["h0"] - n["h0"]).max()), float(np.abs(o["d"] - n["d"]).max())
    to = [VARIANTS["T_clip05_gate"](o["h0"], o["d"][k], o["mu"][k], o["var"][k]) for k in range(len(specs))]
    tn = [wmark.statistic(n["h0"], n["d"][k]) for k in range(len(keys2))]
    dt = max(abs(x - y) for x, y in zip(to, tn))
    ok &= dh < 1e-3 and dd < 1e-3 and dt < 1e-2
    print(f"text {i}: max|dh0|={dh:.2e} max|dd|={dd:.2e} max|dT|={dt:.2e}; T under own key / others: "
          f"{tn[i % 2]:.2f} / {np.mean([t for k, t in enumerate(tn) if k != i % 2]):.2f}")
clean = wmark.generate_clean(gen2, PROMPTS[0], wmark.SamplingConfig(max_new_tokens=96), 5)
print("clean sample:", clean["text"][:160].replace("\n", " "))
print("RELEASE GPU CHECK:", "PASS" if ok else "FAIL")
