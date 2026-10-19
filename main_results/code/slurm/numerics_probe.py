import os, sys, time, json, itertools
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/daniil.orel/wmworld")
import torch
combo = sys.argv[1]  # e.g. bf16_fla
dtype, kern = combo.split("_")
from wmworld.runtime import QwenRuntime, RuntimeConfig
from wmworld.backend import SamplingParams
from wmworld.scoring import KeySpec, LikelihoodScorer
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rt = QwenRuntime(RuntimeConfig(revision="2fc06364715b967f1860aea9cf38778875588b17", dtype=("bfloat16" if dtype=="bf16" else "float32"), kernels=kern))
sc = LikelihoodScorer(rt)
key = KeySpec(key_id="k", prefix="You eat chicken every day.", family="f")
base = ("The market opened early, and the vendors arranged their produce in neat rows before the first customers arrived. "
        "A light rain had fallen overnight, leaving the cobblestones dark and slick, and the smell of bread drifted from the bakery on the corner. ")
texts = ["Short line.", base[:80], base, base*2, base*4, base*8]
toks = [len(sc.tokenize(t)) for t in texts]
# single vs batched
single = [sc.score(t, key) for t in texts]
batched = sc.score_texts(texts, key)
d = [abs(a.score-b.score) for a,b in zip(single,batched)]
# repeatability same shape
batched2 = sc.score_texts(texts, key)
rep = max(abs(a.score-b.score) for a,b in zip(batched,batched2))
# per-token diff for the shortest text: single vs in-batch
ctx = sc.context_ids(key, False)
pt_single = rt.per_token_logprobs(ctx, sc.tokenize(texts[0]))
# batched per-token: emulate by scoring in batch with the longest
out = {"combo": combo, "tokens": toks, "abs_diff_F_single_vs_batched": d, "repeat_same_shape": rep, "ell_single": [s.ell_key for s in single], "ell_batched": [b.ell_key for b in batched]}
# throughput
torch.cuda.synchronize(); t0=time.time()
for _ in range(5): sc.score_texts([base*3]*8, key)
torch.cuda.synchronize(); out["scoring_8x300tok_x5_seconds"] = time.time()-t0
gctx = rt.chat_prefix_ids("You are a helpful assistant.", "Write a story about a lighthouse keeper.")
torch.cuda.synchronize(); t0=time.time()
for i in range(5): rt.sample_units(gctx, 8, 16, SamplingParams(0.8,0.95), seed=i, eos_allowed_from=16)
torch.cuda.synchronize(); out["sample_8x16_x5_seconds"] = time.time()-t0
# long-prefix sampling cost (prefill 500 tokens + 16 steps)
long_ctx = gctx + sc.tokenize(base*4)
torch.cuda.synchronize(); t0=time.time()
for i in range(3): rt.sample_units(long_ctx, 8, 16, SamplingParams(0.8,0.95), seed=i, eos_allowed_from=16)
torch.cuda.synchronize(); out["sample_8x16_prefix500_x3_seconds"] = time.time()-t0
out["peak_mem_GB"] = torch.cuda.max_memory_allocated()/1e9
out["impl"] = rt.deltanet_impl
print("RESULT " + json.dumps(out))
