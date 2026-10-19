import sys; sys.path.insert(0, "/home/daniil.orel/wmworld")
from wmworld.backend import FakeLM, SamplingParams
lm = FakeLM(eos_bias=9.0, max_input_tokens=600)
ctx = lm.chat_prefix_ids("You are a helpful assistant.", "p")
k = lm.chat_prefix_ids("You are a helpful assistant.\nBackground: You eat chicken every day.", "Write a coherent response.")
n = lm.chat_prefix_ids("You are a helpful assistant.\nBackground: No additional background is specified.", "Write a coherent response.")
for steer_eos in (True, False):
    out = lm.sample_poe(ctx, k, n, 40, SamplingParams(1.0, 1.0, 0), 1, 4, 5.0, 5.0, "none", 2.0, 0.5, 0.1, steer_eos)
    print("steer_eos", steer_eos, "ended", out["ended"], "len", len(out["token_ids"]), "contrast tail", [round(x,3) for x in out["contrast"][-3:]])
u = lm.sample_units(ctx, 1, 40, SamplingParams(1.0, 1.0, 0), 1, 4)[0]
print("sample_units ended", u.ended, "len", len(u.token_ids))
import numpy as np
z = lm._logits(ctx[-1], lm._shift(ctx)); print("EOS logit", z[lm.EOS], "max letter logit", z[:256].max())
