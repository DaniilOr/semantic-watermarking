"""The windowed sampler-side view and the windowed detector must agree position by position (real reference model)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.config import load_config  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import load_key_records  # noqa: E402

W, S = 64, 32
cfg = load_config("configs/pilot.yaml")
ref = build_reference(cfg)
scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
ks = load_key_records(cfg, "len10")[0].to_keyspec()
ctxk, ctx0 = scorer.context_ids(ks, neutral=False), scorer.context_ids(ks, neutral=True)
g = next(json.loads(l) for f in sorted(Path("runs/keylen10").glob("generations*.jsonl")) for l in open(f) if json.loads(l)["arm"] == "base" and json.loads(l)["visible_tokens"] > 300)
toks = scorer.tokenize(g["text"])
det = ref.position_stats_windowed(ctx0, [ctxk], [toks], W, S)[0]
full = ref.position_stats(ctx0, [ctxk], [toks])[0]
sk, s0 = ref.stepper(ctxk, window=W, stride=S), ref.stepper(ctx0, window=W, stride=S)
worst, n_checked = 0.0, 0
for r in [0, 3, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 130, 191, 192, 250, len(toks) - 1]:
    prefix = ref.decode(toks[:r])
    if ref.encode_text(prefix) != toks[:r]:
        continue
    lk, l0 = sk.logprobs_after(prefix), s0.logprobs_after(prefix)
    d_step = float(lk[toks[r]] - l0[toks[r]]); h_step = float(-(l0.exp() * l0).sum())
    err = max(abs(d_step - float(det["d"][0][r])), abs(h_step - float(det["h0"][r])))
    worst = max(worst, err); n_checked += 1
    print(f"pos {r:4d}: d sampler-side {d_step:+.4f} detector-side {det['d'][0][r]:+.4f} | whole-text d {full['d'][0][r]:+.4f} | err {err:.1e}")
early = float(np.abs(det["d"][0][:W] - full["d"][0][:W]).max())
print(f"positions checked: {n_checked}; worst sampler-vs-detector error {worst:.2e}; first {W} positions equal the whole-text view: max diff {early:.1e}")
ok = n_checked >= 10 and worst < 2e-3 and early < 2e-3
print("WINDOW SELFCHECK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 3)
