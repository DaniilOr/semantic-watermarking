"""Early read of one finished arm: clean-text detection at the FROZEN thresholds of runs/detect_ss_llama3b.json (no new
null), plus cost.  CPU.  The full tail (attacks, detection with alt keys, judge) still runs afterwards."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld import semwm  # noqa: E402
from wmworld.posstats import wilson  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

run, arm = sys.argv[1], sys.argv[2]
thr = json.loads(Path("runs/detect_ss_llama3b.json").read_text())["thresholds"]
gs = [g for g in collect_run_texts(Path(run))[0] if g["arm"] == arm and g.get("status", "ok") == "ok" and g.get("text")]
emb = semwm.Embedder("/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282", device="cpu", batch_size=128)
key = semwm.Key.shipped(); pl = semwm.Hyperplanes.for_key(key, emb.dim)
S = [semwm.score_text(g["text"], key, emb, W=16, planes=pl) for g in gs]
print(f"{arm}: n={len(gs)}")
for st in ("z_chain", "t_cont", "z_segchain", "t_segcont"):
    v = [s[st] for s in S if s[st] is not None]
    k = sum(x > thr[st] for x in v); lo, hi = wilson(k, len(v))
    print(f"  {st:10s} detected {k / len(v):.3f} [{lo:.3f}, {hi:.3f}]  (threshold {thr[st]:.3f})")
U = [u for g in gs for u in g["cost"]["units"] if u.get("prev_bin") is not None]
nd = np.array([u["n_drawn"] for u in U])
print(f"  candidates/unit {nd.mean():.2f} (median {np.median(nd):.0f}, {np.mean(nd == 16):.2f} of units reach 16); gen-side in-mask {np.mean([u['in_mask'] for u in U]):.3f}; "
      f"s/text {np.mean([g['cost']['wall_seconds'] for g in gs]):.1f}; median tokens {np.median([g['visible_tokens'] for g in gs]):.0f}")
