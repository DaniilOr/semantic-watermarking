"""Detector variants over per-position statistics (model-free)."""

import numpy as np
import pytest

from wmworld.backend import FakeRefLM
from wmworld.posstats import VARIANTS, auroc, gate_weights, matched_sum, upper_tail_threshold, weighted_t, wilson


def test_unit_weights_reproduce_the_plain_detector():
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 1.0, size=200)
    plain = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
    assert weighted_t(d) == pytest.approx(plain)
    assert weighted_t(d, np.ones_like(d)) == pytest.approx(plain, rel=1e-9)


def test_gate_is_the_generators_ramp():
    w = gate_weights(np.array([0.0, 0.2, 0.6, 1.0, 5.0]), 0.2, 1.0)
    assert w.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_too_few_informative_positions_abstain():
    d = np.ones(100); w = np.zeros(100); w[:3] = 1.0
    assert weighted_t(d + np.arange(100) * 1e-3, w) is None
    assert matched_sum(d, np.zeros(100), np.ones(100), w) is None


def test_fake_reference_position_stats_are_consistent():
    """d must equal the realized token's keyed-minus-neutral log-prob; mu/var must be the null moments of that contrast."""
    ref = FakeRefLM()
    ctx0, ctxk = ref.chat_prefix_ids("neutral", "u"), [ref.chat_prefix_ids("key one", "u"), ref.chat_prefix_ids("key two", "u")]
    t = ref.encode_text("the quick brown fox jumps over the lazy dog")
    ps = ref.position_stats(ctx0, ctxk, [t])[0]
    assert ps["d"].shape == (2, len(t)) and ps["h0"].shape == (len(t),)
    for k, ck in enumerate(ctxk):
        want = np.array(ref.logprob_tokens(ck, [t])[0]) - np.array(ref.logprob_tokens(ctx0, [t])[0])
        assert ps["d"][k] == pytest.approx(want, abs=1e-5)
    assert (ps["mu"] <= 1e-6).all()            # E_p0[log pk - log p0] = -KL(p0 || pk) <= 0
    assert (ps["var"] >= 0).all() and (ps["h0"] > 0).all()


def test_weighting_helps_when_the_signal_sits_in_gated_positions():
    """A push confined to high-entropy positions is diluted by the plain mean and recovered by the gate weights."""
    rng = np.random.default_rng(3)
    wins = 0
    for _ in range(200):
        n = 400
        h = np.where(rng.random(n) < 0.3, 2.0, 0.05)          # 30% undecided positions
        d = rng.normal(0.0, 1.0, size=n) + 0.35 * (h > 1.0)  # signal only where the generator was free
        mu, var = np.zeros(n), np.ones(n)
        null = rng.normal(0.0, 1.0, size=n)
        gain_plain = VARIANTS["T"](h, d, mu, var) - VARIANTS["T"](h, null, mu, var)
        gain_gate = VARIANTS["T_gate"](h, d, mu, var) - VARIANTS["T_gate"](h, null, mu, var)
        wins += gain_gate > gain_plain
    assert wins > 140


def test_threshold_auroc_wilson():
    rng = np.random.default_rng(1)
    v = list(rng.normal(size=2000))
    t = upper_tail_threshold(v, 0.01)
    assert np.mean([x >= t for x in v]) <= 0.01
    assert upper_tail_threshold([1.0, 2.0], 0.01) == float("inf")
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0 and auroc([1, 1], [1, 1]) == 0.5
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_offline_pipeline_end_to_end(tmp_path):
    """Dump -> load -> evaluate, model-free: a text tilted toward a key is detected under that key only."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from detector_variants import evaluate
    from wmworld.posstats import Dump, write_dump

    ref = FakeRefLM(alpha=0.6)
    ctx0 = ref.chat_prefix_ids("neutral", "u")
    keys = ["k1", "k2", "k3"]
    ctxk = [ref.chat_prefix_ids(f"secret {k}", "u") for k in keys]
    rng = np.random.default_rng(0)

    def sample(n, key=None, beta=1.5):                     # clean: the neutral world; marked: the same world TILTED
        s0, prev, out = ref._shift(ctx0), (ctx0[-1] if ctx0 else 0), []
        sk = ref._shift(ctxk[key]) if key is not None else None
        for _ in range(n):
            l0 = ref._logprobs(prev, s0)
            w = l0 if sk is None else l0 + beta * np.clip(ref._logprobs(prev, sk) - l0, -0.5, 0.5)   # the method's push
            p = np.exp(w - w.max()); tok = int(rng.choice(ref.VOCAB, p=p / p.sum()))
            out.append(tok); prev = tok
        return out

    def dump(path, rows):                                 # rows: (arm, assigned key, tokens)
        gens = [{"text_hash": f"h{i}", "arm": a, "assigned_key_id": k, "id": f"g{i}", "visible_tokens": len(t)}
                for i, (a, k, t) in enumerate(rows)]
        write_dump(path, keys, keys, gens, ref.position_stats(ctx0, ctxk, [t for _, _, t in rows]))
        return Dump(path)

    null = dump(tmp_path / "null.npz", [("base", "", sample(120)) for _ in range(300)])
    run = dump(tmp_path / "run.npz", [("base", "", sample(120)) for _ in range(40)] +
               [("wm", keys[i % 3], sample(120, key=i % 3)) for i in range(60)])
    for variant in ("T", "matched_gate"):
        res = evaluate(run, null, keys, variant)
        assert res["wm"]["tpr"] > 0.8 and res["wm"]["auroc"] > 0.95      # the right key lights up
        # the fake's neutral world carries its own random shift, so every key shares a "not neutral" component and
        # wrong keys rise too (the confound decoy-standardisation removes); the RIGHT key must still stand clear
        assert res["wm"]["z_mean"] > res["wm"]["wrong_z_mean"] + 1.0
        assert res["base"]["fpr"] < 0.05                                  # and clean text stays clean
    assert run.stat("T", 0, "k1", 0, 50) is not None and run.stat("T", 0, "nope") is None


def test_analysis_script_runs_end_to_end(tmp_path, monkeypatch):
    """The CLI path (tables, prefixes, JSON report) on synthetic dumps -- so a typo cannot cost a queued GPU job."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import detector_variants as dv
    from wmworld.posstats import write_dump

    ref = FakeRefLM(alpha=0.6)
    ctx0 = ref.chat_prefix_ids("neutral", "u")
    keys = ["k1", "k2"]
    ctxk = [ref.chat_prefix_ids(f"secret {k}", "u") for k in keys]
    rng = np.random.default_rng(5)

    def sample(n, key=None):
        s0, prev, out = ref._shift(ctx0), ctx0[-1], []
        sk = ref._shift(ctxk[key]) if key is not None else None
        for _ in range(n):
            l0 = ref._logprobs(prev, s0)
            w = l0 if sk is None else l0 + 1.5 * np.clip(ref._logprobs(prev, sk) - l0, -0.5, 0.5)
            p = np.exp(w - w.max()); tok = int(rng.choice(ref.VOCAB, p=p / p.sum())); out.append(tok); prev = tok
        return out

    def dump(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        gens = [{"text_hash": f"h{i}", "arm": a, "assigned_key_id": k, "id": f"g{i}", "visible_tokens": len(t)}
                for i, (a, k, t) in enumerate(rows)]
        write_dump(path, keys, keys, gens, ref.position_stats(ctx0, ctxk, [t for _, _, t in rows]))

    dump(tmp_path / "null.npz", [("base", "", sample(96)) for _ in range(150)])
    dump(tmp_path / "runs" / "toy" / "eval" / "posstats_randomwords.npz",
         [("base", "", sample(96)) for _ in range(20)] + [("wm", keys[i % 2], sample(96, i % 2)) for i in range(30)])
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["detector_variants.py", "--null", str(tmp_path / "null.npz"), "--run",
                                      str(tmp_path / "runs" / "toy"), "--prefixes", "48,96", "--out", str(out)])
    dv.main()
    rep = json.loads(out.read_text())
    r = rep["runs"]["toy"]
    assert set(r["variants"]) == set(dv.VARIANTS) and "48" in r["by_prefix"]
    assert r["variants"]["T"]["wm"]["tpr"] > 0.5 and len(r["variants"]["T"]["wm"]["pos_z"]) == 30
    assert sum(r["variants"]["T"]["_null"]["hist"]) == r["variants"]["T"]["_null"]["n"]
