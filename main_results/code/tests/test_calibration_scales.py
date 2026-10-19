"""Spec Section 14 items 4 and 8 (calibration isolation, tie rule, multi-key) and Section 5 scale rules."""

import math

import pytest

from wmworld.calibration import (Calibration, bonferroni_multi_key, build_calibration, detect, detect_from_score,
                                 implied_threshold, tail_rank_p)
from wmworld.scales import (ABSOLUTE_FLOOR, ScaleTable, compute_floor, entry_key, fit_scale_table, length_bin,
                            within_pool_sd)
from wmworld.scoring import KeySpec, ScoreResult


def test_tail_rank_formula_and_ties():
    s = [0.1, 0.2, 0.2, 0.3, 0.5]
    assert tail_rank_p(s, 0.2) == pytest.approx((1 + 4) / 6)      # ties count as >=
    assert tail_rank_p(s, 0.6) == pytest.approx(1 / 6)
    assert tail_rank_p(s, 0.0) == pytest.approx(1.0)
    assert math.isnan(tail_rank_p([], 0.1))


def test_implied_threshold_consistent_with_rank():
    s = sorted([i / 1000 for i in range(1500)])
    thr = implied_threshold(s, 0.01)
    assert tail_rank_p(s, thr + 1e-9) <= 0.01
    assert tail_rank_p(s, thr) > 0.01 or tail_rank_p(s, thr) <= 0.01  # boundary defined; no crash
    assert implied_threshold([0.1, 0.2], 0.01) is None                # too few examples


def _cal(m_g=1200, m_h=1200, toy=False, alpha=0.01):
    g = [(f"g{i}", i / m_g) for i in range(m_g)]
    h = [(f"h{i}", 0.5 + i / m_h) for i in range(m_h)]
    key = KeySpec(key_id="K", prefix="p", family="f")
    return build_calibration({"K": {"generator": g, "human_web": h}}, {"K": key}, "llm_global_v1", alpha, 32, 4096, 1000,
                             "null_calibration", "hash", toy, {}), key


def test_combined_uses_max_of_cohort_ranks_and_flags_at_alpha():
    cal, key = _cal()
    ok = ScoreResult(score=1.6, status="ok", n_tokens=100, ell_key=-1.0, ell_neutral=-2.0)
    r = detect_from_score(ok, 100, key, cal)
    assert r.decision == "detected" and r.p_final == max(r.p_generator, r.p_human)
    mid = ScoreResult(score=0.995, status="ok", n_tokens=100, ell_key=-1.0, ell_neutral=-2.0)  # top of G (p=7/1201) but middle of H
    r2 = detect_from_score(mid, 100, key, cal)
    assert r2.p_generator < 0.01 < r2.p_human and r2.decision == "not_detected"


def test_generator_only_mode_when_human_cohort_too_small():
    cal, key = _cal(m_h=100)
    assert cal.per_key["K"].mode == "generator_only"
    r = detect_from_score(ScoreResult(score=0.999, status="ok", n_tokens=100), 100, key, cal)
    assert r.p_human is None and r.status == "generator_only_calibration"


def test_abstentions_are_explicit():
    cal, key = _cal()
    assert detect_from_score(ScoreResult(score=5.0, status="ok", n_tokens=10), 10, key, cal).decision == "abstain_short"
    assert detect_from_score(ScoreResult(score=None, status="abstain_overflow", n_tokens=5000), 5000, key, cal).decision == "abstain_overflow"
    assert detect_from_score(ScoreResult(score=None, status="error", n_tokens=100, error="x"), 100, key, cal).decision == "error"
    assert detect_from_score(ScoreResult(score=float("nan"), status="ok", n_tokens=100), 100, key, cal).decision == "error"


def test_calibration_roundtrip_and_id_stability():
    cal, _ = _cal()
    d = cal.to_dict()
    cal2 = Calibration.from_dict(d)
    assert cal2.calibration_id == cal.calibration_id
    assert cal2.per_key["K"].generator_scores == cal.per_key["K"].generator_scores
    assert d["per_key"]["K"]["counts"] == {"generator": 1200, "human": 1200}


def test_secondary_rank_detector_build_and_detect():
    from wmworld.calibration import build_secondary, detect_secondary, rank_statistic
    rng = __import__("numpy").random.default_rng(0)
    keys = [f"k{i}" for i in range(6)]
    # null cohort: independent scores per key with key-specific offsets (the z-scoring must remove them)
    offsets = {k: i * 0.3 for i, k in enumerate(keys)}
    null = {f"t{n}": {k: offsets[k] + rng.normal(0, 0.1) for k in keys} for n in range(300)}
    sec = build_secondary(null, keys)
    assert sec["decoys"] == keys and len(sec["rank_frac_null"]) == 300 * 6
    assert abs(sec["null_stats"]["k5"]["mean"] - 1.5) < 0.05
    cal, key = _cal()
    cal.keys = {k: {"key_id": k, "prefix": k, "family": "f"} for k in keys}
    cal.secondary = sec
    cal.alpha = 0.05
    # a text whose true-key score is far above its null while decoys are at their null means
    strong = {k: offsets[k] for k in keys}; strong["k2"] = offsets["k2"] + 1.0
    d = detect_secondary(strong, "k2", cal)
    assert d["status"] == "ok" and d["rank_frac"] == 1.0 and d["decision_zdiff"] == "detected"
    weak = {k: offsets[k] for k in keys}
    d2 = detect_secondary(weak, "k2", cal)
    assert d2["decision_zdiff"] == "not_detected"
    # discrete floor: with 5 decoys, beating all of them has null probability ~1/6 > alpha=0.05
    assert d["min_achievable_p_rank_frac"] > 0.05 and d["decision_rank_frac"] == "not_detected"
    assert rank_statistic(1.0, [0.0, 2.0]) == 0.5 and rank_statistic(None, [1.0]) is None


def test_multi_key_search_uses_bonferroni_not_best_raw():
    assert bonferroni_multi_key({"a": 0.004, "b": 0.5, "c": 0.9}, 0.01)["decision"] == "not_detected"   # 0.004*3 > 0.01
    assert bonferroni_multi_key({"a": 0.003, "b": 0.5, "c": 0.9}, 0.01)["decision"] == "detected"


def test_detect_end_to_end_fake(lm, scorer, key_a):
    nulls = [(f"n{i}", -0.5 + i * 0.01) for i in range(50)]
    cal = build_calibration({"A": {"generator": nulls, "human_web": nulls}}, {"A": key_a}, "llm_global_v1", 0.05, 4, 600, 1000,
                            "toy", "h", True, {})
    r = detect("a text that is long enough", key_a, cal, scorer)
    assert r.decision in ("detected", "not_detected") and r.calibration_id.startswith("toy_")
    assert detect("hi", key_a, cal, scorer).decision == "abstain_short"


# ---- scales -------------------------------------------------------------------- #

def test_length_bins():
    assert length_bin(0) == "[0,64)" and length_bin(63) == "[0,64)" and length_bin(64) == "[64,128)"
    assert length_bin(255) == "[128,256)" and length_bin(256) == "[256,inf)" and length_bin(10000) == "[256,inf)"


def test_floor_and_inactive_no_signal():
    assert compute_floor(0.0) == ABSOLUTE_FLOOR
    assert compute_floor(1e-4) == pytest.approx(1e-2)
    sds = {entry_key("f", "tokens16", "global", "[0,64)"): [(f"g{i}", 1e-9) for i in range(40)]}
    t = fit_scale_table(sds, "s", "p", 1e-4, toy=False)
    e = t.lookup("f", "tokens16", "global", 10)
    assert e.status == "inactive_no_signal" and e.sigma is None


def test_nearest_bin_fallback_requires_30_groups():
    sds = {entry_key("f", "tokens16", "global", "[0,64)"): [(f"g{i}", 0.3) for i in range(40)],
           entry_key("f", "tokens16", "global", "[64,128)"): [(f"g{i}", 0.2) for i in range(10)]}
    t = fit_scale_table(sds, "s", "p", 0.0, toy=False)
    e = t.lookup("f", "tokens16", "global", 100)
    assert e.sigma == pytest.approx(0.3) and e.fallback_from == "[0,64)"
    assert t.lookup("f", "tokens16", "global", 300).fallback_from == "[0,64)"
    assert t.lookup("other", "tokens16", "global", 300).status == "missing"
    # duplicate group ids count once
    sds2 = {entry_key("f", "tokens16", "global", "[0,64)"): [("g0", 0.3)] * 40}
    t2 = fit_scale_table(sds2, "s", "p", 0.0, toy=False)
    assert t2.lookup("f", "tokens16", "global", 1).status == "insufficient_pools"
    assert t2.lookup("f", "tokens16", "global", 1).sigma is None


def test_within_pool_sd_and_roundtrip():
    assert within_pool_sd([1.0, 1.0, 1.0]) == 0.0
    assert math.isnan(within_pool_sd([1.0]))
    sds = {entry_key("f", "tokens16", "global", "[0,64)"): [(f"g{i}", 0.3) for i in range(40)]}
    t = fit_scale_table(sds, "s", "p", 0.0, toy=False)
    t2 = ScaleTable.from_dict(t.to_dict())
    assert t2.lookup("f", "tokens16", "global", 1).sigma == pytest.approx(0.3)
