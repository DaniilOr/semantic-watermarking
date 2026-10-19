"""The frozen cross-model null: the conservative rank threshold and per-key application."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from freeze_null import upper_tail_threshold  # noqa: E402


def test_threshold_holds_the_nominal_level():
    """#{null >= threshold} / m must not exceed alpha -- the conservative direction of the rank formula."""
    rng = np.random.default_rng(0)
    for m in (100, 1000, 5000):
        v = list(rng.normal(size=m))
        for alpha in (0.01, 0.05):
            t = upper_tail_threshold(v, alpha)
            assert float(np.mean([x >= t for x in v])) <= alpha + 1e-12


def test_threshold_is_a_value_from_the_sample():
    v = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert upper_tail_threshold(v, 0.5) in v


def test_small_sample_returns_infinity_rather_than_a_false_threshold():
    """With too few observations to resolve the tail, no finite threshold is claimed."""
    assert upper_tail_threshold([1.0, 2.0, 3.0], 0.01) == float("inf")


def test_pooling_correlated_keys_does_not_add_independent_observations():
    """m texts scored under k keys give m*k values but only m independent ones.

    The 1% tail of such a pool is driven by a handful of texts, which is why a per-run null over 60 base texts
    gives an unstable threshold: this reproduces that instability directly.
    """
    rng = np.random.default_rng(1)
    spread = []
    for _ in range(40):
        per_text = rng.normal(size=60)                      # 60 independent texts
        pooled = [float(t + 0.05 * rng.normal()) for t in per_text for _ in range(12)]   # x12 near-identical keys
        spread.append(upper_tail_threshold(pooled, 0.01))
    assert np.std(spread) > 0.2                             # thresholds swing widely across replications
