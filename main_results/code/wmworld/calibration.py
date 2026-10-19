"""Calibration artifact and known-key detection (spec Section 9).

For cohort ``a`` with scores ``S^(a)`` (m_a eligible unwatermarked texts) and a
query score ``s``:

    p_a(s) = (1 + #{i : S_i^(a) >= s}) / (m_a + 1)

``p_final = max(p_G, p_H)``; a watermark is flagged when ``p_final <= alpha``.
The artifact stores the full sorted score arrays per key and cohort so the exact
tail count is reproducible.  Thresholds are frozen from ``null_calibration``
only; ``null_test`` and positive outcomes never touch this artifact.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from .records import DetectionRecord, content_hash
from .scoring import KeySpec, LikelihoodScorer, ScoreResult
from .textnorm import TEXTNORM_VERSION, normalize_text, text_hash

CALIBRATION_VERSION = "calibration_v1"


def tail_rank_p(sorted_scores: list[float], s: float) -> float:
    """Conservative upper-tail rank with ties counted as >= (spec formula)."""
    m = len(sorted_scores)
    if m == 0:
        return float("nan")
    ge = m - bisect.bisect_left(sorted_scores, s)     # count of S_i >= s
    return (1 + ge) / (m + 1)


def implied_threshold(sorted_scores: list[float], alpha: float) -> float | None:
    """Smallest score s such that p(s) <= alpha, i.e. the (1+ge)/(m+1) <= alpha boundary."""
    m = len(sorted_scores)
    if m == 0:
        return None
    max_ge = math.floor(alpha * (m + 1)) - 1
    if max_ge < 0:
        return None
    # need ge <= max_ge  -> s > sorted[m - max_ge - 1]
    idx = m - max_ge - 1
    return float(sorted_scores[idx])


@dataclass
class KeyCalibration:
    key_id: str
    generator_scores: list[float]        # sorted ascending
    human_scores: list[float]            # sorted ascending
    generator_ids: list[str]             # text hashes (for audit), same order as scores pre-sort not required
    human_ids: list[str]
    mode: str                            # combined | generator_only


@dataclass
class Calibration:
    calibration_id: str
    scorer_id: str
    alpha: float
    min_visible_tokens: int
    max_input_tokens: int
    minimum_per_cohort: int
    partition: str
    partition_hash: str
    toy: bool
    keys: dict[str, dict]                # key_id -> KeySpec dict (exact prefixes: an experiment asset)
    per_key: dict[str, KeyCalibration]
    backend_info: dict
    version: str = CALIBRATION_VERSION
    textnorm_version: str = TEXTNORM_VERSION
    meta: dict = field(default_factory=dict)
    # optional SECONDARY rank-based detector assets: per-key null mean/sd for z-scoring, decoy list, null arrays
    secondary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calibration_id": self.calibration_id, "scorer_id": self.scorer_id, "alpha": self.alpha,
            "min_visible_tokens": self.min_visible_tokens, "max_input_tokens": self.max_input_tokens,
            "minimum_per_cohort": self.minimum_per_cohort, "partition": self.partition, "partition_hash": self.partition_hash,
            "toy": self.toy, "keys": self.keys, "backend_info": self.backend_info, "version": self.version,
            "textnorm_version": self.textnorm_version, "meta": self.meta, "secondary": self.secondary,
            "per_key": {k: {"key_id": v.key_id, "generator_scores": v.generator_scores, "human_scores": v.human_scores,
                            "generator_ids": v.generator_ids, "human_ids": v.human_ids, "mode": v.mode,
                            "counts": {"generator": len(v.generator_scores), "human": len(v.human_scores)},
                            "implied_thresholds": {"generator": implied_threshold(v.generator_scores, self.alpha),
                                                   "human": implied_threshold(v.human_scores, self.alpha)}}
                        for k, v in self.per_key.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        per_key = {k: KeyCalibration(key_id=v["key_id"], generator_scores=list(v["generator_scores"]),
                                     human_scores=list(v["human_scores"]), generator_ids=list(v.get("generator_ids", [])),
                                     human_ids=list(v.get("human_ids", [])), mode=v["mode"]) for k, v in d["per_key"].items()}
        return cls(calibration_id=d["calibration_id"], scorer_id=d["scorer_id"], alpha=float(d["alpha"]),
                   min_visible_tokens=int(d["min_visible_tokens"]), max_input_tokens=int(d["max_input_tokens"]),
                   minimum_per_cohort=int(d["minimum_per_cohort"]), partition=d["partition"], partition_hash=d["partition_hash"],
                   toy=bool(d["toy"]), keys=d["keys"], per_key=per_key, backend_info=d.get("backend_info", {}),
                   version=d.get("version", CALIBRATION_VERSION), textnorm_version=d.get("textnorm_version", TEXTNORM_VERSION),
                   meta=d.get("meta", {}), secondary=d.get("secondary", {}))

    def keyspec(self, key_id: str) -> KeySpec:
        if key_id not in self.keys:
            raise KeyError(f"key {key_id} not in calibration {self.calibration_id}")
        return KeySpec.from_dict(self.keys[key_id])


def build_calibration(scores_by_key: dict[str, dict[str, list[tuple[str, float]]]], keys: dict[str, KeySpec],
                      scorer_id: str, alpha: float, min_visible_tokens: int, max_input_tokens: int, minimum_per_cohort: int,
                      partition: str, partition_hash: str, toy: bool, backend_info: dict, meta: dict | None = None) -> Calibration:
    """``scores_by_key[key_id][cohort]`` = list of (text_hash, score) for eligible unwatermarked texts."""
    per_key: dict[str, KeyCalibration] = {}
    modes = {}
    for key_id, cohorts in scores_by_key.items():
        g = sorted(cohorts.get("generator", []), key=lambda x: x[1])
        h = sorted(cohorts.get("human_web", []), key=lambda x: x[1])
        mode = "combined" if (len(g) >= minimum_per_cohort and len(h) >= minimum_per_cohort) else "generator_only"
        if toy:
            mode = "combined" if h else "generator_only"
        modes[key_id] = mode
        per_key[key_id] = KeyCalibration(key_id=key_id, generator_scores=[float(s) for _, s in g],
                                         human_scores=[float(s) for _, s in h], generator_ids=[t for t, _ in g],
                                         human_ids=[t for t, _ in h], mode=mode)
    cal = Calibration(calibration_id="", scorer_id=scorer_id, alpha=alpha, min_visible_tokens=min_visible_tokens,
                      max_input_tokens=max_input_tokens, minimum_per_cohort=minimum_per_cohort, partition=partition,
                      partition_hash=partition_hash, toy=toy, keys={k: v.to_dict() for k, v in keys.items()},
                      per_key=per_key, backend_info=backend_info, meta=dict(meta or {}, modes=modes))
    cal.calibration_id = ("toy_" if toy else "cal_") + content_hash(cal.to_dict(), length=12)
    return cal


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #

@dataclass
class DetectionResult:
    decision: str                    # detected | not_detected | abstain_short | abstain_overflow | error
    score: float | None
    p_generator: float | None
    p_human: float | None
    p_final: float | None
    visible_tokens: int
    key_id: str
    scorer_id: str
    calibration_id: str
    status: str
    components: dict = field(default_factory=dict)

    def to_record(self, text: str) -> DetectionRecord:
        th = text_hash(text)
        return DetectionRecord(id=f"{th}.{self.key_id}.{self.calibration_id}", text_hash=th, key_id=self.key_id,
                               scorer_id=self.scorer_id, calibration_id=self.calibration_id, score=self.score,
                               visible_tokens=self.visible_tokens, p_generator=self.p_generator, p_human=self.p_human,
                               p_final=self.p_final, decision=self.decision, status=self.status, components=self.components)


def detect_from_score(res: ScoreResult, visible_tokens: int, key: KeySpec, cal: Calibration) -> DetectionResult:
    kc = cal.per_key.get(key.key_id)
    base = dict(key_id=key.key_id, scorer_id=cal.scorer_id, calibration_id=cal.calibration_id, visible_tokens=visible_tokens)
    if kc is None:
        return DetectionResult(decision="error", score=None, p_generator=None, p_human=None, p_final=None, status="no_calibration_for_key", **base)
    if visible_tokens < cal.min_visible_tokens:
        return DetectionResult(decision="abstain_short", score=res.score, p_generator=None, p_human=None, p_final=None,
                               status=f"visible_tokens<{cal.min_visible_tokens}", **base)
    if res.status == "abstain_overflow":
        return DetectionResult(decision="abstain_overflow", score=None, p_generator=None, p_human=None, p_final=None,
                               status="input_exceeds_max_tokens", **base)
    if res.status != "ok" or res.score is None or not math.isfinite(res.score):
        return DetectionResult(decision="error", score=None, p_generator=None, p_human=None, p_final=None,
                               status=res.error or res.status, **base)
    s = float(res.score)
    p_g = tail_rank_p(kc.generator_scores, s)
    p_h = tail_rank_p(kc.human_scores, s) if kc.mode == "combined" else None
    p_final = p_g if p_h is None else max(p_g, p_h)
    decision = "detected" if p_final <= cal.alpha else "not_detected"
    return DetectionResult(decision=decision, score=s, p_generator=p_g, p_human=p_h, p_final=p_final,
                           status="ok" if kc.mode == "combined" else "generator_only_calibration",
                           components={"mode": kc.mode, "m_generator": len(kc.generator_scores),
                                       "m_human": len(kc.human_scores), "alpha": cal.alpha,
                                       "contrast_sum": res.contrast_sum, "sqrt_n_score": s * math.sqrt(res.n_tokens)},
                           **base)


def detect(text: str, key: KeySpec, cal: Calibration, scorer: LikelihoodScorer) -> DetectionResult:
    """Known-key test from text + key + calibration only."""
    text = normalize_text(text)
    n_vis = len(scorer.tokenize(text))
    res = scorer.score(text, key) if n_vis >= cal.min_visible_tokens else ScoreResult(score=None, status="short", n_tokens=n_vis)
    return detect_from_score(res, n_vis, key, cal)


def detect_many(texts: list[str], key: KeySpec, cal: Calibration, scorer: LikelihoodScorer) -> list[DetectionResult]:
    texts = [normalize_text(t) for t in texts]
    n_vis = [len(scorer.tokenize(t)) for t in texts]
    idx = [i for i, n in enumerate(n_vis) if n >= cal.min_visible_tokens]
    results: list[ScoreResult | None] = [None] * len(texts)
    if idx:
        scored = scorer.score_texts([texts[i] for i in idx], key)
        for i, r in zip(idx, scored):
            results[i] = r
    out = []
    for i, t in enumerate(texts):
        r = results[i] if results[i] is not None else ScoreResult(score=None, status="short", n_tokens=n_vis[i])
        out.append(detect_from_score(r, n_vis[i], key, cal))
    return out


# --------------------------------------------------------------------------- #
# secondary rank-based detector: correct key versus decoy keys on the same text
# --------------------------------------------------------------------------- #

def rank_statistic(z_true: float, z_decoys: list[float]) -> float | None:
    """Fraction of decoy keys whose standardized score on this text is strictly below the true key's.

    Standardization uses each key's own null mean/SD (from the calibration cohort), so the
    statistic is comparable across keys; text-level nuisance shared by all keys cancels.
    """
    if not z_decoys or z_true is None or not math.isfinite(z_true):
        return None
    z = [d for d in z_decoys if d is not None and math.isfinite(d)]
    if not z:
        return None
    return sum(1 for d in z if d < z_true) / len(z)


def key_z(score: float, null_mean: float, null_sd: float) -> float | None:
    if null_sd is None or null_sd <= 0 or score is None:
        return None
    return (score - null_mean) / null_sd


def build_secondary(all_key_scores: dict[str, dict[str, float]], true_keys: list[str], decoys: list[str] | None = None,
                    cohort: str = "generator") -> dict:
    """Secondary-detector assets from a null cohort scored under every target key and decoy.

    ``all_key_scores[text_hash][key_id]`` = score of an eligible unwatermarked text.  Produces per-key null
    mean/sd (for z-scoring), the decoy list, and sorted null arrays of the two statistics (fraction of decoys
    beaten; z(true) - mean z(decoys)) computed with every target key in turn as the "true" key.  Without a
    separate decoy bank the other target keys serve as decoys.
    """
    import numpy as np
    bank = list(decoys) if decoys else list(true_keys)
    all_ids = list(dict.fromkeys(list(true_keys) + bank))
    stats = {}
    for k in all_ids:
        vals = [s[k] for s in all_key_scores.values() if k in s and s[k] is not None and math.isfinite(s[k])]
        stats[k] = {"mean": float(np.mean(vals)) if vals else None, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "n": len(vals)}
    rank_null, zdiff_null = [], []
    for th, s in all_key_scores.items():
        z = {k: key_z(s.get(k), stats[k]["mean"], stats[k]["sd"]) for k in all_ids if stats[k]["sd"]}
        for k in true_keys:
            zt = z.get(k)
            if zt is None:
                continue
            zd = [z[d] for d in bank if d != k and z.get(d) is not None]
            rf = rank_statistic(zt, zd)
            if rf is not None:
                rank_null.append(rf); zdiff_null.append(zt - float(np.mean(zd)))
    return {"cohort": cohort, "decoys": bank, "decoy_bank": bool(decoys), "null_stats": stats, "rank_frac_null": sorted(rank_null),
            "zdiff_null": sorted(zdiff_null), "n_null_texts": len(all_key_scores),
            "note": "SECONDARY detector (amendment): correct key ranked against decoy keys on the same text; "
                    "null values from the same texts under every target key are correlated"}


def detect_secondary(scores_by_key: dict[str, float | None], key_id: str, cal: Calibration) -> dict:
    """Rank-based decision from the text's scores under the true key and every decoy in the calibration."""
    sec = cal.secondary
    if not sec or key_id not in sec.get("null_stats", {}):
        return {"status": "no_secondary_calibration"}
    st = sec["null_stats"]
    zt = key_z(scores_by_key.get(key_id), st[key_id]["mean"], st[key_id]["sd"])
    zd = [key_z(scores_by_key.get(d), st[d]["mean"], st[d]["sd"]) for d in sec["decoys"] if d != key_id and d in st and st[d]["sd"]]
    zd = [v for v in zd if v is not None]
    if zt is None or not zd:
        return {"status": "insufficient_scores"}
    import numpy as np
    rf = rank_statistic(zt, zd)
    zdiff = zt - float(np.mean(zd))
    p_rank = tail_rank_p(sec["rank_frac_null"], rf)
    p_z = tail_rank_p(sec["zdiff_null"], zdiff)
    return {"status": "ok", "rank_frac": rf, "zdiff": zdiff, "p_rank_frac": p_rank, "p_zdiff": p_z, "n_decoys": len(zd),
            "decision_rank_frac": "detected" if p_rank <= cal.alpha else "not_detected",
            "decision_zdiff": "detected" if p_z <= cal.alpha else "not_detected",
            "min_achievable_p_rank_frac": tail_rank_p(sec["rank_frac_null"], 1.0)}


def bonferroni_multi_key(p_finals: dict[str, float], alpha: float) -> dict:
    """Optional pre-specified multi-key search: flag only if min p * K <= alpha (never the unadjusted best)."""
    K = len(p_finals)
    if K == 0:
        return {"decision": "not_detected", "K": 0}
    best = min(p_finals.items(), key=lambda kv: kv[1])
    adj = min(1.0, best[1] * K)
    return {"decision": "detected" if adj <= alpha else "not_detected", "best_key": best[0], "p_adjusted": adj, "K": K}
