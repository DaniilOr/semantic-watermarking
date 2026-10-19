"""Frozen steering scales ``sigma`` (spec Section 5) and the ``fit-scorer`` stage.

sigma is the median within-pool standard deviation of gains over independent,
unwatermarked candidate pools, grouped by scorer, key family, generation-unit
setting, objective and the current output-length bin.  Bins: [0,64), [64,128),
[128,256), [256,inf).  A bin needs >= 30 independent source groups; otherwise
the nearest populated bin is used and the fallback recorded.  The minimum usable
scale is max(1e-6, 100 x observed numerical discrepancy); below it the bin is
marked ``inactive_no_signal`` and selection is uniform.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import numpy as np

from .records import content_hash

LENGTH_BINS: list[tuple[int, float, str]] = [
    (0, 64, "[0,64)"),
    (64, 128, "[64,128)"),
    (128, 256, "[128,256)"),
    (256, math.inf, "[256,inf)"),
]
BIN_NAMES = [b[2] for b in LENGTH_BINS]
MIN_POOLS_PER_BIN = 30
ABSOLUTE_FLOOR = 1e-6
FLOOR_MULTIPLIER = 100.0


def length_bin(accepted_len: int) -> str:
    for lo, hi, name in LENGTH_BINS:
        if lo <= accepted_len < hi:
            return name
    return BIN_NAMES[-1]


def unit_setting_name(unit: str, unit_tokens: int, sentence_cap: int) -> str:
    return f"tokens{unit_tokens}" if unit == "tokens" else f"sentence{sentence_cap}"


def entry_key(family: str, unit_setting: str, objective: str, bin_name: str) -> str:
    return f"{family}|{unit_setting}|{objective}|{bin_name}"


@dataclass
class ScaleEntry:
    sigma: float | None
    status: str                       # ok | inactive_no_signal | missing
    n_pools: int
    fallback_from: str | None = None  # bin whose pools were used, if not this bin
    raw_median_sd: float | None = None


@dataclass
class ScaleTable:
    scorer_id: str
    fitted_on: str                    # partition name
    toy: bool
    floor: float
    numerical_discrepancy: float
    entries: dict[str, dict] = field(default_factory=dict)
    version: str = "scales_v1"
    meta: dict = field(default_factory=dict)

    def lookup(self, family: str, unit_setting: str, objective: str, accepted_len: int) -> ScaleEntry:
        k = entry_key(family, unit_setting, objective, length_bin(accepted_len))
        e = self.entries.get(k)
        if e is None:
            return ScaleEntry(sigma=None, status="missing", n_pools=0)
        return ScaleEntry(**e)

    def to_dict(self) -> dict:
        d = {"scorer_id": self.scorer_id, "fitted_on": self.fitted_on, "toy": self.toy, "floor": self.floor,
             "numerical_discrepancy": self.numerical_discrepancy, "entries": self.entries, "version": self.version,
             "meta": self.meta}
        d["content_hash"] = content_hash({k: v for k, v in d.items()})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScaleTable":
        return cls(scorer_id=d["scorer_id"], fitted_on=d["fitted_on"], toy=bool(d["toy"]), floor=float(d["floor"]),
                   numerical_discrepancy=float(d["numerical_discrepancy"]), entries=dict(d["entries"]),
                   version=d.get("version", "scales_v1"), meta=d.get("meta", {}))


def compute_floor(numerical_discrepancy: float) -> float:
    return max(ABSOLUTE_FLOOR, FLOOR_MULTIPLIER * float(numerical_discrepancy))


def fit_scale_table(pool_sds: dict[str, list[tuple[str, float]]], scorer_id: str, fitted_on: str,
                    numerical_discrepancy: float, toy: bool, min_pools: int = MIN_POOLS_PER_BIN,
                    meta: dict | None = None) -> ScaleTable:
    """Build a table from per-entry lists of (group_id, within-pool SD).

    ``pool_sds`` maps ``entry_key(family, unit_setting, objective, bin)`` to a list of
    (source group id, SD of gains within one candidate pool).  Pools of the same
    source group count once toward the independent-group requirement.
    """
    floor = compute_floor(numerical_discrepancy)
    table = ScaleTable(scorer_id=scorer_id, fitted_on=fitted_on, toy=toy, floor=floor,
                       numerical_discrepancy=float(numerical_discrepancy), meta=meta or {})
    # Index available bins per (family, unit, objective)
    by_prefix: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for k, sds in pool_sds.items():
        family, unit, objective, b = k.split("|")
        by_prefix.setdefault(f"{family}|{unit}|{objective}", {})[b] = sds
    for prefix, bins in by_prefix.items():
        for bi, bname in enumerate(BIN_NAMES):
            sds = bins.get(bname, [])
            groups = {g for g, _ in sds}
            fallback = None
            if len(groups) < min_pools:
                # nearest populated bin by index distance (ties -> lower bin)
                best = None
                for bj, other in enumerate(BIN_NAMES):
                    osds = bins.get(other, [])
                    if len({g for g, _ in osds}) >= min_pools:
                        dist = abs(bj - bi)
                        if best is None or dist < best[0]:
                            best = (dist, other, osds)
                if best is not None:
                    fallback, sds = best[1], best[2]
            if not sds:
                table.entries[f"{prefix}|{bname}"] = ScaleEntry(sigma=None, status="missing", n_pools=0,
                                                                fallback_from=None).__dict__
                continue
            if len({g for g, _ in sds}) < min_pools:
                # neither this bin nor any other bin reaches the independent-group requirement: not usable
                table.entries[f"{prefix}|{bname}"] = ScaleEntry(sigma=None, status="insufficient_pools",
                                                                n_pools=len({g for g, _ in sds}), fallback_from=fallback,
                                                                raw_median_sd=float(statistics.median([s for _, s in sds]))).__dict__
                continue
            med = float(statistics.median([s for _, s in sds]))
            if not math.isfinite(med) or med < floor:
                table.entries[f"{prefix}|{bname}"] = ScaleEntry(sigma=None, status="inactive_no_signal",
                                                                n_pools=len({g for g, _ in sds}),
                                                                fallback_from=fallback, raw_median_sd=med).__dict__
            else:
                table.entries[f"{prefix}|{bname}"] = ScaleEntry(sigma=med, status="ok",
                                                                n_pools=len({g for g, _ in sds}),
                                                                fallback_from=fallback, raw_median_sd=med).__dict__
    return table


def within_pool_sd(values: list[float]) -> float:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return float("nan")
    return float(v.std(ddof=1))
