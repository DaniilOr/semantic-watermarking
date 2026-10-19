"""The pre-registered selection rules, applied mechanically (docs/preregistration.md)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from select_config import choose_arm, choose_detector  # noqa: E402


def dev(tprs: dict, fprs: dict) -> dict:
    return {"runs": {"r": {"variants": {v: {"wm": {"tpr": t}, "base": {"fpr": fprs.get(v, 0.01), "n_pairs": 1000,
                                                                         "fp": int(1000 * fprs.get(v, 0.01))}} for v, t in tprs.items()}}}}


def test_a_variant_must_beat_plain_T_by_five_points():
    assert choose_detector(dev({"T": 0.60, "matched_gate": 0.63}, {}))["chosen"] == "T"
    assert choose_detector(dev({"T": 0.60, "matched_gate": 0.70}, {}))["chosen"] == "matched_gate"


def test_a_variant_that_leaks_false_positives_is_not_eligible():
    assert choose_detector(dev({"T": 0.60, "z_gate": 0.90}, {"z_gate": 0.05}))["chosen"] == "T"


def arms(rows):
    t = {"runs": {"tradeoff": {"variants": {"T": {a: {"tpr": r[0], "tpr_ci": [0, 1], "auroc": 0.9} for a, r in rows.items()}}}}}
    j = {"per_arm": {a: {"win_rate": r[1], "major_flaw_delta": r[2], "on_task_delta": 0.0} for a, r in rows.items()}}
    return t, j


def test_most_detectable_arm_among_those_that_pass_the_quality_bar():
    t, j = arms({"wm_c20": (0.95, 0.30, 0.25), "wm_c12": (0.80, 0.45, 0.05), "wm_c8": (0.60, 0.48, 0.02)})
    r = choose_arm(t, j, "T")
    assert r["chosen"] == "wm_c12" and not r["table"]["wm_c20"]["passes_quality_bar"]


def test_fallback_is_flagged_when_nothing_passes():
    t, j = arms({"wm_c20": (0.95, 0.30, 0.25), "wm_c12": (0.80, 0.35, 0.20)})
    r = choose_arm(t, j, "T")
    assert r["chosen"] == "wm_c12" and r["how"].startswith("NO arm passed")


def test_write_json_survives_parallel_writers(tmp_path):
    """Parallel shards write the same manifest at the same moment; a shared temp name made one of them crash."""
    from concurrent.futures import ThreadPoolExecutor
    from wmworld.records import read_json, write_json
    target = tmp_path / "manifest.json"
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda i: write_json(target, {"writer": i, "payload": list(range(2000))}), range(200)))
    assert "writer" in read_json(target) and not list(tmp_path.glob("*.tmp"))
