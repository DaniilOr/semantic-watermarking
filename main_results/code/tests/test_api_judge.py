"""The API judge's offline logic: parsing, validation and the paired summary (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from api_judge import FLAWS, SCHEMA, parse_json, summarize, valid  # noqa: E402


def J(better, a_ok="yes", b_ok="yes", a_flaws=(), b_flaws=(), a_sev="none", b_sev="none", same=True):
    return {"a": {"addresses_task": a_ok, "flaws": list(a_flaws), "severity": a_sev, "note": "ok"},
            "b": {"addresses_task": b_ok, "flaws": list(b_flaws), "severity": b_sev, "note": "ok"},
            "same_question": same, "better": better, "margin": "slight"}


def test_parse_and_validate():
    assert valid(parse_json('```json\n' + __import__("json").dumps(J("A")) + '\n```'))
    assert valid(parse_json("noise before " + __import__("json").dumps(J("tie")) + " noise after"))
    assert parse_json("not json") is None and not valid({"better": "C"}) and not valid(None)
    assert set(SCHEMA["schema"]["properties"]["a"]["properties"]["flaws"]["items"]["enum"]) == set(FLAWS)


def test_summary_unswaps_positions_and_pairs_with_the_clean_text():
    rows = []
    for i in range(10):
        # the watermarked text is worse and flawed, whichever slot it is shown in
        rows.append({"prompt_id": f"p{i}", "wm_is": "A", "judgment": J("B", a_flaws=["repetition_or_loops"], a_sev="major")})
        rows.append({"prompt_id": f"p{i}", "wm_is": "B", "judgment": J("A", b_flaws=["repetition_or_loops"], b_sev="major")})
    s = summarize(rows)
    assert s["n_prompts"] == 10 and s["win_rate"] == 0.0 and s["order_consistency"] == 1.0
    assert s["major_flaw_wm"] == 1.0 and s["major_flaw_clean"] == 0.0 and s["major_flaw_delta"] == 1.0
    assert s["flaws_wm_per_100"] == {"repetition_or_loops": 100.0} and s["flaws_clean_per_100"] == {}


def test_pure_position_bias_nets_out_to_a_coin_flip():
    rows = []
    for i in range(10):                      # a judge that always says "A": no information, and the summary shows it
        rows.append({"prompt_id": f"p{i}", "wm_is": "A", "judgment": J("A")})
        rows.append({"prompt_id": f"p{i}", "wm_is": "B", "judgment": J("A")})
    s = summarize(rows)
    assert s["win_rate"] == 0.5 and s["order_consistency"] == 0.0


def test_off_task_and_different_question_are_reported():
    rows = [{"prompt_id": "p0", "wm_is": "A", "judgment": J("B", a_ok="no", same=False)},
            {"prompt_id": "p0", "wm_is": "B", "judgment": J("A", b_ok="partly", same=False)}]
    s = summarize(rows)
    assert s["on_task_wm"] == 0.25 and s["on_task_clean"] == 1.0 and s["same_question"] == 0.0


def test_preference_only_summary_unswaps_the_two_orders():
    from api_judge import summarize_preference
    rows = []
    for i in range(8):                                   # watermarked loses in both slots
        rows += [{"prompt_id": f"p{i}", "wm_is": "A", "judgment": {"better": "B"}}, {"prompt_id": f"p{i}", "wm_is": "B", "judgment": {"better": "A"}}]
    for i in range(8, 12):                               # a judge that always says "A": nets out to 0.5, and is shown as inconsistent
        rows += [{"prompt_id": f"p{i}", "wm_is": "A", "judgment": {"better": "A"}}, {"prompt_id": f"p{i}", "wm_is": "B", "judgment": {"better": "A"}}]
    s = summarize_preference(rows)
    assert s["n_prompts"] == 12 and abs(s["win_rate"] - (8 * 0.0 + 4 * 0.5) / 12) < 1e-9
    assert abs(s["order_consistency"] - 8 / 12) < 1e-9 and s["tie_share"] == 0.0
