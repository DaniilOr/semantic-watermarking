"""Quality against the clean generator's own variation (model-free)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import quality_vs_clean as q  # noqa: E402


def J(major: bool, flaws=(), on="yes", quality=3):
    return {"severity": "major" if major else "minor", "flaws": list(flaws), "addresses_task": on, "quality": quality, "note": ""}


def write(run: Path, rows):
    (run / "eval").mkdir(parents=True)
    with open(run / "eval" / f"api_pointwise_{q.SLUG}.jsonl", "w") as fh:
        for arm, seed, pid, j in rows:
            fh.write(json.dumps({"arm": arm, "seed": seed, "prompt_id": pid, "judgment": j}) + "\n")


def test_an_arm_like_clean_is_inside_the_band_and_a_damaged_arm_is_not(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    P = [f"p{i}" for i in range(60)]
    clean = [("base", s, p, J(rng.random() < 0.6)) for s in (0, 1, 2, 3) for p in P]          # clean: 60% major flaws
    same = [("wm_gentle", 0, p, J(rng.random() < 0.6)) for p in P]                            # no worse than clean
    hurt = [("wm_strong", 0, p, J(rng.random() < 0.95, ["drifts_off_topic"])) for p in P]     # clearly worse
    write(tmp_path / "c", clean); write(tmp_path / "m", same + hurt)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["x", "--clean", f"{tmp_path / 'c'}:base", "--marked", str(tmp_path / "m"), "--out", str(out)])
    q.main()
    r = json.loads(out.read_text())
    assert r["n_clean_draws"] == 4 and r["n_prompts"] == 60
    mf = r["metrics"]["major_flaw"]
    assert 0.45 < mf["clean_mean"] < 0.75 and mf["clean_band95"][0] < mf["clean_mean"] < mf["clean_band95"][1]
    assert mf["arms"]["m/wm_gentle"]["inside_clean_band"] and not mf["arms"]["m/wm_gentle"]["significantly_worse"]
    assert not mf["arms"]["m/wm_strong"]["inside_clean_band"] and mf["arms"]["m/wm_strong"]["significantly_worse"]
    assert r["metrics"]["insertions_or_drift"]["arms"]["m/wm_strong"]["significantly_worse"]
