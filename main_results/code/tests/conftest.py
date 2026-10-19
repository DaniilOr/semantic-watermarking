import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmworld.backend import FakeLM  # noqa: E402
from wmworld.scales import ScaleTable  # noqa: E402
from wmworld.scoring import NEUTRAL_PREFIX, KeySpec, LikelihoodScorer  # noqa: E402


@pytest.fixture
def lm():
    return FakeLM(max_input_tokens=600)


@pytest.fixture
def scorer(lm):
    return LikelihoodScorer(lm)


@pytest.fixture
def key_a():
    return KeySpec(key_id="A", prefix="You eat chicken every day.", family="common_single")


@pytest.fixture
def key_b():
    return KeySpec(key_id="B", prefix="You collect stamps.", family="common_single")


@pytest.fixture
def key_neutral():
    return KeySpec(key_id="N", prefix=NEUTRAL_PREFIX, family="common_single")


@pytest.fixture
def toy_scales():
    entries = {}
    for fam in ("common_single", "none"):
        for unit in ("tokens8", "tokens16", "sentence64"):
            for obj in ("global", "local"):
                for b in ("[0,64)", "[64,128)", "[128,256)", "[256,inf)"):
                    entries[f"{fam}|{unit}|{obj}|{b}"] = {"sigma": 0.2, "status": "ok", "n_pools": 5,
                                                          "fallback_from": None, "raw_median_sd": 0.2}
    return ScaleTable(scorer_id="llm_global_v1", fitted_on="toy", toy=True, floor=1e-6, numerical_discrepancy=0.0,
                      entries=entries)


def model_available() -> bool:
    if os.environ.get("WMWORLD_RUN_MODEL_TESTS") != "1":
        return False
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False
