"""Keys (Section 6), data isolation (Section 14 item 7), records/resume, config validation."""

import json
from pathlib import Path

import pytest

from wmworld.config import DEFAULTS, Config, load_config
from wmworld.data.dedup import group_duplicates
from wmworld.data.prepare import PARTITION_ORDER, prepare_domain
from wmworld.data.sources import SourceExample, looks_english, wp_clean_prompt, wp_detokenize
from wmworld.key_catalog import COMPLETIONS, TEMPLATES, validate_catalog
from wmworld.keys import FAMILIES, assign_keys, build_keys, guard_keys, load_keys, keys_manifest, score_catalog_rarity
from wmworld.records import JsonlWriter, existing_ids, load_jsonl, repair_jsonl
from wmworld.scoring import LikelihoodScorer
from wmworld.textnorm import normalize_text, text_hash


def test_catalog_valid():
    validate_catalog()
    assert all(len(v) >= 100 for v in COMPLETIONS.values())


def test_build_keys_families_and_wrong_key_ids(lm):
    validate_catalog()
    # restrict catalog to keep the fake rarity scoring fast
    small_t = {k: TEMPLATES[k] for k in list(TEMPLATES)[:4]}
    small_c = {k: COMPLETIONS[k][:30] for k in small_t}
    rows = score_catalog_rarity(lm, small_t, small_c)
    assert all(r.completion_tokens >= 1 and r.quantile is not None for r in rows)
    import wmworld.keys as K
    orig = K.TEMPLATES
    K.TEMPLATES = small_t
    try:
        keys = build_keys(lm, "development", 2, 1, rows, "rev0")
    finally:
        K.TEMPLATES = orig
    fams = [k.family for k in keys]
    for f in FAMILIES:
        assert fams.count(f) == 2, f
    for k in keys:
        assert k.token_length > 0
        assert all(o != k.key_id for o in k.wrong_key_ids)
        assert all(any(x.key_id == o and x.family == k.family and x.length_stratum == k.length_stratum for x in keys) for o in k.wrong_key_ids)
        if k.family.endswith("multi"):
            assert len(set(k.template_ids)) == 4
        if k.family == "random_digits":
            assert k.prefix.isdigit()
        if k.family.startswith("rare"):
            assert all(r["quantile"] >= 0.9 for r in k.rarity)
        if k.family.startswith("common"):
            assert all(r["quantile"] <= 0.2 for r in k.rarity)
    assert len({k.prefix for k in keys}) == len(keys)
    # guard + manifest + load
    guard_keys(lm, keys, LikelihoodScorer(lm))
    m = keys_manifest(keys, "development", 1, {"name": "fake"}, rows)
    loaded = load_keys(json.loads(json.dumps(m)))
    assert [k.key_id for k in loaded] == [k.key_id for k in keys]
    # keys re-drawn with the same seed are identical; different seed differs
    K.TEMPLATES = small_t
    try:
        again = build_keys(lm, "development", 2, 1, rows, "rev0")
        other = build_keys(lm, "development", 2, 2, rows, "rev0")
    finally:
        K.TEMPLATES = orig
    assert [k.prefix for k in again] == [k.prefix for k in keys]
    assert [k.prefix for k in other] != [k.prefix for k in keys]


def test_assignment_balanced_and_deterministic():
    prompts = [f"p{i}" for i in range(150)]
    keys = [f"k{i}" for i in range(20)]
    a = assign_keys(prompts, keys, 1, "dev")
    b = assign_keys(prompts, keys, 1, "dev")
    assert a == b
    counts = {}
    for v in a.values():
        counts[v] = counts.get(v, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1
    assert assign_keys(prompts, keys, 2, "dev") != a


# ---- data --------------------------------------------------------------------- #

def test_wp_cleaning():
    assert wp_clean_prompt("[ WP ] You wake up , and it ' s dark .") == "You wake up, and it's dark."
    s = wp_detokenize("He said `` hello '' . <newline> <newline> Then he left .")
    assert s == 'He said "hello".\n\nThen he left.'
    assert looks_english("the cat and the dog went to the park and it was a good day for them all")
    assert not looks_english("ñandú ñandú ñandú ñandú ñandú ñandú ñandú ñandú ñandú ñandú")


def test_dedup_groups_exact_and_near_duplicates():
    base = " ".join(f"word{i}" for i in range(200))
    near = " ".join(f"word{i}" for i in range(198)) + " extra token"
    other = " ".join(f"other{i}" for i in range(200))
    unrelated = " ".join(f"zeta{i}" for i in range(200))
    g = group_duplicates(["p1", "p2", "p3", "P1 ", "p5"], [base, near, other, unrelated, other + " x"])
    assert g[0] == g[1]              # near-duplicate docs
    assert g[0] == g[3]              # exact normalized prompt duplicate (different doc)
    assert g[2] == g[4]              # near-duplicate docs with different prompts
    assert g[2] != g[0]


def _ex(i, split="train", ref_words=100):
    return SourceExample(source_id=f"s{i}", domain="dolly", source_split=split, prompt=f"task {i}",
                         prompt_core=f"task {i} unique {i * 7919}", reference=" ".join(f"w{j}{i}" for j in range(ref_words)),
                         doc_text=" ".join(f"doc{i}_{j}" for j in range(80)), meta={"filter_reasons": []})


def test_partitions_disjoint_and_shortfall_raises(lm):
    exs = [_ex(i) for i in range(40)]
    budgets = {"smoke": 2, "scale_fit": 3, "dev": 5, "null_calibration": 6, "null_test": 6, "positive_test": 4, "donor_pool": 3, "ablation": 2}
    prepared, report = prepare_domain("dolly", exs, budgets, lm.encode_text, lm.decode, 1, reference_cap_tokens=50, min_reference_tokens=8)
    parts = {}
    for p in prepared:
        parts.setdefault(p.partition, set()).add(p.group_id)
    for a in PARTITION_ORDER:
        for b in PARTITION_ORDER:
            if a != b:
                assert not (parts.get(a, set()) & parts.get(b, set()))
    assert sum(1 for p in prepared if p.ablation_subset) == 2 and all(p.partition == "dev" for p in prepared if p.ablation_subset)
    assert all(p.reference_tokens <= 50 for p in prepared)
    with pytest.raises(RuntimeError):
        prepare_domain("dolly", exs[:5], budgets, lm.encode_text, lm.decode, 1, min_reference_tokens=8)
    prepared2, report2 = prepare_domain("dolly", exs[:5], budgets, lm.encode_text, lm.decode, 1, min_reference_tokens=8, allow_shortfall=True)
    assert report2["shortfall"]


# ---- records / resume ------------------------------------------------------------- #

def test_jsonl_writer_idempotent_resume(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    with JsonlWriter(p) as w:
        assert w.write({"id": "a", "v": 1})
        assert not w.write({"id": "a", "v": 2})
        assert w.write({"id": "b", "v": 1})
    with JsonlWriter(p) as w:
        assert "a" in w and "b" in w
        assert not w.write({"id": "b", "v": 9})
    rows = load_jsonl(p)
    assert [r["id"] for r in rows] == ["a", "b"] and rows[0]["v"] == 1
    p.write_text(p.read_text() + '{"id": "c", "v"')          # crash mid-line
    with pytest.raises(ValueError):
        load_jsonl(p)
    assert repair_jsonl(p) == 2
    assert existing_ids(p) == {"a", "b"}


def test_textnorm():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    assert normalize_text("  keep   spaces \n") == "  keep   spaces \n"
    assert text_hash("x\r\n") == text_hash("x\n")


# ---- config ----------------------------------------------------------------------- #

def test_config_rejects_other_models_and_thinking(tmp_path: Path):
    good = tmp_path / "c.yaml"
    good.write_text("models:\n  shared:\n    revision: abc\n")
    cfg = load_config(good)
    assert cfg.model_revision == "abc"
    bad = tmp_path / "b.yaml"
    bad.write_text("models:\n  shared:\n    id: Qwen/Qwen3.5-2B\n    revision: abc\n")
    with pytest.raises(ValueError):
        load_config(bad)
    bad2 = tmp_path / "b2.yaml"
    bad2.write_text("models:\n  shared:\n    enable_thinking: true\n")
    with pytest.raises(ValueError):
        load_config(bad2)
    unresolved = tmp_path / "u.yaml"
    unresolved.write_text("seed: 1\n")
    c = load_config(unresolved)
    with pytest.raises(ValueError):
        c.validate(require_revision=True)


def test_random_words_family(lm):
    from wmworld.keys import build_keys
    keys = build_keys(lm, "randomwords", 6, 5, [], "rev0", families=["random_words"])
    assert len(keys) == 6 and all(k.family == "random_words" for k in keys)
    assert sum(k.length_stratum == "short" for k in keys) == 3 and all(len(k.prefix.split()) in (4, 10) for k in keys)
    assert len({k.prefix for k in keys}) == 6
    again = build_keys(lm, "randomwords", 6, 5, [], "rev0", families=["random_words"])
    assert [k.prefix for k in again] == [k.prefix for k in keys]


def test_random_words_rare_vocab(lm):
    from wmworld.key_catalog import SEED_WORDS, SEED_WORDS_RARE
    from wmworld.keys import build_keys
    assert set(SEED_WORDS_RARE) <= set(SEED_WORDS) and len(SEED_WORDS_RARE) >= 40
    keys = build_keys(lm, "randomwords_rare", 4, 9, [], "rev0", families=["random_words"], vocab=SEED_WORDS_RARE)
    assert len(keys) == 4 and all(w in SEED_WORDS_RARE for k in keys for w in k.prefix.split())


def test_mbpp_loader_and_code_partition(tmp_path: Path, lm):
    import pandas as pd
    from wmworld.data.prepare import load_domain_examples, prepare_domain
    root = tmp_path / "mbpp"; root.mkdir()
    df = pd.DataFrame([{"task_id": i, "text": f"Write a function that returns {i} plus one.", "code": f"def f{i}(x):\n    return x + 1",
                        "test_list": [f"assert f{i}(1) == 2", f"assert f{i}(2) == 3"], "test_setup_code": "", "challenge_test_list": []} for i in range(12)])
    df.to_parquet(root / "full-test.parquet")
    exs = load_domain_examples("mbpp", tmp_path, lm.encode_text, lm.decode, 6000, 1)
    assert len(exs) == 12 and exs[0].prompt.startswith("Write a Python function") and "assert f0(1) == 2" in exs[0].prompt
    prepared, report = prepare_domain("mbpp", exs, {"code": 5}, lm.encode_text, lm.decode, 1, min_reference_tokens=4)
    assert len(prepared) == 5 and all(p.partition == "code" for p in prepared) and prepared[0].meta["test_list"]


def test_code_eval_runs_generated_programs(tmp_path: Path):
    from wmworld.code_eval import extract_code, run_tests
    assert extract_code("here\n```python\ndef f(x):\n    return x\n```\nthanks") == "def f(x):\n    return x"
    assert extract_code("def g(): pass") == "def g(): pass"
    assert run_tests("def f(x):\n    return x + 1", "", ["assert f(1) == 2"], timeout=10)["status"] == "pass"
    assert run_tests("def f(x):\n    return x", "", ["assert f(1) == 2"], timeout=10)["status"] == "fail"
    assert run_tests("while True:\n    pass", "", ["assert True"], timeout=2)["status"] == "timeout"
    assert run_tests("import os\nx = 1", "y = 2", ["assert x + y == 3"], timeout=10)["status"] == "pass"
