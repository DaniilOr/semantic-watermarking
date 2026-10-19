"""End-to-end stage pipeline on the fake LM: prepare -> keys -> generate stage (resume) -> score -> evaluate -> detect CLI assets.

Covers spec Section 14 items 1, 2, 7 (resume idempotency, partition-following) and the detector isolation requirement.
"""

import json
from pathlib import Path

import pytest
import yaml

import wmworld.stages as stages
from wmworld.backend import FakeLM
from wmworld.calibration import Calibration, detect
from wmworld.config import load_config
from wmworld.data.prepare import PreparedExample, write_prepared
from wmworld.evaluation import evaluate_run
from wmworld.key_catalog import COMPLETIONS, TEMPLATES
from wmworld.keys import build_keys, keys_manifest, score_catalog_rarity
from wmworld.records import load_jsonl, read_json, write_json
from wmworld.scales import ScaleTable
from wmworld.scoring import LikelihoodScorer
from wmworld.textnorm import text_hash


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    lm = FakeLM(max_input_tokens=2000)
    monkeypatch.setattr(stages, "build_backend", lambda cfg, device=None: lm)
    root = tmp_path
    (root / "configs").mkdir()
    cfg_yaml = {"models": {"shared": {"revision": "fakerev"}}, "paths": {"root": str(root)},
                "generation": {"unit_tokens": 8, "candidates": 3, "min_new_tokens": 8, "max_new_tokens": 40, "temperature": 1.0, "top_p": 1.0},
                "data": {"domains": ["dolly"], "groups_per_domain": {"smoke": 6, "dev": 6}},
                "evaluation": {"bootstrap_replicates": 50},
                "stages": {"smoke": {"partition": "smoke", "seeds": [0], "keys": "development", "scales": "assets/scales_smoke.json",
                                     "generation": {"max_new_tokens": 40, "min_new_tokens": 8, "candidates": 3}}},
                "runtime": {"require_budget_manifest_for_large_runs": False}}
    (root / "configs" / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_config(root / "configs" / "c.yaml")
    # prepared data: 6 smoke prompts with references
    prepared = []
    for i in range(6):
        ref = " ".join(f"reference word{j} number{i}" for j in range(30))
        prepared.append(PreparedExample(prompt_id=f"dolly:s{i}", group_id=f"g_{i:03d}", source_id=f"s{i}", domain="dolly",
                                        source_split="single", partition="smoke", ablation_subset=False,
                                        prompt=f"Describe object number {i} in detail.", prompt_hash=text_hash(f"p{i}"),
                                        reference=ref, reference_hash=text_hash(ref), reference_tokens=len(lm.encode_text(ref)),
                                        reference_eligible=True, meta={}))
    write_prepared(cfg.path_of("data_prepared"), prepared, [], {}, {})
    # keys (small catalog for speed)
    small_t = {k: TEMPLATES[k] for k in list(TEMPLATES)[:4]}
    small_c = {k: COMPLETIONS[k][:25] for k in small_t}
    rows = score_catalog_rarity(lm, small_t, small_c)
    import wmworld.keys as K
    orig = K.TEMPLATES
    K.TEMPLATES = small_t
    try:
        keys = build_keys(lm, "development", 1, 3, rows, "fakerev")
    finally:
        K.TEMPLATES = orig
    write_json(stages.keys_path(cfg, "development"), keys_manifest(keys, "development", 3, {"name": "fake"}, None))
    # toy scales
    entries = {}
    for k in keys:
        for obj in ("global", "local"):
            for b in ("[0,64)", "[64,128)", "[128,256)", "[256,inf)"):
                entries[f"{k.family}|tokens8|{obj}|{b}"] = {"sigma": 0.2, "status": "ok", "n_pools": 6, "fallback_from": None, "raw_median_sd": 0.2}
    write_json(root / "assets" / "scales_smoke.json", ScaleTable("llm_global_v1", "toy", True, 1e-6, 0.0, entries).to_dict())
    return cfg, lm, keys


def test_stage_generate_resume_score_evaluate(project):
    cfg, lm, keys = project
    run_dir = cfg.path_of("runs") / "smoke"
    items, manifest = stages.enumerate_stage(cfg, "smoke")
    assert manifest["n_items"] == 6 * 5
    # balanced assignment across the 5 keys
    counts = {}
    for v in manifest["assignment"].values():
        counts[v] = counts.get(v, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1
    # partial run then resume: no duplicates, no re-draw
    s1 = stages.run_generate_stage(cfg, "smoke", lm, run_dir, max_items=7, log=lambda *_: None)
    assert s1["done"] == 7
    first = {r["id"]: r["text"] for r in load_jsonl(run_dir / "generations.jsonl")}
    s2 = stages.run_generate_stage(cfg, "smoke", lm, run_dir, log=lambda *_: None)
    assert s2["skipped_existing"] == 7 and s2["done"] == 23
    gens = load_jsonl(run_dir / "generations.jsonl")
    assert len(gens) == 30 and len({g["id"] for g in gens}) == 30
    for g in gens:
        if g["id"] in first:
            assert g["text"] == first[g["id"]]           # resume did not redraw
    # every generation belongs to the stage's partition
    prepared = {p.prompt_id for p in stages.stage_prompts(cfg, "smoke")}
    assert all(g["prompt_id"] in prepared for g in gens)
    refs = load_jsonl(run_dir / "references.jsonl")
    assert len(refs) == 6
    # trajectories exist for steering arms and telescoping holds after reload
    trs = load_jsonl(run_dir / "trajectories.jsonl")
    sc = LikelihoodScorer(lm)
    kmap = {k.key_id: k.to_keyspec() for k in keys}
    for g in gens:
        if g["arm"] == "llm_global":
            t = [x for x in trs if x["output_id"] == g["id"]]
            tot = sum(x["gains"][x["chosen"]] for x in t)
            assert abs(tot - sc.score(g["text"], kmap[g["key_id"]]).score) < 1e-9
    # detection records carry no prompt / label / arm
    res = evaluate_run(cfg, run_dir, score=True, log=lambda *_: None)
    det = load_jsonl(run_dir / "detections_raw.jsonl")
    assert det and all(set(d) <= {"id", "text_hash", "key_id", "scorer_id", "score", "status", "visible_tokens", "eligible",
                                  "contrast_sum", "error", "prefix_views", "t_stat", "frac_pos", "t_n"} for d in det)
    assert set(res["per_arm"]) == set(manifest["arms"])
    assert (run_dir / "eval" / "results.json").exists() and (run_dir / "eval" / "report.md").exists()
    # the steering arm should score higher under the assigned key than base on the fake world
    assert res["per_arm"]["llm_global"]["paired_shift_vs_base"]["mean"] > 0
    # standalone detection from the toy calibration reproduces the evaluation score from text alone
    cal = Calibration.from_dict(read_json(run_dir / "eval" / "calibration_toy.json"))
    g = next(x for x in gens if x["arm"] == "llm_global")
    d = detect(g["text"], cal.keyspec(g["key_id"]), cal, LikelihoodScorer(lm))
    row = next(r for r in det if r["text_hash"] == g["text_hash"] and r["key_id"] == g["key_id"])
    assert d.score == pytest.approx(row["score"])
    # secondary rank detector without a decoy bank falls back to the other target keys
    assert res["rank_detector"]["decoy_bank"] is False and res["rank_detector"]["n_decoys"] == len(keys) - 1
    # with a decoy bank: texts are scored under decoys, and decoys (never targets) are used
    small_t = {k: TEMPLATES[k] for k in list(TEMPLATES)[:4]}
    small_c = {k: COMPLETIONS[k][:25] for k in small_t}
    rows_r = score_catalog_rarity(lm, small_t, small_c)
    import wmworld.keys as K
    orig = K.TEMPLATES
    K.TEMPLATES = small_t
    try:
        decoys = build_keys(lm, "decoy", 2, 9, rows_r, "fakerev", exclude_prefixes={k.prefix for k in keys})
    finally:
        K.TEMPLATES = orig
    write_json(stages.keys_path(cfg, "decoy"), keys_manifest(decoys, "decoy", 9, {"name": "fake"}, None))
    res2 = evaluate_run(cfg, run_dir, score=True, log=lambda *_: None)
    assert res2["rank_detector"]["decoy_bank"] is True and res2["rank_detector"]["n_decoys"] == len(decoys)
    det2 = load_jsonl(run_dir / "detections_raw.jsonl")
    assert {r["key_id"] for r in det2} >= {d.key_id for d in decoys}
    assert not ({d.prefix for d in decoys} & {k.prefix for k in keys})


def test_dry_run_budget_and_estimates(project):
    cfg, lm, keys = project
    items, manifest = stages.enumerate_stage(cfg, "smoke")
    b = stages.budget_manifest(cfg, "smoke", items, manifest, None, None)
    e = b["estimates"]["total"]
    assert e["outputs"] == 30 and e["max_proposal_tokens"] > 0 and e["max_scoring_tokens"] > 0
    assert b["limits"]["max_outputs"] == 30
    # base arm has no scoring cost; steering arms do
    assert b["estimates"]["per_arm"]["base"]["max_scoring_tokens"] == 0
    assert b["estimates"]["per_arm"]["llm_global"]["max_scoring_tokens"] > b["estimates"]["per_arm"]["llm_local"]["max_scoring_tokens"]


def test_overrides_reach_the_generator_config(project):
    """--override values for objective/selection/center_gains must survive the arm preset (regression: they were dropped)."""
    cfg, lm, keys = project
    items, manifest = stages.enumerate_stage(cfg, "smoke", arms=["llm_global"],
                                             config_overrides={"objective": "specific", "selection": "greedy", "center_gains": True,
                                                               "candidates": 5, "standardized_gain_clip": 3.0})
    g = items[0].gen_cfg
    assert g.objective == "specific" and g.selection == "greedy" and g.center_gains is True and g.candidates == 5 and g.clip == 3.0
    assert manifest["arm_configs"]["llm_global"]["selection"] == "greedy"
    # stage-level generation settings also pass through, and unknown keys are rejected
    cfg.raw["stages"]["smoke"]["generation"]["selection"] = "rank"
    assert cfg.generation_config("smoke").selection == "rank"
    cfg.raw["stages"]["smoke"]["generation"]["bogus"] = 1
    import pytest as _pt
    with _pt.raises(ValueError):
        cfg.generation_config("smoke")
    del cfg.raw["stages"]["smoke"]["generation"]["bogus"]
    del cfg.raw["stages"]["smoke"]["generation"]["selection"]


def test_specific_objective_runs_through_the_stage(project):
    """A 'specific' run attaches steering decoys to the key and completes (scale entries missing -> uniform, but no failure)."""
    cfg, lm, keys = project
    small_t = {k: TEMPLATES[k] for k in list(TEMPLATES)[:4]}
    small_c = {k: COMPLETIONS[k][:25] for k in small_t}
    rows_r = score_catalog_rarity(lm, small_t, small_c)
    import wmworld.keys as K
    orig = K.TEMPLATES
    K.TEMPLATES = small_t
    try:
        decoys = build_keys(lm, "decoy", 5, 9, rows_r, "fakerev", exclude_prefixes={k.prefix for k in keys})
    finally:
        K.TEMPLATES = orig
    write_json(stages.keys_path(cfg, "decoy"), keys_manifest(decoys, "decoy", 9, {"name": "fake"}, None))
    run_dir = cfg.path_of("runs") / "smoke_specific"
    s = stages.run_generate_stage(cfg, "smoke", lm, run_dir, arms=["llm_global"], config_overrides={"objective": "specific"}, log=lambda *_: None)
    gens = load_jsonl(run_dir / "generations.jsonl")
    assert s["done"] == 6 and all(g["status"] == "ok" and g["key_id"] for g in gens)
    assert all(g["arm_config"]["objective"] == "specific" for g in gens)
    sd = read_json(cfg.path_of("assets") / "steering_decoys.json")["by_family"]
    assert all(len(v) <= 4 or fam == "random_words" for fam, v in sd.items()) and all(d["prefix"] not in {k.prefix for k in keys} for v in sd.values() for d in v)


def test_base_control_never_receives_overrides(project):
    """--override must not alter the `base` control arm (regression: base was steered in the activation sweep)."""
    cfg, lm, keys = project
    items, manifest = stages.enumerate_stage(cfg, "smoke", arms=["base", "llm_global"],
                                             config_overrides={"steer_alpha": 0.25, "candidates": 7, "selection": "greedy"})
    b = manifest["arm_configs"]["base"]; g = manifest["arm_configs"]["llm_global"]
    assert b["steer_alpha"] == 0.0 and b["candidates"] == 1 and b["selection"] == "uniform" and b["objective"] == "none"
    assert g["steer_alpha"] == 0.25 and g["candidates"] == 7 and g["selection"] == "greedy"
    assert all(it.gen_cfg.steer_alpha == 0.0 for it in items if it.arm == "base")


def test_t_statistic_detector_present(project):
    cfg, lm, keys = project
    run_dir = cfg.path_of("runs") / "smoke_t"
    stages.run_generate_stage(cfg, "smoke", lm, run_dir, arms=["base", "llm_global"], log=lambda *_: None)
    res = evaluate_run(cfg, run_dir, score=True, log=lambda *_: None)
    td = res["t_detector"]
    assert td["null_cohort_size"] == 6 and set(td["per_arm"]) == {"base", "llm_global"}
    assert td["per_arm"]["llm_global"]["t_mean"]["mean"] > td["per_arm"]["base"]["t_mean"]["mean"]
    det = load_jsonl(run_dir / "detections_raw.jsonl")
    assert any("t_stat" in d for d in det) and all(0 <= d["frac_pos"] <= 1 for d in det if "frac_pos" in d)


def test_attack_run_evaluates_with_attacked_controls(project, monkeypatch):
    """Attack runs carry arms '<parent>+<attack>[params]'; evaluation pairs each marked attacked text with the attacked
    base text of the same prompt and attack, and builds its toy null from the attacked base rows."""
    import wmworld.attacks as A
    cfg, lm, keys = project
    monkeypatch.setattr(A, "build_backend", lambda cfg, device=None: lm, raising=False)
    run_dir = cfg.path_of("runs") / "smoke"
    stages.run_generate_stage(cfg, "smoke", lm, run_dir, log=lambda *_: None)
    out = A.run_attacks(cfg, run_name="smoke", subset="crop,resegment", log=lambda *_: None)
    gens = load_jsonl(out / "generations.jsonl")
    arms = {g["arm"] for g in gens}
    assert any(a.startswith("base+crop[") for a in arms) and any(a.startswith("llm_global+crop[") for a in arms)
    cfg.raw.setdefault("detection", {})["min_visible_tokens"] = 8       # 40-token fakes cropped to 50-75% stay eligible
    res = evaluate_run(cfg, out, score=True, log=lambda *_: None)
    crop = sorted(a for a in res["per_arm"] if a.startswith("llm_global+crop[0.75"))
    assert crop
    pa = res["per_arm"][crop[0]]
    assert pa["n"] > 0 and pa["paired_shift_vs_base"]["n"] > 0          # paired within the same attack
    assert res["t_detector"]["null_cohort_size"] > 0                     # attacked base rows form the null
    assert all(res["per_arm"][a]["paired_shift_vs_base"]["mean"] == 0.0 for a in res["per_arm"] if a.startswith("base+"))


def test_random_word_steering_decoys_are_frozen(project):
    from wmworld.key_catalog import SEED_WORDS
    cfg, lm, keys = project
    d = stages.steering_decoys(cfg)
    rw = d["random_words"]
    assert len(rw) == stages.RW_STEERING_DECOY_POOL and all(len(x["prefix"].split()) == 4 for x in rw)
    # per-key selection skips decoys sharing a word with the key
    from wmworld.keys import KeyRecord
    w = rw[0]["prefix"].split()[0]
    k = KeyRecord(key_id="t", key_set="randomwords", family="random_words", prefix=f"{w} zzz yyy xxx", template_ids=[], completions=[],
                  statements=[], seed=0, token_length=4, rarity=[], length_stratum="short", model_revision="r")
    ks = stages.keyspec_with_decoys(cfg, k)
    assert rw[0]["prefix"] not in ks.meta["steering_decoys"] and len(ks.meta["steering_decoys"]) >= 3
    assert all(w in SEED_WORDS for x in rw for w in x["prefix"].split())
    assert stages.steering_decoys(cfg)["random_words"] == rw                     # re-read from the frozen file
    assert (cfg.path_of("assets") / "steering_decoys.json").exists()


def test_freeze_extension_arm_with_named_variants(project):
    """freeze accepts extension arms and the seed-phrase family, and the frozen config runs the named variants."""
    from wmworld.config import config_from_protocol
    from wmworld.generation import arm_config
    from wmworld.protocol import freeze_protocol
    cfg, lm, keys = project
    run_dir = cfg.path_of("runs") / "smoke"
    stages.run_generate_stage(cfg, "smoke", lm, run_dir, log=lambda *_: None)
    evaluate_run(cfg, run_dir, score=True, log=lambda *_: None)
    sel = [{"arm": "poe_steer", "name": "poe_t4", "overrides": {"poe_beta": 10.0, "poe_adapt": "tstat", "poe_t_target": 4.0}, "families": ["random_words"]},
           {"arm": "poe_steer", "name": "poe_t6", "overrides": {"poe_beta": 10.0, "poe_adapt": "tstat", "poe_t_target": 6.0}, "families": ["random_words"]}]
    out = freeze_protocol(cfg, run_dir, cfg.path_of("runs") / "proto.json", sel, "development: adaptive beta 10 detects 95% at 1% FPR", log=lambda *_: None)
    proto = load_jsonl.__globals__["read_json"](out) if False else __import__("json").loads(out.read_text())
    assert proto["selected_families"] == ["random_words"] and proto["config"]["keys"]["families"] == ["random_words"]
    cfg2 = config_from_protocol(proto)
    assert cfg2.stage_arms("positive_test") == ["base", "poe_t4", "poe_t6"]
    g4 = arm_config("poe_t4", cfg2.generation_config("positive_test"), cfg2.arm_overrides("poe_t4"))
    g6 = arm_config("poe_t6", cfg2.generation_config("positive_test"), cfg2.arm_overrides("poe_t6"))
    assert g4.poe_beta == 10.0 and g4.poe_t_target == 4.0 and g6.poe_t_target == 6.0 and g4.candidates == 1
    assert proto["selected_configurations"][0]["resolved_generation"]["poe_t_target"] == 4.0
    with pytest.raises(ValueError):
        freeze_protocol(cfg, run_dir, cfg.path_of("runs") / "p2.json", [{"arm": "poe_steer"}, {"arm": "poe_steer"}], "duplicate names must fail here", log=lambda *_: None)


def test_stage_limit_per_domain_and_arm_aliases(project):
    from wmworld.generation import arm_config
    cfg, lm, keys = project
    cfg.raw["stages"]["smoke"]["limit_per_domain"] = 3
    rows = stages.stage_prompts(cfg, "smoke")
    assert len(rows) == 3
    cfg.raw["stages"]["smoke"]["limit_per_domain"] = 100
    assert len(stages.stage_prompts(cfg, "smoke")) == 6
    cfg.raw["arm_aliases"] = {"wm_x": "poe_steer"}; cfg.raw["arm_overrides"] = {"wm_x": {"poe_beta": 7.0}}
    cfg.validate()
    cfg.raw["stages"]["smoke"]["arms"] = ["base", "wm_x"]
    items, manifest = stages.enumerate_stage(cfg, "smoke")
    assert {it.arm for it in items} == {"base", "wm_x"}
    assert all(it.gen_cfg.poe_beta == 7.0 for it in items if it.arm == "wm_x") and all(it.gen_cfg.poe_beta == 0 for it in items if it.arm == "base")
