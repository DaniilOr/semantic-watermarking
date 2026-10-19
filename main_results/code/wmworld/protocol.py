"""``freeze``: record an explicit, human-chosen configuration selection (spec Section 8, stage 5).

The command refuses to choose automatically.  It validates the selection
(at most ``max_confirmation_configs`` arms/overrides, families to draw fresh
confirmation keys for, a written rationale), snapshots every frozen input
(config, scorer wrapper, key construction, attack instructions, quality
criteria, preprocessing versions, development-fitted scales) and writes a
content-hashed protocol manifest.  Confirmation commands run from it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .config import Config
from .key_catalog import CATALOG_VERSION, catalog_dict
from .generation import ARM_PRESETS, arm_config
from .keys import EXTRA_FAMILIES, FAMILIES
from .records import content_hash, read_json, write_json
from .scoring import SCORER_ID
from .textnorm import TEXTNORM_VERSION

PARAPHRASE_INSTRUCTIONS = {
    "light": ("Rewrite the following text, changing the wording while preserving its meaning, facts, names, numbers, "
              "qualifications and approximate length. Output only the rewritten text.\n\n{text}"),
    "strong": ("Substantially rewrite the following text using different sentence structure and vocabulary while preserving "
               "all of its content, facts, names and numbers. Keep the length between 80% and 120% of the original. "
               "Output only the rewritten text.\n\n{text}"),
}
JUDGE_INSTRUCTION = (
    "You are grading a response to a task. Task:\n\n{task}\n\nResponse:\n\n{response}\n\n"
    "Rate the response on a 1-5 scale for each of: relevance to the task, coherence, absence of repetition, "
    "preservation of facts and stated constraints, and overall usefulness. Reply with JSON only, like "
    '{{"relevance": 4, "coherence": 4, "repetition": 5, "constraints": 4, "overall": 4, "off_topic_subject": ""}} '
    "where off_topic_subject names any subject the response discusses that the task did not ask about, or an empty string."
)
QUALITY_TOLERANCES = {"overall_1to5_max_decline": 0.25, "ifeval_max_decline_pp": 5.0,
                      "note": "research decision thresholds (spec Section 11), not established standards"}


def freeze_protocol(cfg: Config, run_dir: Path, output: Path, selected: list[dict] | None, rationale: str, log=print) -> Path:
    results_path = run_dir / "eval" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} missing: evaluate the development run before freezing")
    results = read_json(results_path)
    max_cfg = int(cfg["evaluation"]["max_confirmation_configs"])
    if not selected:
        summary = {arm: {"paired_shift": a["paired_shift_vs_base"]["mean"], "n": a["n"]} for arm, a in results["per_arm"].items()}
        raise SystemExit("freeze requires an explicit --select JSON list (no automatic selection). Development summary for reference:\n"
                         + json.dumps(summary, indent=2) + "\nExample: --select '[{\"arm\": \"llm_global\", \"overrides\": {}, "
                         "\"families\": [\"common_single\", \"rare_single\"]}]' --rationale '...'")
    if len(selected) > max_cfg:
        raise ValueError(f"at most {max_cfg} configurations may be frozen (got {len(selected)})")
    if not rationale or len(rationale.strip()) < 20:
        raise ValueError("a written selection rationale (>= 20 characters) is required")
    families: list[str] = []
    names: list[str] = []
    for s in selected:
        if s.get("arm") not in ARM_PRESETS:
            raise ValueError(f"unknown arm in selection: {s}")
        name = s.get("name") or s["arm"]
        if name in names or name == "base":
            raise ValueError(f"configuration names must be unique and not 'base' (got {name!r}); give each selection a 'name'")
        names.append(name)
        for f in s.get("families", []) or []:
            if f not in FAMILIES + EXTRA_FAMILIES:
                raise ValueError(f"unknown family {f}")
            if f not in families:
                families.append(f)
    if not families:
        families = list(cfg["keys"]["families"])
    run_manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    dev_stage = run_manifest.get("stage") if run_manifest.get("stage") in cfg.raw.get("stages", {}) else "dev"
    scales_path = cfg.resolve_path(cfg.stage(dev_stage).get("scales", "assets/scales.json"))
    if not scales_path.exists():
        raise FileNotFoundError(f"development scales {scales_path} missing (fit-scorer on scale_fit)")
    data_manifest = read_json(cfg.path_of("data_prepared") / "manifest.json")
    dev_keys = read_json(cfg.path_of("assets") / "keys_development.json")
    # the frozen config runs exactly the selected configurations in the confirmation stages: named variants become arm
    # aliases with their overrides; the key families are the selected ones (fresh confirmation keys are drawn from them)
    frozen = copy.deepcopy(cfg.raw)
    frozen.setdefault("arm_aliases", {}); frozen.setdefault("arm_overrides", {})
    resolved = []
    for s, name in zip(selected, names):
        if name != s["arm"]:
            frozen["arm_aliases"][name] = s["arm"]
        frozen["arm_overrides"][name] = {**cfg.arm_overrides(s["arm"]), **(s.get("overrides") or {})}
        ARM_PRESETS.setdefault(name, dict(ARM_PRESETS[s["arm"]]))
        resolved.append({**s, "name": name, "resolved_generation": arm_config(name, cfg.generation_config("positive_test"),
                                                                                frozen["arm_overrides"][name]).to_dict()})
    frozen["arms"] = ["base"] + names
    frozen.setdefault("stages", {}).setdefault("positive_test", {})["arms"] = ["base"] + names
    frozen["keys"]["families"] = families
    proto = {
        "protocol_version": "protocol_v1",
        "config": frozen, "config_path": cfg.path, "config_hash": cfg.content_hash(),
        "selected_configurations": resolved, "selected_families": families, "selection_rationale": rationale.strip(),
        "development_results_hash": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "development_run": str(run_dir),
        "scorer": {"scorer_id": SCORER_ID, "system_template": cfg["scoring"]["system_template"], "user_message": cfg["scoring"]["user_message"],
                   "neutral_prefix": cfg["keys"]["neutral_prefix"], "formula": "F_k(y) = (ell_k(y) - ell_k0(y)) / n_y; F_k('') = 0",
                   "textnorm_version": TEXTNORM_VERSION, "max_input_tokens": cfg["scoring"]["max_input_tokens"]},
        "key_construction": {"catalog_version": CATALOG_VERSION, "catalog_hash": content_hash(catalog_dict()),
                             "rarity_definition": dev_keys.get("rarity_definition"), "development_keys_hash": dev_keys.get("content_hash"),
                             "confirmation_per_selected_family": cfg["keys"]["confirmation_per_selected_family"]},
        "scales": {"path": str(scales_path), "sha256": hashlib.sha256(scales_path.read_bytes()).hexdigest(),
                   "note": "development-fitted family-level scales reused at confirmation (spec Section 8)"},
        "attacks": {"paraphrase_instructions": PARAPHRASE_INSTRUCTIONS, "paraphrase_decoding": {
            "temperature": cfg["attacks"].get("paraphrase_temperature", 0.7), "top_p": cfg["attacks"].get("paraphrase_top_p", 0.9),
            "max_new_tokens": cfg["attacks"].get("paraphrase_max_new_tokens", 640)}, **{k: v for k, v in cfg["attacks"].items()}},
        "quality": {"judge_instruction": JUDGE_INSTRUCTION, "tolerances": QUALITY_TOLERANCES,
                    "human_review": "pending unless reviewers are available; model judge shares weights with the generator"},
        "preprocessing": {"data_manifest_hash": data_manifest.get("content_hash"), "preprocessing_version": data_manifest.get("preprocessing_version"),
                          "dedup_version": data_manifest.get("dedup_version")},
        "model": cfg["models"]["shared"],
    }
    proto["content_hash"] = content_hash(proto)
    write_json(output, proto)
    log(f"[freeze] protocol {proto['content_hash']} with {len(selected)} configuration(s), families {families}")
    return output
