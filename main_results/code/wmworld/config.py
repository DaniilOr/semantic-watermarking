"""Configuration loading and validation (spec Section 12).

The YAML schema from the specification is the base; this module adds the
frozen scorer wrapper, per-arm overrides, paths, runtime and per-stage settings.
Every model-backed role must reference only ``Qwen/Qwen3.5-0.8B`` and an
immutable revision.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .generation import ARM_PRESETS, GenerationConfig
from .records import content_hash
from .runtime import ALLOWED_MODEL_ID, RuntimeConfig
from .scoring import NEUTRAL_PREFIX, SCORER_ID, SCORING_SYSTEM_TEMPLATE, SCORING_USER

DEFAULTS: dict[str, Any] = {
    "experiment": "prefix_world_semantic_watermark_qwen35_08b_v11",
    "seed": 20260911,
    "models": {
        # float32 (not bf16): measured bf16 batched-vs-single score discrepancy was up to 0.037 nats/token, which under
        # the spec's floor rule (100 x discrepancy) would disable steering; fp32 agrees to ~2e-6 at ~1.5x scoring cost.
        "shared": {"id": ALLOWED_MODEL_ID, "revision": "RESOLVE_AND_FREEZE", "model_class": "Qwen3_5ForCausalLM",
                   "text_only": True, "enable_thinking": False, "dtype": "float32"},
        "share_weights_across_roles": True,
        "roles": ["generator", "world_keyed", "world_neutral", "key_rarity", "paraphraser", "optional_judge"],
    },
    "arms": ["base", "budget_control", "passive_prefix", "llm_global", "llm_local"],
    "data": {
        "domains": ["writingprompts", "dolly", "c4_realnewslike"],
        "domain_weights": [0.3333333333, 0.3333333333, 0.3333333334],
        "groups_per_domain": {"smoke": 4, "scale_fit": 30, "dev": 50, "null_calibration": 700, "null_test": 700,
                              "positive_test": 200, "donor_pool": 200, "ablation": 20},
        "writingprompts_candidate_cap": 6000,
        "reference_cap_tokens": 512,
        "allow_shortfall": False,
    },
    "keys": {
        "families": ["common_single", "rare_single", "common_multi", "rare_multi", "random_digits"],
        "development_per_family": 4,
        "confirmation_per_selected_family": 16,
        "neutral_prefix": NEUTRAL_PREFIX,
    },
    "scoring": {
        "primary": "llm_global", "scorer_id": SCORER_ID, "max_input_tokens": 4096,
        "initial_mode": "full_teacher_forcing", "reductions_dtype": "float32",
        "system_template": SCORING_SYSTEM_TEMPLATE, "user_message": SCORING_USER,
    },
    "generation": {
        "unit": "tokens", "unit_tokens": 16, "sentence_cap_tokens": 64, "candidates": 8, "beta": 2.0,
        "standardized_gain_clip": 5.0, "active_fraction": 1.0, "temperature": 0.8, "top_p": 0.95, "top_k": 0,
        "min_new_tokens": 64, "max_new_tokens": 512, "dev_seeds": [0], "confirmation_seeds": [0, 1],
    },
    "arm_overrides": {"budget_control": {"matched_scoring": False}},
    "detection": {"alpha": 0.01, "min_visible_tokens": 32, "null_cohorts": ["generator", "human_web"],
                  "combine_tail_ranks": "maximum", "minimum_calibration_per_cohort": 1000},
    "evaluation": {"bootstrap_replicates": 2000, "length_views": [64, 128, 256, 512], "max_ablation_configs": 12,
                   "ablation_prompts_per_config": 60, "max_confirmation_configs": 2},
    "attacks": {"strong_paraphrase_seed0_prompts": 600, "wider_suite_seed0_prompts": 150, "crop_fractions": [0.5, 0.75],
                "dilution_added_length_fractions": [0.25, 0.5, 1.0], "paraphrase_max_new_tokens": 640,
                "paraphrase_temperature": 0.7, "paraphrase_top_p": 0.9},
    "runtime": {"resume": True, "automatic_hyperparameter_search": False, "require_budget_manifest_for_large_runs": True,
                "paid_api_calls": 0, "device": "auto", "kernels": "torch", "attn_implementation": None, "allow_tf32": False,
                "fork_prefix_cache": False, "scoring_token_budget": 24576, "head_chunk_positions": 256},
    "paths": {"root": ".", "data_raw": "data/raw", "data_prepared": "data/prepared", "assets": "assets", "runs": "runs"},
    "stages": {
        # per-stage overrides of generation settings, arms, seeds, key set, scales artifact
        "smoke": {"partition": "smoke", "arms": None, "seeds": [0], "keys": "development", "scales": "assets/scales_smoke.json",
                  "generation": {"max_new_tokens": 64, "min_new_tokens": 16, "candidates": 4}, "toy_calibration": True,
                  "matched_scoring_measurement": True},
        "dev": {"partition": "dev", "arms": None, "seeds": [0], "keys": "development", "scales": "assets/scales.json"},
        "ablation": {"partition": "dev", "ablation_subset_only": True, "arms": ["llm_global"], "seeds": [0],
                     "keys": "development", "scales": "assets/scales.json"},
        "null_calibration": {"partition": "null_calibration", "arms": ["base"], "seeds": [0], "keys": "confirmation"},
        "null_test": {"partition": "null_test", "arms": ["base"], "seeds": [0], "keys": "confirmation"},
        "positive_test": {"partition": "positive_test", "arms": None, "seeds": [0, 1], "keys": "confirmation",
                          "scales": "assets/scales.json"},
    },
    "budgets": {},   # per-stage hard limits, written by --dry-run into the budget manifest and checked at run time
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


@dataclass
class Config:
    raw: dict
    path: str | None = None

    # ---- accessors ------------------------------------------------------ #
    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    @property
    def root(self) -> Path:
        base = Path(self.path).parent.parent if self.path else Path(".")
        r = Path(self.raw["paths"]["root"])
        return (base / r).resolve() if not r.is_absolute() else r

    def path_of(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else (self.root / p)

    def resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def model_revision(self) -> str:
        return str(self.raw["models"]["shared"]["revision"])

    def runtime_config(self) -> RuntimeConfig:
        m = self.raw["models"]["shared"]
        r = self.raw["runtime"]
        return RuntimeConfig(model_id=m["id"], revision=m["revision"], dtype=m.get("dtype", "bfloat16"),
                             device=r.get("device", "auto"), attn_implementation=r.get("attn_implementation"),
                             max_input_tokens=int(self.raw["scoring"]["max_input_tokens"]),
                             head_chunk_positions=int(r.get("head_chunk_positions", 256)),
                             scoring_token_budget=int(r.get("scoring_token_budget", 24576)),
                             fork_prefix_cache=bool(r.get("fork_prefix_cache", False)), kernels=r.get("kernels", "torch"),
                             allow_tf32=bool(r.get("allow_tf32", False)))

    def generation_config(self, stage: str | None = None, extra: dict | None = None) -> GenerationConfig:
        g = dict(self.raw["generation"])
        if stage and stage in self.raw["stages"]:
            g.update(self.raw["stages"][stage].get("generation", {}) or {})
        if extra:
            g.update(extra)
        gc = GenerationConfig(
            unit=g["unit"], unit_tokens=int(g["unit_tokens"]), sentence_cap_tokens=int(g.get("sentence_cap_tokens", 64)),
            candidates=int(g["candidates"]), beta=float(g["beta"]), clip=float(g.get("standardized_gain_clip", g.get("clip", 5.0))),
            active_fraction=float(g["active_fraction"]), temperature=float(g["temperature"]), top_p=float(g["top_p"]),
            top_k=int(g.get("top_k", 0)), min_new_tokens=int(g["min_new_tokens"]), max_new_tokens=int(g["max_new_tokens"]),
        )
        # every other GenerationConfig field (objective, selection, center_gains, matched_scoring, inactive_n1, ...)
        # passes through from the stage settings / --override; unknown keys are an error, never silently dropped
        known = set(GenerationConfig.__dataclass_fields__)
        aliases = {"standardized_gain_clip": "clip", "dev_seeds": None, "confirmation_seeds": None}
        for k, v in g.items():
            if k in aliases:
                continue
            if k not in known:
                raise ValueError(f"unknown generation setting {k!r}; known: {sorted(known)}")
            if k not in ("unit", "unit_tokens", "sentence_cap_tokens", "candidates", "beta", "clip", "active_fraction",
                         "temperature", "top_p", "top_k", "min_new_tokens", "max_new_tokens"):
                setattr(gc, k, v)
        return gc

    def arm_overrides(self, arm: str) -> dict:
        return dict((self.raw.get("arm_overrides") or {}).get(arm, {}) or {})

    def stage(self, name: str) -> dict:
        if name not in self.raw["stages"]:
            raise KeyError(f"stage {name!r} not configured; known: {sorted(self.raw['stages'])}")
        return self.raw["stages"][name]

    def stage_arms(self, name: str) -> list[str]:
        arms = self.stage(name).get("arms") or self.raw["arms"]
        for a in arms:
            if a not in ARM_PRESETS:
                raise ValueError(f"unknown arm {a}")
        return list(arms)

    def content_hash(self) -> str:
        return content_hash(self.raw)

    def validate(self, require_revision: bool = False) -> None:
        # named arm variants (frozen protocols): alias -> base arm; registered as presets so stages/arm_config resolve them
        for name, base in (self.raw.get("arm_aliases") or {}).items():
            if base not in ARM_PRESETS:
                raise ValueError(f"arm alias {name!r} refers to unknown arm {base!r}")
            ARM_PRESETS.setdefault(name, dict(ARM_PRESETS[base]))
        m = self.raw["models"]["shared"]
        if m["id"] != ALLOWED_MODEL_ID:
            raise ValueError(f"models.shared.id must be {ALLOWED_MODEL_ID}")
        if require_revision and (not m.get("revision") or m["revision"] == "RESOLVE_AND_FREEZE"):
            raise ValueError("models.shared.revision must be resolved to an immutable sha before model-backed runs")
        if m.get("enable_thinking", False):
            raise ValueError("enable_thinking must be false")
        if int(self.raw["runtime"].get("paid_api_calls", 0)) != 0:
            raise ValueError("paid_api_calls must be 0")
        if self.raw["runtime"].get("automatic_hyperparameter_search"):
            raise ValueError("automatic_hyperparameter_search must be false")
        for a in self.raw["arms"]:
            if a not in ARM_PRESETS:
                raise ValueError(f"unknown arm {a}")
        g = self.raw["generation"]
        if g["unit"] not in ("tokens", "sentence"):
            raise ValueError("generation.unit must be tokens|sentence")
        if not (0 < float(g["active_fraction"]) <= 1):
            raise ValueError("active_fraction must be in (0, 1]")
        if int(g["candidates"]) < 1:
            raise ValueError("candidates must be >= 1")
        d = self.raw["detection"]
        if not (0 < float(d["alpha"]) < 1):
            raise ValueError("alpha must be in (0,1)")
        for role in self.raw["models"]["roles"]:
            if role not in ("generator", "world_keyed", "world_neutral", "key_rarity", "paraphraser", "optional_judge"):
                raise ValueError(f"unknown role {role}")


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    raw = copy.deepcopy(DEFAULTS)
    with Path(path).open("r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    _deep_update(raw, user)
    if overrides:
        _deep_update(raw, overrides)
    cfg = Config(raw=raw, path=str(Path(path).resolve()))
    cfg.validate()
    return cfg


def config_from_protocol(protocol: dict) -> Config:
    cfg = Config(raw=copy.deepcopy(protocol["config"]), path=protocol.get("config_path"))
    cfg.validate(require_revision=True)
    return cfg
