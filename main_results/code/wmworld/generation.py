"""Watermark embedding by candidate selection (spec Section 5) and the five arms
(spec Section 8).

At every decision the generator holds an immutable accepted prefix ``y`` (its
own token stream after the chat prompt), samples ``N`` candidate units from it,
scores each *cumulative* text ``decode(y || s_j)`` with the world scorer, and
selects with ``softmax(beta * clip(Delta_j / sigma, -clip, clip))``.  Segments
exist only here and in the private trajectory log; the detector scores the
finished text once.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import replace, asdict, dataclass, field

import numpy as np

from .backend import Backend, Candidate, SamplingParams
from .records import CostRecord, TrajectoryRecord
from .scales import ScaleTable, length_bin, unit_setting_name
from .scoring import KeySpec, LikelihoodScorer, ScoreResult
from .textnorm import normalize_text, seed_from, text_hash

GENERATOR_SYSTEM = "You are a helpful assistant."
PASSIVE_PREFIX_SYSTEM_TEMPLATE = "You are a helpful assistant.\nBackground: {prefix}"

# Deterministic sentence boundary: terminal punctuation (optionally followed by
# closing quotes/brackets) followed by whitespace or end of text; or a newline.
SENTENCE_BOUNDARY_RE = re.compile(r'[.!?]+["\'\)\]]*(?=\s|$)|\n')


@dataclass
class GenerationConfig:
    unit: str = "tokens"               # tokens | sentence
    unit_tokens: int = 16
    sentence_cap_tokens: int = 64
    candidates: int = 8
    beta: float = 2.0
    clip: float = 5.0
    active_fraction: float = 1.0
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    min_new_tokens: int = 64
    max_new_tokens: int = 512
    objective: str = "global"          # global | local | specific | none   (specific = amendment: contrast vs steering decoys)
    selection: str = "softmax"         # softmax | greedy | uniform | rank
    center_gains: bool = False         # amendment: subtract the pool mean before /sigma and clipping (identical unless the clip binds)
    steer_alpha: float = 0.0           # extension: activation steering strength (0 = off); proposals only, never scoring
    steer_layers: str = "all"          # "all" or comma-separated decoder layer indices
    poe_beta: float = 0.0              # extension: token-level product-of-experts steering strength (0 = off)
    poe_gate: str = "entropy"          # entropy | kl | lp | none  (lp = budget on expected base log-likelihood loss per token)
    poe_entropy_ref: float = 2.0       # nats: full strength at/above this entropy
    poe_entropy_min: float = 0.5       # nats: no steering below this entropy
    poe_clip: float = 5.0              # per-token contrast clip (nats)
    poe_kl_budget: float = 0.1         # gate "kl": KL budget; gate "lp": expected base log-likelihood loss budget (nats/token)
    poe_steer_eos: bool = True         # False: p'(EOS) is pinned to the generator's p(EOS) (length is not a channel; v2 exact pin)
    poe_block_key_tokens: bool = False # leak guard: zero the contrast of the key's own completion words (and variants)
    poe_block_category: bool = False   # topic guard: also zero the contrast of every catalog word of the key's categories
    poe_adapt: str = "none"            # none | tstat (linear ramp) | bang (full beta until the running t reaches poe_t_target, then floor)
    poe_t_target: float = 5.0          # running within-text t-statistic at which steering stops (adapt = tstat)
    poe_contrast_max: float = 0.0      # >0: semantic guard -- zero push on tokens whose raw contrast exceeds this (the key's
    poe_ref_window: int = 0            # cross-model only: the reference sees the last W..W+S-1 tokens of the text (0 = whole text)
    poe_ref_stride: int = 0            # block size S of the bounded view (0 = W/2); the detector must use the same pair
                                       # own semantic neighbourhood); the mark is then carried by many subtle preferences
    poe_warmup_tokens: int = 0         # >0: ramp the push linearly from 0 over the first N tokens (protects the opening)
    poe_t_window: int = 0              # >0: measure the mark over the LAST N tokens, not the whole text, so the
                                       # controller regulates the marking RATE and keeps correcting to the end
    poe_t_target_scale: str = "none"   # none | budget: scale the target by sqrt(max_new_tokens/512), so the margin over
                                       # the clean threshold does not shrink as the text grows (T itself grows like sqrt(n))

    poe_beta_floor: float = 0.0        # fraction of poe_beta kept once the target is reached (adapt = tstat)
    poe_block_repeat_n: int = 0        # repetition guard: no push for a token that would complete an n-gram already in the text (0 = off)
    poe_reference: str = ""            # cross-model steering: non-empty -> the world is the configured SECOND model
    poe_span_tokens: int = 0           # >0: the controller's running statistic resets every N tokens, so EVERY span is
                                       # pushed until it alone is detectable (the reference contexts never restart)
    poe_key_rotation: int = 0          # >1: spans rotate through this many keys (assigned key + rotation partners),
                                       # so different parts of one text carry different keys
    poe_contrast: str = "neutral"      # neutral: log p_key - log p_neutral | decoys: log p_key - mean_d log p_decoy_d (key-specific push)
    poe_n_decoys: int = 3              # decoy contexts used when poe_contrast = decoys (from key.meta['steering_decoys'])
    matched_scoring: bool = False      # uniform arms may still run (and discard) scoring calls
    inactive_n1: bool = False          # cost ablation: N=1 in inactive slots
    prefix_in_generator: bool = False  # passive_prefix arm
    generator_system: str = GENERATOR_SYSTEM
    passive_system_template: str = PASSIVE_PREFIX_SYSTEM_TEMPLATE

    @property
    def unit_setting(self) -> str:
        return unit_setting_name(self.unit, self.unit_tokens, self.sentence_cap_tokens)

    def sampling(self) -> SamplingParams:
        return SamplingParams(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)

    def to_dict(self) -> dict:
        return asdict(self)


ARM_PRESETS: dict[str, dict] = {
    "base": dict(candidates=1, objective="none", selection="uniform"),
    "budget_control": dict(objective="none", selection="uniform"),   # inherits the stage's N (8 in dev; spec: N=8, uniform)
    "passive_prefix": dict(candidates=1, objective="none", selection="uniform", prefix_in_generator=True),
    "llm_global": dict(objective="global", selection="softmax"),
    "llm_local": dict(objective="local", selection="softmax"),
    # labelled extensions (not part of the spec's five arms)
    "activation_steer": dict(candidates=1, objective="none", selection="uniform", steer_alpha=1.0),
    "steer_select": dict(objective="global", selection="softmax", steer_alpha=1.0),
    "poe_steer": dict(candidates=1, objective="none", selection="uniform", poe_beta=1.0),
}


def arm_config(arm: str, base: GenerationConfig, overrides: dict | None = None) -> GenerationConfig:
    if arm not in ARM_PRESETS:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARM_PRESETS)}")
    d = base.to_dict()
    d.update(ARM_PRESETS[arm])
    if overrides:
        d.update(overrides)
    return GenerationConfig(**d)


@dataclass
class GenerationResult:
    text: str
    text_hash: str
    accepted_tokens: list[int]
    visible_tokens: int
    stop_reason: str
    ends_mid_sentence: bool
    n_decisions: int
    n_active_decisions: int
    n_inactive_no_signal: int
    cost: CostRecord
    trajectory: list[TrajectoryRecord] = field(default_factory=list)
    final_score: float | None = None
    status: str = "ok"
    error: str | None = None


def sentence_truncate(backend: Backend, token_ids: list[int]) -> tuple[list[int], int]:
    """Commit only through the first valid sentence boundary (or all tokens if none).

    Returns (kept_ids, discarded_tail_tokens).  The boundary is located on the
    decoded text; tokens are kept while the decoded prefix does not yet cover
    the boundary character.
    """
    if not token_ids:
        return token_ids, 0
    text = backend.decode(token_ids)
    m = SENTENCE_BOUNDARY_RE.search(text)
    if m is None:
        return token_ids, 0
    end_char = m.end()
    if end_char >= len(text):
        return token_ids, 0
    for k in range(1, len(token_ids) + 1):
        if len(backend.decode(token_ids[:k])) >= end_char:
            return token_ids[:k], len(token_ids) - k
    return token_ids, 0


def ends_mid_sentence(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and not re.search(r'[.!?]["\'\)\]]*$', t)


def softmax_selection(gains: list[float], sigma: float, beta: float, clip: float, center: bool = False) -> np.ndarray:
    g = np.asarray(gains, dtype=np.float64)
    if center:
        g = g - g.mean()                # softmax is shift-invariant: identical to the spec rule unless the clip binds
    z = g / float(sigma)
    z = np.clip(z, -clip, clip)
    w = beta * z
    w = w - w.max()
    p = np.exp(w)
    return p / p.sum()


def rank_selection(gains: list[float], beta: float) -> np.ndarray:
    """Rank-based alternative (ablation): P(j) ∝ exp(beta * r_j), r_j = +1 for the best candidate ... -1 for the worst.

    Invariant to the gain scale (no sigma, floor or clip); ties keep candidate order (stable sort).
    """
    g = np.asarray(gains, dtype=np.float64)
    n = g.size
    if n == 1:
        return np.ones(1)
    order = np.argsort(-g, kind="stable")          # best first
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    r = 1.0 - 2.0 * ranks / (n - 1)
    w = beta * r
    w = w - w.max()
    p = np.exp(w)
    return p / p.sum()


def _word_variants(w: str) -> list[str]:
    w = w.strip()
    if not w:
        return []
    forms = {w, w.lower(), w.capitalize()}
    forms |= {f + "s" for f in list(forms)}
    return sorted(set(list(forms) + [" " + f for f in forms]))


def poe_block_ids(backend, key, block_key_tokens: bool, block_category: bool) -> list[int]:
    """Token ids whose contrast the PoE sampler zeroes (leak/topic guards).

    Key guard: the key's completion words (catalog keys), the seed words (random_words) or, failing that, the prefix's
    content words; category guard: every catalog completion of the key's template categories.  For each word and
    surface variant (case, plural, leading space) the FIRST token id is blocked, which is enough to stop the word from
    starting; blocking a sub-token of a longer word costs a little signal, never correctness.
    """
    if not (block_key_tokens or block_category):
        return []
    words: set[str] = set()
    if block_key_tokens:
        comps = list((key.meta or {}).get("completions") or [])
        if not comps:
            comps = [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", key.prefix) if len(w) > 3]
        words |= set(comps)
    if block_category:
        from .key_catalog import COMPLETIONS
        for tid in (key.meta or {}).get("template_ids") or []:
            words |= set(COMPLETIONS.get(tid, []))
    ids: set[int] = set()
    for w in words:
        for v in _word_variants(w):
            toks = backend.encode_text(v)
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)


class CandidateSelectionGenerator:
    def __init__(self, backend: Backend, scorer: LikelihoodScorer | None, scales: ScaleTable | None,
                 reference=None, ref_token_map=None):
        self.backend = backend
        self.scorer = scorer
        self.scales = scales
        self.reference = reference            # cross-model steering: a ReferenceRuntime (or fake) or None
        self.ref_token_map = ref_token_map    # generator token id -> first reference token id

    # -------------------------------------------------------------------- #
    def active_schedule(self, key_id: str | None, run_seed: int, cfg: GenerationConfig, n_decisions: int) -> list[bool]:
        """Deterministic pseudorandom schedule from (key id, run seed); detection never sees it."""
        if cfg.active_fraction >= 1.0:
            return [True] * n_decisions
        rng = np.random.default_rng(seed_from("active_schedule", key_id, run_seed))
        return [bool(x) for x in (rng.random(n_decisions) < cfg.active_fraction)]

    # -------------------------------------------------------------------- #
    def generate(self, prompt: str, key: KeySpec | None, cfg: GenerationConfig, run_seed: int,
                 output_id: str) -> GenerationResult:
        t0 = time.time()
        cost0 = self.backend.cost_snapshot()
        steering = cfg.objective in ("global", "local", "specific")
        if steering and key is None:
            raise ValueError("steering arms require a key")
        if steering and self.scorer is None:
            raise ValueError("steering arms require a scorer")
        # activation steering (extension): per-layer residual offsets from the keyed vs neutral scoring context,
        # applied by the backend to proposal forward passes only
        if cfg.steer_alpha > 0:
            if key is None:
                raise ValueError("activation steering requires a key")
            if not hasattr(self.backend, "set_steering"):
                raise ValueError("backend does not support activation steering")
            cache = getattr(self, "_steer_cache", None)
            if cache is None:
                cache = self._steer_cache = {}
            if key.key_id not in cache:
                cache[key.key_id] = self.backend.steering_vectors(key.system_text(neutral=False), key.system_text(neutral=True), key.user_message)
            layers = None if cfg.steer_layers == "all" else [int(x) for x in str(cfg.steer_layers).split(",") if x.strip()]
            self.backend.set_steering(cache[key.key_id]["vectors"], cfg.steer_alpha, layers)
        else:
            if hasattr(self.backend, "set_steering"):
                self.backend.set_steering(None)
        if cfg.prefix_in_generator:
            if key is None:
                raise ValueError("passive_prefix requires a key")
            system = cfg.passive_system_template.format(prefix=key.prefix)
        else:
            system = cfg.generator_system
        ctx = self.backend.chat_prefix_ids(system, prompt)
        if hasattr(self.backend, "forget_prefix"):
            self.backend.forget_prefix()

        # ---- token-level product-of-experts steering (extension): one call generates the whole response ---------- #
        if cfg.poe_beta > 0:
            t_target = cfg.poe_t_target
            if cfg.poe_t_target_scale == "budget" and cfg.max_new_tokens > 0:
                t_target = cfg.poe_t_target * (cfg.max_new_tokens / 512.0) ** 0.5
            if key is None or self.scorer is None:
                raise ValueError("poe steering requires a key and a scorer")
            if not hasattr(self.backend, "sample_poe"):
                raise ValueError("backend does not support poe steering")
            block = poe_block_ids(self.backend, key, cfg.poe_block_key_tokens, cfg.poe_block_category)
            cross = bool(cfg.poe_reference)
            if cross and (self.reference is None or self.ref_token_map is None):
                raise ValueError("poe_reference is set but no reference model / token map was provided")
            decoy_ctx: list[list[int]] = []
            if cfg.poe_contrast == "decoys":
                dps = [d for d in (key.meta or {}).get("steering_decoys") or [] if d != key.prefix][:max(1, cfg.poe_n_decoys)]
                if not dps:
                    raise ValueError(f"poe_contrast=decoys needs steering decoys on key {key.key_id} (stages.keyspec_with_decoys)")
                decoy_ctx = [self.scorer.context_ids(replace(key, key_id=f"{key.key_id}.d{i}", prefix=dp, meta={}), neutral=False)
                             for i, dp in enumerate(dps)]
            elif cfg.poe_contrast != "neutral":
                raise ValueError(f"unknown poe_contrast {cfg.poe_contrast!r}")
            win = {"window": int(cfg.poe_ref_window), "stride": int(cfg.poe_ref_stride)} if cfg.poe_ref_window else {}
            try:
                if cross:
                    out = self.backend.sample_poe_cross(
                        ctx,
                        self.reference.stepper(self.reference.chat_prefix_ids(key.system_text(False), key.user_message), **win),
                        self.reference.stepper(self.reference.chat_prefix_ids(key.system_text(True), key.user_message), **win),
                        self.ref_token_map, cfg.max_new_tokens, cfg.sampling(), seed_from("poe", run_seed, output_id),
                        max(0, cfg.min_new_tokens), cfg.poe_beta, cfg.poe_clip, cfg.poe_gate, cfg.poe_entropy_ref,
                        cfg.poe_entropy_min, cfg.poe_steer_eos, block, cfg.poe_adapt, t_target,
                        cfg.poe_beta_floor, int(cfg.poe_block_repeat_n), t_window=int(cfg.poe_t_window), warmup=int(cfg.poe_warmup_tokens), d_max=float(cfg.poe_contrast_max),
                        span_tokens=int(cfg.poe_span_tokens),
                        ref_key_factories=([(lambda pfx=rp: self.reference.stepper(self.reference.chat_prefix_ids(
                            replace(key, prefix=pfx, meta={}).system_text(False), key.user_message), **win))
                            for rp in ([key.prefix] + [d for d in (key.meta or {}).get("rotation_keys") or []
                                                       if d != key.prefix])[:cfg.poe_key_rotation]]
                            if cfg.poe_key_rotation > 1 else None),
                        ref_decoys=[self.reference.stepper(self.reference.chat_prefix_ids(
                            replace(key, prefix=dp, meta={}).system_text(False), key.user_message), **win)
                            for dp in ([d for d in (key.meta or {}).get("steering_decoys") or [] if d != key.prefix]
                                       [:max(1, cfg.poe_n_decoys)] if cfg.poe_contrast == "decoys" else [])])
                else:
                    out = self.backend.sample_poe(ctx, self.scorer.context_ids(key, neutral=False), self.scorer.context_ids(key, neutral=True),
                                                  cfg.max_new_tokens, cfg.sampling(), seed_from("poe", run_seed, output_id),
                                                  max(0, cfg.min_new_tokens), cfg.poe_beta, cfg.poe_clip, cfg.poe_gate,
                                                  cfg.poe_entropy_ref, cfg.poe_entropy_min, cfg.poe_kl_budget, cfg.poe_steer_eos,
                                                  block_ids=block, adapt=cfg.poe_adapt, t_target=t_target, beta_floor=cfg.poe_beta_floor,
                                                  repeat_n=int(cfg.poe_block_repeat_n), decoy_ctx_ids=decoy_ctx or None,
                                                  t_window=int(cfg.poe_t_window), warmup=int(cfg.poe_warmup_tokens), d_max=float(cfg.poe_contrast_max), span_tokens=int(cfg.poe_span_tokens))
            except Exception as e:
                return GenerationResult(text="", text_hash=text_hash(""), accepted_tokens=[], visible_tokens=0, stop_reason="error",
                                        ends_mid_sentence=False, n_decisions=0, n_active_decisions=0, n_inactive_no_signal=0,
                                        cost=self._cost_delta(cost0, t0), status="failed", error=repr(e))
            y = list(out["token_ids"])
            text = normalize_text(self.backend.decode(y))
            final = self.scorer.score(text, key)
            n_steered = sum(1 for b in out["beta_t"] if b > 0)
            traj = [TrajectoryRecord(
                id=f"{output_id}.poe", output_id=output_id, decision=0, accepted_prefix_hash=text_hash(""), accepted_len=0,
                unit="token", active=True, scale_status="poe", sigma=None, length_bin=None,
                candidate_token_ids=[[int(t) for t in y]], candidate_texts=[text], candidate_scores=[final.score],
                prefix_score=0.0, gains=[final.score], incremental_contrast=[final.contrast_sum], probabilities=[1.0], chosen=0,
                chosen_ended=bool(out["ended"]),
                cost={"per_token": {"entropy": out["entropy"], "beta_t": out["beta_t"], "contrast": out["contrast"], "logp_g": out["logp_g"],
                                    "kl": out.get("kl", []), "lp_loss": out.get("lp_loss", []), "contrast_raw": out.get("contrast_raw", []),
                                    "t_run": out.get("t_run", []), "contrast_steer_raw": out.get("contrast_steer_raw", [])},
                      "poe_contrast": cfg.poe_contrast, "n_decoys": len(decoy_ctx),
                      "n_tokens": len(y), "n_steered_tokens": n_steered, "n_blocked_ids": len(block),
                      "t_target_effective": t_target, "span_tokens": cfg.poe_span_tokens, "span_starts": out.get("span_starts"),
                      "key_order": out.get("key_order"), "n_rotation_keys": out.get("n_rotation_keys"), "cross_reference": cfg.poe_reference or None, "n_unmapped_candidates": out.get("n_unmapped_candidates"),
                      "ref_forward_calls": out.get("ref_forward_calls"), "n_ref_decoys": out.get("n_ref_decoys"),
                      "n_repeat_blocked": out.get("n_repeat_blocked", 0),
                      "final_t_run": (out["t_run"][-1] if out.get("t_run") else None), "mean_beta_t": (sum(out["beta_t"]) / len(out["beta_t"])) if out["beta_t"] else 0.0,
                      "sum_chosen_contrast": sum(out["contrast"]), "mean_kl": (sum(out.get("kl", [])) / len(out["kl"])) if out.get("kl") else None,
                      "mean_lp_loss": (sum(out.get("lp_loss", [])) / len(out["lp_loss"])) if out.get("lp_loss") else None,
                      "mean_logp_g": (sum(out["logp_g"]) / len(out["logp_g"])) if out["logp_g"] else None},
            )]
            cost = self._cost_delta(cost0, t0)
            return GenerationResult(text=text, text_hash=text_hash(text), accepted_tokens=y, visible_tokens=len(self.backend.encode_text(text)),
                                    stop_reason="eos" if out["ended"] else "max_new_tokens", ends_mid_sentence=ends_mid_sentence(text),
                                    n_decisions=len(y), n_active_decisions=n_steered, n_inactive_no_signal=0, cost=cost, trajectory=traj,
                                    final_score=final.score)

        max_decisions = math.ceil(cfg.max_new_tokens / max(1, cfg.unit_tokens if cfg.unit == "tokens" else 1)) + 1
        schedule = self.active_schedule(key.key_id if key else None, run_seed, cfg, max_decisions * 2)
        cand_seed = seed_from("candidates", run_seed, output_id)
        sel_rng = np.random.default_rng(seed_from("selection", run_seed, output_id))
        sampling = cfg.sampling()

        y: list[int] = []
        trajectory: list[TrajectoryRecord] = []
        prev_score: float = 0.0            # F_k(empty) = 0
        prev_contrast_sum: float = 0.0
        stop_reason = "max_new_tokens"
        n_active = 0
        n_no_signal = 0
        decision = 0
        discarded_total = 0
        try:
            while True:
                remaining = cfg.max_new_tokens - len(y)
                if remaining <= 0:
                    stop_reason = "max_new_tokens"
                    break
                unit_cap = cfg.unit_tokens if cfg.unit == "tokens" else cfg.sentence_cap_tokens
                unit_max = min(unit_cap, remaining)
                score_before = prev_score
                eos_allowed_from = max(0, cfg.min_new_tokens - len(y))
                slot_active = steering and schedule[decision] if decision < len(schedule) else steering
                n = cfg.candidates
                if not slot_active and cfg.inactive_n1:
                    n = 1
                cands = self.backend.sample_units(ctx + y, n, unit_max, sampling, cand_seed + decision, eos_allowed_from)
                if cfg.unit == "sentence":
                    new_cands = []
                    for c in cands:
                        kept, disc = sentence_truncate(self.backend, c.token_ids)
                        discarded_total += disc
                        new_cands.append(Candidate(token_ids=kept, ended=(c.ended and disc == 0), n_generated=c.n_generated))
                    cands = new_cands
                texts = [self.backend.decode(y + c.token_ids) for c in cands]

                scores: list[ScoreResult | None] = [None] * n
                gains: list[float | None] = [None] * n
                incr: list[float | None] = [None] * n
                need_scores = (slot_active or cfg.matched_scoring) and self.scorer is not None and key is not None
                if need_scores:
                    if cfg.objective == "local":
                        res = self.scorer.score_texts([self.backend.decode(c.token_ids) for c in cands], key)
                        scores = list(res)
                        gains = [r.score if r.status in ("ok", "empty") else None for r in res]
                        cum = self.scorer.score_texts(texts, key) if cfg.matched_scoring else None
                        incr = [(r.contrast_sum - prev_contrast_sum) if (cum is not None and r.contrast_sum is not None) else None
                                for r in (cum or [ScoreResult(None, "skipped", 0)] * n)]
                    else:
                        # cumulative objectives: "global" (contrast vs neutral) or "specific" (contrast vs steering decoys)
                        res = (self.scorer.score_specific(texts, key) if cfg.objective == "specific"
                               else self.scorer.score_texts(texts, key))
                        scores = list(res)
                        gains = [(r.score - prev_score) if r.status in ("ok", "empty") else None for r in res]
                        incr = [(r.contrast_sum - prev_contrast_sum) if r.contrast_sum is not None else None for r in res]

                # ---- selection ------------------------------------------------ #
                sigma = None
                bin_name = length_bin(len(y))
                if not steering or cfg.selection == "uniform":
                    probs = np.full(n, 1.0 / n)
                    scale_status = "uniform_arm"
                elif not slot_active:
                    probs = np.full(n, 1.0 / n)
                    scale_status = "inactive_schedule"
                else:
                    n_active += 1
                    entry = self.scales.lookup(key.family, cfg.unit_setting, cfg.objective, len(y)) if self.scales else None
                    usable = entry is not None and entry.status == "ok" and entry.sigma is not None and entry.sigma > 0
                    valid_gains = all(g is not None and math.isfinite(g) for g in gains)
                    if cfg.selection == "rank" and valid_gains:
                        # rank-based ablation: needs no scale table
                        sigma = None
                        scale_status = "rank"
                        probs = rank_selection([float(g) for g in gains], cfg.beta)
                    elif usable and valid_gains:
                        sigma = float(entry.sigma)
                        scale_status = "ok"
                        if cfg.selection == "greedy":
                            probs = np.zeros(n)
                            probs[int(np.argmax(gains))] = 1.0
                        else:
                            probs = softmax_selection([float(g) for g in gains], sigma, cfg.beta, cfg.clip, center=cfg.center_gains)
                    else:
                        probs = np.full(n, 1.0 / n)
                        if not usable:
                            scale_status = (entry.status if entry is not None and entry.status in
                                            ("inactive_no_signal", "insufficient_pools", "missing") else "inactive_no_signal")
                        else:
                            scale_status = "inactive_invalid_gain"
                        n_no_signal += 1
                chosen = int(sel_rng.choice(n, p=probs))
                c = cands[chosen]
                y = y + c.token_ids
                if scores[chosen] is not None and scores[chosen].status in ("ok", "empty") and cfg.objective in ("global", "specific"):
                    prev_score = float(scores[chosen].score)
                    prev_contrast_sum = float(scores[chosen].contrast_sum)
                elif need_scores and cfg.objective == "local" and cfg.matched_scoring and cum is not None and cum[chosen].contrast_sum is not None:
                    prev_score = float(cum[chosen].score)
                    prev_contrast_sum = float(cum[chosen].contrast_sum)

                trajectory.append(TrajectoryRecord(
                    id=f"{output_id}.d{decision}", output_id=output_id, decision=decision,
                    accepted_prefix_hash=text_hash(self.backend.decode(y[:len(y) - len(c.token_ids)])),
                    accepted_len=len(y) - len(c.token_ids), unit=cfg.unit_setting, active=bool(slot_active),
                    scale_status=scale_status, sigma=sigma, length_bin=bin_name,
                    candidate_token_ids=[list(map(int, cc.token_ids)) for cc in cands],
                    candidate_texts=[self.backend.decode(cc.token_ids) for cc in cands],
                    candidate_scores=[(s.score if s is not None else None) for s in scores],
                    prefix_score=(score_before if (need_scores and cfg.objective in ("global", "specific")) else None),
                    gains=[(float(g) if g is not None else None) for g in gains],
                    incremental_contrast=[(float(x) if x is not None else None) for x in incr],
                    probabilities=[float(p) for p in probs], chosen=chosen, chosen_ended=bool(c.ended),
                    cost={"n_candidates": n, "unit_max": unit_max},
                ))
                decision += 1
                if c.ended:
                    stop_reason = "eos"
                    break
                if hasattr(self.backend, "remember_prefix"):
                    self.backend.remember_prefix(ctx + y)
        except Exception as e:  # keep partial output and the reason (spec Section 12)
            text = normalize_text(self.backend.decode(y))
            cost = self._cost_delta(cost0, t0)
            return GenerationResult(text=text, text_hash=text_hash(text), accepted_tokens=y,
                                    visible_tokens=len(self.backend.encode_text(text)), stop_reason="error",
                                    ends_mid_sentence=ends_mid_sentence(text), n_decisions=decision,
                                    n_active_decisions=n_active, n_inactive_no_signal=n_no_signal, cost=cost,
                                    trajectory=trajectory, status="failed", error=repr(e))
        finally:
            if hasattr(self.backend, "forget_prefix"):
                self.backend.forget_prefix()
            if hasattr(self.backend, "set_steering"):
                self.backend.set_steering(None)

        text = normalize_text(self.backend.decode(y))
        cost = self._cost_delta(cost0, t0)
        cost.discarded_tail_tokens = discarded_total
        return GenerationResult(text=text, text_hash=text_hash(text), accepted_tokens=y,
                                visible_tokens=len(self.backend.encode_text(text)), stop_reason=stop_reason,
                                ends_mid_sentence=ends_mid_sentence(text), n_decisions=decision,
                                n_active_decisions=n_active, n_inactive_no_signal=n_no_signal, cost=cost,
                                trajectory=trajectory,
                                final_score=(prev_score if cfg.objective in ("global", "specific") and steering else None))

    def _cost_delta(self, cost0: dict, t0: float) -> CostRecord:
        c1 = self.backend.cost_snapshot()
        return CostRecord(
            proposal_tokens=c1.get("proposal_tokens", 0) - cost0.get("proposal_tokens", 0),
            proposal_forward_calls=c1.get("proposal_forward_calls", 0) - cost0.get("proposal_forward_calls", 0),
            scoring_tokens=c1.get("scoring_tokens", 0) - cost0.get("scoring_tokens", 0),
            scoring_forward_calls=c1.get("scoring_forward_calls", 0) - cost0.get("scoring_forward_calls", 0),
            scored_sequences=c1.get("scored_sequences", 0) - cost0.get("scored_sequences", 0),
            wall_seconds=time.time() - t0,
            peak_memory_bytes=int(c1.get("peak_memory_bytes", 0)),
        )
