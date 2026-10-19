"""Whole-text likelihood-contrast score ``llm_global`` (spec Section 4).

    F_k(y) = (ell_k(y) - ell_k0(y)) / n_y,   n_y > 0;   F_k("") = 0

``ell_k(y)`` is the teacher-forced sum of log-probabilities of the visible text
tokens under the fixed scoring context

    system: You are a helpful assistant.\\nBackground: {prefix}
    user:   Write a coherent response.
    assistant: {text}

rendered with the official chat template, thinking disabled.  The task prompt
is deliberately absent.  The text is tokenized alone (``add_special_tokens=False``)
and the same target token sequence is used for the keyed and the neutral
context.  The closing chat delimiter / EOS is never scored.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .backend import Backend
from .textnorm import TEXTNORM_VERSION, normalize_text

SCORER_ID = "llm_global_v1"
NEUTRAL_PREFIX = "No additional background is specified."
SCORING_SYSTEM_TEMPLATE = "You are a helpful assistant.\nBackground: {prefix}"
SCORING_USER = "Write a coherent response."


@dataclass
class KeySpec:
    """Key object: the exact secret prefix plus the frozen scorer wrapper configuration."""
    key_id: str
    prefix: str
    family: str = "unspecified"
    neutral_prefix: str = NEUTRAL_PREFIX
    scorer_id: str = SCORER_ID
    system_template: str = SCORING_SYSTEM_TEMPLATE
    user_message: str = SCORING_USER
    textnorm_version: str = TEXTNORM_VERSION
    meta: dict = field(default_factory=dict)

    def system_text(self, neutral: bool = False) -> str:
        return self.system_template.format(prefix=self.neutral_prefix if neutral else self.prefix)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KeySpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ScoreResult:
    score: float | None
    status: str                     # ok | empty | abstain_overflow | error
    n_tokens: int
    ell_key: float | None = None
    ell_neutral: float | None = None
    ctx_tokens_key: int = 0
    ctx_tokens_neutral: int = 0
    error: str | None = None

    @property
    def contrast_sum(self) -> float | None:
        if self.ell_key is None or self.ell_neutral is None:
            return None
        return self.ell_key - self.ell_neutral


class LikelihoodScorer:
    """Computes ``F_k`` from text and key alone.  Never sees prompts, labels or traces."""

    def __init__(self, backend: Backend, max_input_tokens: int | None = None, neutral_cache_size: int = 20_000):
        self.backend = backend
        self.max_input_tokens = max_input_tokens or backend.max_input_tokens
        self._ctx_cache: dict[tuple[str, str, str, bool], list[int]] = {}
        self.scorer_id = SCORER_ID
        # The neutral-context sum ell_k0(y) does not depend on the key: cache it per (neutral context, token sequence).
        # Values are exact re-uses of a previous forward pass on the identical token sequence (batch-shape independent
        # within the measured float32 tolerance), never approximations.
        self._neutral_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        self._neutral_cache_size = neutral_cache_size
        self.neutral_cache_hits = 0

    # ---- contexts ------------------------------------------------------- #
    def context_ids(self, key: KeySpec, neutral: bool) -> list[int]:
        ck = (key.system_text(neutral), key.user_message, key.scorer_id, neutral)
        ids = self._ctx_cache.get(ck)
        if ids is None:
            ids = self.backend.chat_prefix_ids(key.system_text(neutral), key.user_message)
            self._ctx_cache[ck] = ids
        return ids

    # ---- scoring -------------------------------------------------------- #
    def tokenize(self, text: str) -> list[int]:
        return self.backend.encode_text(normalize_text(text))

    def score_texts(self, texts: list[str], key: KeySpec) -> list[ScoreResult]:
        """Score many texts under one key; texts are batched per context for efficiency."""
        ctx_k = self.context_ids(key, neutral=False)
        ctx_0 = self.context_ids(key, neutral=True)
        toks = [self.tokenize(t) for t in texts]
        results: list[ScoreResult | None] = [None] * len(texts)
        valid: list[int] = []
        for i, t in enumerate(toks):
            n = len(t)
            if n == 0:
                results[i] = ScoreResult(score=0.0, status="empty", n_tokens=0, ell_key=0.0, ell_neutral=0.0,
                                         ctx_tokens_key=len(ctx_k), ctx_tokens_neutral=len(ctx_0))
            elif len(ctx_k) + n > self.max_input_tokens or len(ctx_0) + n > self.max_input_tokens:
                results[i] = ScoreResult(score=None, status="abstain_overflow", n_tokens=n,
                                         ctx_tokens_key=len(ctx_k), ctx_tokens_neutral=len(ctx_0))
            else:
                valid.append(i)
        if valid:
            targets = [toks[i] for i in valid]
            try:
                ell_k = self.backend.logprob_sums(ctx_k, targets)
                ell_0 = self._neutral_sums(ctx_0, targets)
            except Exception as e:  # keep the failure explicit
                for i in valid:
                    results[i] = ScoreResult(score=None, status="error", n_tokens=len(toks[i]), error=repr(e),
                                             ctx_tokens_key=len(ctx_k), ctx_tokens_neutral=len(ctx_0))
            else:
                for j, i in enumerate(valid):
                    n = len(toks[i])
                    lk, l0 = float(ell_k[j]), float(ell_0[j])
                    if not (math.isfinite(lk) and math.isfinite(l0)):
                        results[i] = ScoreResult(score=None, status="error", n_tokens=n, ell_key=lk, ell_neutral=l0,
                                                 error="nonfinite log-likelihood",
                                                 ctx_tokens_key=len(ctx_k), ctx_tokens_neutral=len(ctx_0))
                    else:
                        results[i] = ScoreResult(score=(lk - l0) / n, status="ok", n_tokens=n, ell_key=lk, ell_neutral=l0,
                                                 ctx_tokens_key=len(ctx_k), ctx_tokens_neutral=len(ctx_0))
        return results  # type: ignore[return-value]

    def _neutral_sums(self, ctx_0: list[int], targets: list[list[int]]) -> list[float]:
        ck = tuple(ctx_0)
        out: list[float | None] = [None] * len(targets)
        todo = []
        for i, t in enumerate(targets):
            v = self._neutral_cache.get((ck, tuple(t)))
            if v is None:
                todo.append(i)
            else:
                out[i] = v
                self.neutral_cache_hits += 1
        if todo:
            fresh = self.backend.logprob_sums(ctx_0, [targets[i] for i in todo])
            for i, v in zip(todo, fresh):
                out[i] = float(v)
                if len(self._neutral_cache) < self._neutral_cache_size:
                    self._neutral_cache[(ck, tuple(targets[i]))] = float(v)
        return out  # type: ignore[return-value]

    def score(self, text: str, key: KeySpec) -> ScoreResult:
        return self.score_texts([text], key)[0]

    # ---- key-specific contrast against steering decoys (amendment; steering objective "specific") ------- #
    def score_specific(self, texts: list[str], key: KeySpec) -> list[ScoreResult]:
        """S_k(y) = (ell_k(y) - mean_d ell_d(y)) / n_y over the key's steering decoys d (key.meta['steering_decoys']).

        The neutral context cancels in the difference, so only keyed and decoy contexts are scored.
        ``ell_neutral`` in the result holds the decoy mean so ``contrast_sum`` is the summed specific contrast.
        """
        decoys = list(key.meta.get("steering_decoys") or [])
        if not decoys:
            raise ValueError(f"key {key.key_id} has no steering decoys attached (needed for objective 'specific')")
        ctx_k = self.context_ids(key, neutral=False)
        ctx_ds = [self.backend.chat_prefix_ids(key.system_template.format(prefix=d), key.user_message) for d in decoys]
        toks = [self.tokenize(t) for t in texts]
        results: list[ScoreResult | None] = [None] * len(texts)
        valid = []
        for i, t in enumerate(toks):
            n = len(t)
            if n == 0:
                results[i] = ScoreResult(score=0.0, status="empty", n_tokens=0, ell_key=0.0, ell_neutral=0.0, ctx_tokens_key=len(ctx_k))
            elif any(len(c) + n > self.max_input_tokens for c in [ctx_k] + ctx_ds):
                results[i] = ScoreResult(score=None, status="abstain_overflow", n_tokens=n, ctx_tokens_key=len(ctx_k))
            else:
                valid.append(i)
        if valid:
            targets = [toks[i] for i in valid]
            try:
                ell_k = self.backend.logprob_sums(ctx_k, targets)
                ell_d = [self.backend.logprob_sums(c, targets) for c in ctx_ds]
            except Exception as e:
                for i in valid:
                    results[i] = ScoreResult(score=None, status="error", n_tokens=len(toks[i]), error=repr(e))
            else:
                for j, i in enumerate(valid):
                    n = len(toks[i])
                    lk = float(ell_k[j]); ld = float(sum(x[j] for x in ell_d) / len(ell_d))
                    if not (math.isfinite(lk) and math.isfinite(ld)):
                        results[i] = ScoreResult(score=None, status="error", n_tokens=n, ell_key=lk, ell_neutral=ld, error="nonfinite")
                    else:
                        results[i] = ScoreResult(score=(lk - ld) / n, status="ok", n_tokens=n, ell_key=lk, ell_neutral=ld,
                                                 ctx_tokens_key=len(ctx_k), ctx_tokens_neutral=int(sum(len(c) for c in ctx_ds) / len(ctx_ds)))
        return results  # type: ignore[return-value]

    def per_token_contrasts(self, texts: list[str], key: KeySpec) -> list[list[float] | None]:
        """Per-token keyed-minus-neutral log-prob differences (amendment: within-text t-statistic detector)."""
        ctx_k = self.context_ids(key, neutral=False)
        ctx_0 = self.context_ids(key, neutral=True)
        toks = [self.tokenize(t) for t in texts]
        valid = [i for i, t in enumerate(toks) if t and len(ctx_k) + len(t) <= self.max_input_tokens and len(ctx_0) + len(t) <= self.max_input_tokens]
        out: list[list[float] | None] = [None] * len(texts)
        if not valid:
            return out
        lk = self.backend.logprob_tokens(ctx_k, [toks[i] for i in valid])
        l0 = self.backend.logprob_tokens(ctx_0, [toks[i] for i in valid])
        for j, i in enumerate(valid):
            out[i] = [a - b for a, b in zip(lk[j], l0[j])]
        return out

    def score_prefix_views(self, text: str, key: KeySpec, lengths: list[int]) -> dict[int, ScoreResult]:
        """Score token-prefixes of ``text`` (length diagnostics, spec Section 9)."""
        toks = self.tokenize(text)
        out: dict[int, ScoreResult] = {}
        views = [L for L in lengths if L <= len(toks)]
        if not views:
            return out
        texts = [self.backend.decode(toks[:L]) for L in views]
        res = self.score_texts(texts, key)
        for L, r in zip(views, res):
            out[L] = r
        return out


def score_to_dict(r: ScoreResult) -> dict:
    d = asdict(r)
    d["contrast_sum"] = r.contrast_sum
    return d
