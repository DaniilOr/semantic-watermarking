"""Backend protocol shared by the scorer and the generator, plus a deterministic
model-free fake LM used by the focused tests (spec Section 12: "a small
synthetic fixture that requires no model download").

A backend exposes exactly what generation and scoring need:

* ``encode_text`` / ``decode``: canonical text-only tokenization (no special tokens);
* ``chat_prefix_ids``: the rendered chat context (system + user + generation
  prompt with thinking disabled) as token ids;
* ``logprob_sums``: teacher-forced sum of log-probabilities of target token
  sequences given a context, computed in float32 or better;
* ``sample_units``: N independent continuations of at most ``max_new`` tokens
  from one immutable context, with a private RNG seed.
"""

from __future__ import annotations

from collections import deque

import hashlib
import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0                 # 0 disables top-k


@dataclass
class Candidate:
    token_ids: list[int]           # visible tokens only (EOS stripped)
    ended: bool                    # True if the continuation emitted EOS
    n_generated: int               # tokens sampled, including the EOS token if any


@dataclass
class BackendInfo:
    name: str
    model_id: str
    revision: str | None
    tokenizer_id: str | None
    dtype: str
    device: str
    attn_implementation: str | None
    kernels: dict = field(default_factory=dict)
    template_hash: str | None = None
    generation_prefix_text: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    info: BackendInfo
    eos_ids: list[int]
    max_input_tokens: int

    def encode_text(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def chat_prefix_ids(self, system: str, user: str) -> list[int]: ...
    def logprob_sums(self, ctx_ids: list[int], targets: list[list[int]]) -> list[float]: ...
    def sample_units(self, ctx_ids: list[int], n: int, max_new: int, sampling: SamplingParams,
                     seed: int, eos_allowed_from: int) -> list[Candidate]: ...
    def cost_snapshot(self) -> dict: ...


# --------------------------------------------------------------------------- #
# Shared sampling helpers (used by both backends on numpy/torch arrays)
# --------------------------------------------------------------------------- #

def apply_top_k_top_p_np(logits: np.ndarray, temperature: float, top_p: float, top_k: int) -> np.ndarray:
    """Return a probability vector after temperature, top-k and nucleus filtering."""
    z = logits.astype(np.float64)
    if temperature <= 0:
        p = np.zeros_like(z)
        p[int(np.argmax(z))] = 1.0
        return p
    z = z / temperature
    if top_k and top_k > 0 and top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z >= kth, z, -np.inf)
    z = z - np.max(z)
    p = np.exp(z)
    p /= p.sum()
    if 0 < top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        keep = cum - p[order] < top_p          # keep tokens until cumulative mass reaches top_p
        keep[0] = True
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return p


# --------------------------------------------------------------------------- #
# Fake LM: byte-level bigram model whose table shifts with the chat context
# --------------------------------------------------------------------------- #

class FakeLM:
    """Deterministic byte-level bigram LM.

    Vocabulary: 256 byte tokens + special tokens.  The next-byte distribution is
    ``softmax(W[prev] + alpha * K(ctx))`` where ``K`` is a pseudo-random vector
    derived from the hash of the *system message bytes* in the context, so a
    different secret prefix yields a different world while the same prefix
    yields identical likelihoods (neutral contrast is exactly zero).
    """

    EOS = 256
    IM_START = 257
    IM_END = 258
    THINK_OPEN = 259
    THINK_CLOSE = 260
    VOCAB = 261

    def __init__(self, alpha: float = 1.5, max_input_tokens: int = 4096, seed: int = 0,
                 eos_bias: float = -2.0, name: str = "fake"):
        self.alpha = alpha
        self.max_input_tokens = max_input_tokens
        self.eos_ids = [self.EOS]
        rng = np.random.default_rng(seed)
        # A bigram table that prefers letters/spaces and gives EOS a modest chance.
        W = rng.normal(0.0, 1.0, size=(self.VOCAB, self.VOCAB))
        letters = np.array([ord(c) for c in "abcdefghijklmnopqrstuvwxyz "])
        W[:, letters] += 3.0
        W[:, ord(".")] += 1.0
        W[:, ord("\n")] += 0.0
        W[:, self.EOS] += eos_bias
        W[:, self.IM_START] = -30.0
        W[:, self.IM_END] = -30.0
        W[:, self.THINK_OPEN] = -30.0
        W[:, self.THINK_CLOSE] = -30.0
        self.W = W
        self.info = BackendInfo(name=name, model_id="fake-bigram", revision="0", tokenizer_id="bytes",
                                dtype="float64", device="cpu", attn_implementation=None,
                                template_hash="fake-template-v1",
                                generation_prefix_text="<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self._cost = {"scoring_tokens": 0, "scoring_forward_calls": 0, "scored_sequences": 0,
                      "proposal_tokens": 0, "proposal_forward_calls": 0}

    # -- tokenization ------------------------------------------------------- #
    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")

    def chat_prefix_ids(self, system: str, user: str) -> list[int]:
        ids = [self.IM_START] + self.encode_text("system\n" + system.strip()) + [self.IM_END, 10]
        ids += [self.IM_START] + self.encode_text("user\n" + user.strip()) + [self.IM_END, 10]
        ids += [self.IM_START] + self.encode_text("assistant\n") + [self.THINK_OPEN, 10, 10, self.THINK_CLOSE, 10, 10]
        return ids

    # -- world shift from the context ---------------------------------------- #
    def _shift(self, ctx_ids: list[int]) -> np.ndarray:
        # The system message is everything between the first IM_START and IM_END.
        try:
            start = ctx_ids.index(self.IM_START)
            end = ctx_ids.index(self.IM_END, start)
            sys_bytes = bytes(i for i in ctx_ids[start + 1:end] if i < 256)
        except ValueError:
            sys_bytes = b""
        h = hashlib.sha256(sys_bytes).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        k = rng.normal(0.0, 1.0, size=self.VOCAB)
        k[256:] = 0.0
        return self.alpha * k

    def _logits(self, prev: int, shift: np.ndarray) -> np.ndarray:
        return self.W[prev] + shift

    def _logprobs(self, prev: int, shift: np.ndarray) -> np.ndarray:
        z = self._logits(prev, shift)
        z = z - z.max()
        return z - math.log(np.exp(z).sum())

    # -- scoring ------------------------------------------------------------- #
    def logprob_sums(self, ctx_ids: list[int], targets: list[list[int]]) -> list[float]:
        if not ctx_ids:
            raise ValueError("context must be non-empty")
        shift = self._shift(ctx_ids)
        out = []
        for t in targets:
            prev = ctx_ids[-1]
            total = 0.0
            for tok in t:
                total += float(self._logprobs(prev, shift)[tok])
                prev = tok
            out.append(total)
            self._cost["scoring_tokens"] += len(ctx_ids) + len(t)
            self._cost["scored_sequences"] += 1
        self._cost["scoring_forward_calls"] += 1
        return out

    def sample_poe_cross(self, ctx_ids, ref_key, ref_neutral, gen_to_ref, max_new, sampling, seed, eos_allowed_from,
                         beta, clip, gate, entropy_ref, entropy_min, steer_eos=True, block_ids=None, adapt="none",
                         t_target=5.0, beta_floor=0.0, repeat_n=0, ref_decoys=None, span_tokens=0, ref_key_factories=None,
                         t_window=0, warmup=0, d_max=0.0) -> dict:
        """Cross-model steering on the fakes: the contrast comes from a SECOND fake model with its own vocabulary."""
        from .runtime import adaptive_beta, running_t
        rng = np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, 11])
        sg = self._shift(ctx_ids)
        prev = ctx_ids[-1]
        g2r = np.asarray(gen_to_ref, dtype=np.int64)
        ids, ent, betas, contrasts, lp_g, kls, lploss, raw, truns = [], [], [], [], [], [], [], [], []
        ended = False
        block = sorted(set(int(b) for b in (block_ids or [])))
        ngram_next: dict = {}
        n_rep_blocked = n_unmapped = 0
        s1 = s2 = 0.0
        win: "deque[float]" = deque()
        span_starts: list = [0]
        rot = list(ref_key_factories or [])
        key_order: list = [0] if rot else []
        beta0 = float(beta)
        text = ""
        for step in range(max_new):
            z = self._logits(prev, sg).copy()
            if step < eos_allowed_from:
                z[self.EOS] = -np.inf
            p_g = apply_top_k_top_p_np(z, sampling.temperature, sampling.top_p, sampling.top_k)
            support = p_g > 0
            logp_g = np.log(p_g[support])
            h = float(-(p_g[support] * logp_g).sum())
            lp_key = ref_key.logprobs_after(text)
            dneutral = lp_key - ref_neutral.logprobs_after(text)
            dref = (lp_key - np.mean([d.logprobs_after(text) for d in ref_decoys], axis=0)) if ref_decoys else dneutral
            dfull = np.zeros(p_g.shape[0], dtype=np.float64)
            dneu_gen = np.zeros(p_g.shape[0], dtype=np.float64)
            idx = g2r[:p_g.shape[0]] if g2r.shape[0] >= p_g.shape[0] else np.concatenate(
                [g2r, np.full(p_g.shape[0] - g2r.shape[0], -1, dtype=np.int64)])
            ok = (idx >= 0) & (idx < dref.shape[0])
            dfull[ok] = dref[idx[ok]]
            dneu_gen[ok] = dneutral[idx[ok]]
            if not steer_eos:
                dfull[self.EOS] = 0.0
            if block:
                dfull[block] = 0.0
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                rep = ngram_next.get(tuple(ids[len(ids) - (repeat_n - 1):]))
                if rep:
                    dfull[list(rep)] = 0.0
                    n_rep_blocked += len(rep)
            n_unmapped += int((~ok[support]).sum())
            if d_max > 0:
                dfull[dfull > d_max] = 0.0
            d = np.clip(dfull[support], -clip, clip)
            if span_tokens and step and step % span_tokens == 0:
                s1 = s2 = 0.0; win.clear()
                span_starts.append(step)
                if rot:
                    idx = (len(span_starts) - 1) % len(rot)     # index of the span just started
                    ref_key = rot[idx]()
                    key_order.append(idx)
            t_run = running_t(s1, s2, len(win) if t_window else step - (span_starts[-1] if span_starts else 0))
            beta = adaptive_beta(beta0, adapt, t_run, t_target, beta_floor)
            if gate == "entropy":
                g = 0.0 if h <= entropy_min else min(1.0, (h - entropy_min) / max(1e-6, entropy_ref - entropy_min))
                beta_t = beta * g
            else:
                beta_t = float(beta)
            if warmup and step < warmup:
                beta_t *= step / float(warmup)
            w = logp_g + beta_t * d
            w -= w.max()
            p = np.exp(w); p /= p.sum()
            if not steer_eos:
                sup_idx = np.nonzero(support)[0]
                eos_mask = sup_idx == self.EOS
                if eos_mask.any():
                    p_eos = float(p_g[support][eos_mask].sum())
                    p = np.where(eos_mask, p_g[support], p * (1.0 - p_eos) / max(1e-30, float(p[~eos_mask].sum())))
                    p /= p.sum()
            kls_t = float((p * (np.log(p + 1e-30) - logp_g)).sum())
            lploss_t = float(((p_g[support] - p) * logp_g).sum())
            j = int(rng.choice(p.size, p=p))
            tok = int(np.nonzero(support)[0][j])
            r = float(dneu_gen[tok]) if tok != self.EOS else 0.0
            ent.append(h); betas.append(beta_t); contrasts.append(float(d[j])); lp_g.append(float(logp_g[j]))
            kls.append(kls_t); lploss.append(lploss_t); raw.append(r); truns.append(t_run)
            self._cost["proposal_tokens"] += 1
            if tok == self.EOS:
                ended = True
                break
            s1 += r; s2 += r * r
            if t_window:
                win.append(r)
                if len(win) > t_window:
                    o = win.popleft(); s1 -= o; s2 -= o * o
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                ngram_next.setdefault(tuple(ids[len(ids) - (repeat_n - 1):]), set()).add(tok)
            ids.append(tok); prev = tok
            text = self.decode(ids)
        return {"token_ids": ids, "ended": ended, "entropy": ent, "beta_t": betas, "contrast": contrasts, "logp_g": lp_g,
                "kl": kls, "lp_loss": lploss, "contrast_raw": raw, "t_run": truns, "n_repeat_blocked": n_rep_blocked,
                "span_starts": span_starts, "key_order": key_order, "n_rotation_keys": len(rot), "n_unmapped_candidates": n_unmapped, "n_ref_decoys": len(ref_decoys or []),
                "ref_forward_calls": ref_key.calls + ref_neutral.calls + sum(d.calls for d in (ref_decoys or [])),
                "ref_recomputed_tokens": 0}

    def logprob_tokens(self, ctx_ids: list[int], targets: list[list[int]]) -> list[list[float]]:
        shift = self._shift(ctx_ids)
        out = []
        for t in targets:
            prev = ctx_ids[-1]
            row = []
            for tok in t:
                row.append(float(self._logprobs(prev, shift)[tok]))
                prev = tok
            out.append(row)
        return out

    # -- proposal sampling --------------------------------------------------- #
    def sample_units(self, ctx_ids: list[int], n: int, max_new: int, sampling: SamplingParams,
                     seed: int, eos_allowed_from: int) -> list[Candidate]:
        shift = self._shift(ctx_ids)
        cands = []
        for j in range(n):
            rng = np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, j])
            prev = ctx_ids[-1]
            ids: list[int] = []
            ended = False
            n_gen = 0
            for step in range(max_new):
                z = self._logits(prev, shift).copy()
                if step < eos_allowed_from:
                    z[self.EOS] = -np.inf
                p = apply_top_k_top_p_np(z, sampling.temperature, sampling.top_p, sampling.top_k)
                tok = int(rng.choice(self.VOCAB, p=p))
                n_gen += 1
                if tok == self.EOS:
                    ended = True
                    break
                ids.append(tok)
                prev = tok
            cands.append(Candidate(token_ids=ids, ended=ended, n_generated=n_gen))
            self._cost["proposal_tokens"] += n_gen
        self._cost["proposal_forward_calls"] += max_new
        return cands

    def cost_snapshot(self) -> dict:
        return dict(self._cost)

    def sample_poe(self, ctx_ids, key_ctx_ids, neutral_ctx_ids, max_new, sampling, seed, eos_allowed_from, beta, clip,
                   gate, entropy_ref, entropy_min, kl_budget=0.1, steer_eos=True, block_ids=None, adapt="none",
                   t_target=5.0, beta_floor=0.0, repeat_n=0, decoy_ctx_ids=None, span_tokens=0,
                   t_window=0, warmup=0, d_max=0.0) -> dict:
        """Token-level product-of-experts steering on the fake bigram world (same rule as the real runtime)."""
        from .runtime import adaptive_beta, running_t
        rng = np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, 7])
        sg, sk, s0 = self._shift(ctx_ids), self._shift(key_ctx_ids), self._shift(neutral_ctx_ids)
        sdec = [self._shift(c) for c in (decoy_ctx_ids or [])]
        prev = ctx_ids[-1]
        ids, ent, betas, contrasts, lp_g, kls, lploss, raw, truns, steer_raw = [], [], [], [], [], [], [], [], [], []
        ended = False
        block = sorted(set(int(b) for b in (block_ids or [])))
        s1 = s2 = 0.0
        win: "deque[float]" = deque()
        span_starts: list = [0]
        beta0 = float(beta)
        ngram_next: dict = {}
        n_rep_blocked = 0
        for step in range(max_new):
            z = self._logits(prev, sg).copy()
            if step < eos_allowed_from:
                z[self.EOS] = -np.inf
            p_g = apply_top_k_top_p_np(z, sampling.temperature, sampling.top_p, sampling.top_k)
            support = p_g > 0
            logp_g = np.log(p_g[support])
            h = float(-(p_g[support] * logp_g).sum())
            lpk = self._logprobs(prev, sk)
            dneutral = lpk - self._logprobs(prev, s0)
            dfull = (lpk - np.mean([self._logprobs(prev, sd) for sd in sdec], axis=0)) if sdec else dneutral.copy()
            if not steer_eos:
                dfull[self.EOS] = 0.0
            if block:
                dfull[block] = 0.0
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                rep = ngram_next.get(tuple(ids[len(ids) - (repeat_n - 1):]))
                if rep:
                    dfull[list(rep)] = 0.0
                    n_rep_blocked += len(rep)
            if d_max > 0:
                dfull[dfull > d_max] = 0.0
            d = np.clip(dfull[support], -clip, clip)
            if span_tokens and step and step % span_tokens == 0:
                s1 = s2 = 0.0; win.clear()
                span_starts.append(step)
            t_run = running_t(s1, s2, len(win) if t_window else step - (span_starts[-1] if span_starts else 0))
            beta = adaptive_beta(beta0, adapt, t_run, t_target, beta_floor)
            def kl_at(b):
                w = logp_g + b * d; w = w - w.max(); q = np.exp(w); q /= q.sum()
                return float((q * (np.log(q + 1e-30) - logp_g)).sum())
            if gate == "entropy":
                g = 0.0 if h <= entropy_min else min(1.0, (h - entropy_min) / max(1e-6, entropy_ref - entropy_min))
                beta_t = beta * g
            elif gate == "kl":
                if kl_at(beta) <= kl_budget:
                    beta_t = float(beta)
                else:
                    lo, hi = 0.0, float(beta)
                    for _ in range(12):
                        mid = 0.5 * (lo + hi)
                        if kl_at(mid) <= kl_budget: lo = mid
                        else: hi = mid
                    beta_t = lo
            elif gate == "lp":
                ref = float((np.exp(logp_g) * logp_g).sum()); beta_t = 0.0
                for i in range(1, 17):
                    b = beta * i / 16
                    w = logp_g + b * d; w = w - w.max(); q = np.exp(w); q /= q.sum()
                    if float((q * logp_g).sum()) >= ref - kl_budget: beta_t = b
                    else: break
            else:
                beta_t = float(beta)
            if warmup and step < warmup:
                beta_t *= step / float(warmup)
            w = logp_g + beta_t * d
            w -= w.max()
            p = np.exp(w); p /= p.sum()
            if not steer_eos:
                sup_idx = np.nonzero(support)[0]
                eos_mask = sup_idx == self.EOS
                if eos_mask.any():
                    p_eos = float(p_g[support][eos_mask].sum())
                    rest = float(p[~eos_mask].sum())
                    p = np.where(eos_mask, p_g[support], p * (1.0 - p_eos) / max(1e-30, rest))
                    p /= p.sum()
            kls_t = float((p * (np.log(p + 1e-30) - logp_g)).sum())
            lploss_t = float(((p_g[support] - p) * logp_g).sum())
            j = int(rng.choice(p.size, p=p))
            tok = int(np.nonzero(support)[0][j])
            r = float(dneutral[tok]) if tok != self.EOS else 0.0
            ent.append(h); betas.append(beta_t); contrasts.append(float(d[j])); lp_g.append(float(logp_g[j])); kls.append(kls_t); lploss.append(lploss_t)
            raw.append(r); truns.append(t_run); steer_raw.append(float(dfull[tok]))
            self._cost["proposal_tokens"] += 1
            if tok == self.EOS:
                ended = True
                break
            s1 += r; s2 += r * r
            if t_window:
                win.append(r)
                if len(win) > t_window:
                    o = win.popleft(); s1 -= o; s2 -= o * o
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                ngram_next.setdefault(tuple(ids[len(ids) - (repeat_n - 1):]), set()).add(tok)
            ids.append(tok); prev = tok
        return {"token_ids": ids, "ended": ended, "entropy": ent, "beta_t": betas, "contrast": contrasts, "logp_g": lp_g, "kl": kls,
                "lp_loss": lploss, "contrast_raw": raw, "t_run": truns, "n_repeat_blocked": n_rep_blocked, "contrast_steer_raw": steer_raw,
                "span_starts": span_starts}

    # activation steering is a no-op for the fake LM (the extension is model-backed); the interface exists so the
    # generator's code path can be exercised in tests
    def steering_vectors(self, keyed_system: str, neutral_system: str, user: str) -> dict:
        return {"vectors": None, "norms": [], "residual_norms": [], "n_layers": 0}

    def set_steering(self, vectors, alpha: float = 0.0, layers=None) -> None:
        self._steer_alpha = alpha if vectors is not None else 0.0


class FakeRefLM:
    """A second fake language model with its own (different) vocabulary and tokenizer, for cross-model tests.

    By default one token per byte over a smaller, differently-indexed vocabulary (the bridge is meaningful); with
    ``pair=True`` it merges byte pairs, so its token boundaries never align with the generator's, which is the
    degenerate case the loop must still survive.
    """

    VOCAB = 97

    def __init__(self, alpha: float = 1.5, seed: int = 7, max_input_tokens: int = 4096, pair: bool = False):
        self.pair = pair                      # True: merge byte PAIRS (token boundaries never align with the generator)
        self.alpha = alpha
        self.max_input_tokens = max_input_tokens
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0, size=(self.VOCAB, self.VOCAB))
        self.info = {"model_id": "fake-ref", "vocab_size": self.VOCAB}
        self.vocab_size = self.VOCAB
        self.eos_ids: list[int] = []

    # -- tokenizer: byte pairs ------------------------------------------------ #
    def encode_text(self, text: str) -> list[int]:
        b = text.encode("utf-8")
        if not self.pair:
            return [x % self.VOCAB for x in b]         # one token per byte, but a different vocabulary than the generator
        out = []
        for i in range(0, len(b) - 1, 2):
            out.append((b[i] * 31 + b[i + 1]) % self.VOCAB)
        if len(b) % 2:
            out.append(b[-1] % self.VOCAB)
        return out

    def decode(self, ids: list[int]) -> str:                      # lossy; never used by the detector
        return "".join(chr(97 + (i % 26)) for i in ids)

    def chat_prefix_ids(self, system: str, user: str) -> list[int]:
        return self.encode_text(f"{system}\n\n{user}\n\n")

    def _shift(self, ctx_ids: list[int]) -> np.ndarray:
        h = int(hashlib.sha256(bytes(bytearray((i % 251) for i in ctx_ids))).hexdigest()[:16], 16)
        return np.random.default_rng(h % (2 ** 32)).normal(0.0, 1.0, size=self.VOCAB) * self.alpha

    def _logprobs(self, prev: int, shift: np.ndarray) -> np.ndarray:
        z = self.W[prev % self.VOCAB] + shift
        z = z - z.max()
        p = np.exp(z)
        return np.log(p / p.sum())

    # -- teacher forcing (detection) ------------------------------------------ #
    def logprob_tokens(self, ctx_ids: list[int], targets: list[list[int]]) -> list[list[float]]:
        shift = self._shift(ctx_ids)
        out = []
        for t in targets:
            prev = ctx_ids[-1] if ctx_ids else 0
            lp = []
            for tok in t:
                lp.append(float(self._logprobs(prev, shift)[tok]))
                prev = tok
            out.append(lp)
        return out

    def logprob_sums(self, ctx_ids: list[int], targets: list[list[int]]) -> list[float]:
        return [float(sum(x)) for x in self.logprob_tokens(ctx_ids, targets)]

    def position_stats(self, ctx_neutral: list[int], ctx_keys: list[list[int]], targets: list[list[int]],
                       token_budget: int = 0) -> list[dict]:
        """Same contract as ``ReferenceRuntime.position_stats`` (h0 [n]; d, mu, var [K, n])."""
        s0 = self._shift(ctx_neutral)
        sk = [self._shift(c) for c in ctx_keys]
        out = []
        for t in targets:
            if not t:
                out.append(None); continue
            n, K = len(t), len(ctx_keys)
            h0 = np.zeros(n, np.float32); d = np.zeros((K, n), np.float32)
            mu = np.zeros((K, n), np.float32); var = np.zeros((K, n), np.float32)
            prev0 = ctx_neutral[-1] if ctx_neutral else 0
            prevk = [(c[-1] if c else 0) for c in ctx_keys]
            for j, tok in enumerate(t):
                l0 = self._logprobs(prev0, s0); p0 = np.exp(l0)
                h0[j] = -(p0 * l0).sum()
                for k in range(K):
                    D = self._logprobs(prevk[k], sk[k]) - l0
                    d[k, j] = D[tok]; m = (p0 * D).sum(); mu[k, j] = m; var[k, j] = max(0.0, (p0 * D * D).sum() - m * m)
                prev0 = tok; prevk = [tok] * K
            out.append({"h0": h0, "d": d, "mu": mu, "var": var})
        return out

    # -- incremental stepping (generation) ------------------------------------ #
    class Stepper:
        def __init__(self, rt: "FakeRefLM", ctx_ids: list[int]):
            self.rt = rt
            self.ctx = list(ctx_ids)
            self.shift = rt._shift(ctx_ids)
            self.calls = 0

        def logprobs_after(self, text: str) -> np.ndarray:
            ids = self.rt.encode_text(text)
            prev = ids[-1] if ids else (self.ctx[-1] if self.ctx else 0)
            self.calls += 1
            return self.rt._logprobs(prev, self.shift)

    def stepper(self, ctx_ids: list[int], window: int = 0, stride: int = 0) -> "FakeRefLM.Stepper":
        return FakeRefLM.Stepper(self, ctx_ids)      # a bigram model only ever sees one token: every window is equivalent

    def token_map(self, gen_backend, cache_path=None, log=None) -> np.ndarray:
        n = gen_backend.VOCAB if hasattr(gen_backend, "VOCAB") else 261
        out = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            ids = self.encode_text(gen_backend.decode([i]))
            if ids:
                out[i] = ids[0]
        return out
