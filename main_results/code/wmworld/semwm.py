"""Semantic (embedding-space) watermark: keyed LSH partition, token-level steering bias, text-only detection.

SemStamp (Hou et al.) accepts a sentence only if its sentence embedding falls in a valid LSH bin and sits at least a
margin away from every hyperplane; on our generator that costs ~25 candidate sentences per accepted one, most of it
the margin.  This module keeps SemStamp's partition and statistics exactly (nearpy hyperplanes from
``RandomState(1234)``, the mask seeded by ``15485863 * previous bin``) and adds what SemStamp cannot do: a per-token
logit bias that pushes the embedding of the unit being written toward its target region, so that rejection becomes
rare (sentence arms) or unnecessary (segment arm with a continuous statistic).

Everything a detector needs is a function of the text and the key material alone.  The mask PRNG is numpy so that a
CPU detector reproduces the bins of a GPU generator bit for bit (SemStamp's is a CUDA ``randperm``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor, StoppingCriteria

from .textnorm import normalize_text

HASH_KEY = 15485863            # SemStamp sampling_utils.hash_key
RAND_SEED = 1234               # SemStamp sbert_lsh_model: RandomBinaryProjections(..., rand_seed=1234)
LSH_DIM = 3
LMBD = 0.25
MARGIN = 0.02

# One boundary rule at both ends: sentence punctuation (with closing quotes/brackets) followed by whitespace, or a
# newline.  "arts.After" is NOT a boundary for either side, so generation and detection cannot disagree on it.
BOUNDARY_RE = re.compile(r"[.!?][\"')\]]*(?=\s)|\n")


# --------------------------------------------------------------------------- #
# text units
# --------------------------------------------------------------------------- #

def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        seg = text[start:m.end()].strip()
        if seg:
            out.append(seg)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_segments(text: str, W: int) -> list[str]:
    """Fixed-length units of W whitespace words; the last one is kept when it has at least W // 2 words."""
    words = normalize_text(text).split()
    segs = [" ".join(words[i:i + W]) for i in range(0, len(words), W)]
    if len(segs) > 1 and len(segs[-1].split()) < max(2, W // 2):
        segs = segs[:-1]
    return segs


LEAD_JOIN = False   # set by steered_semstamp --join-leading-newline


def first_boundary(text: str) -> int | None:
    if LEAD_JOIN:
        # 2026-09-30: leading whitespace / newlines are not a sentence -- they join the next one, as the detector sees it
        # (split_sentences drops empty segments).  Without this, a tokenizer that emits "\n" apart from the "." (OPT)
        # made every line break a 1-token unit of its own, shifting the generator's mask chain away from the detector's.
        m = BOUNDARY_RE.search(text, len(text) - len(text.lstrip()))
    else:
        m = BOUNDARY_RE.search(text)
    return m.end() if m else None


# --------------------------------------------------------------------------- #
# key material and partition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Key:
    rand_seed: int = RAND_SEED
    hash_key: int = HASH_KEY
    lsh_dim: int = LSH_DIM

    @classmethod
    def shipped(cls) -> "Key":
        return cls()

    @classmethod
    def alt(cls, i: int) -> "Key":
        """Wrong-key diagnostic: different hyperplanes AND a different mask stream."""
        return cls(rand_seed=RAND_SEED + i, hash_key=HASH_KEY + 2 * i)

    def label(self) -> str:
        return f"seed{self.rand_seed}_hk{self.hash_key}"


class Hyperplanes:
    """nearpy's convention: raw normals ``RandomState(seed).randn(d, dim)``; bit j is ``proj_j > 0``, plane 0 is the MSB."""

    def __init__(self, normals: np.ndarray):
        self.raw = np.asarray(normals, dtype=np.float64)
        self.unit = self.raw / np.linalg.norm(self.raw, axis=1, keepdims=True)
        self.d = self.raw.shape[0]

    @classmethod
    def for_key(cls, key: Key, dim: int) -> "Hyperplanes":
        return cls(np.random.RandomState(key.rand_seed).randn(key.lsh_dim, dim))

    def projections(self, emb: np.ndarray) -> np.ndarray:
        """Cosines of unit embeddings [n, dim] with the planes -> [n, d]."""
        return np.asarray(emb, dtype=np.float64) @ self.unit.T

    def bin_of(self, proj: np.ndarray) -> int:
        b = 0
        for x in proj:
            b = (b << 1) | int(x > 0.0)
        return b

    def bin_signs(self, b: int) -> np.ndarray:
        return np.array([1.0 if (b >> (self.d - 1 - j)) & 1 else -1.0 for j in range(self.d)])

    def margin_ok(self, proj: np.ndarray, margin: float) -> bool:
        return bool(np.all(np.abs(proj) >= margin))


def valid_mask(key: Key, prev_bin: int, lmbd: float = LMBD) -> tuple[int, ...]:
    """SemStamp's mask: a pseudo-random ``int(2^d * lmbd)`` of the ``2^d`` bins, seeded by ``hash_key * prev_bin``."""
    n = 2 ** key.lsh_dim
    rng = np.random.default_rng(key.hash_key * int(prev_bin))
    return tuple(int(x) for x in rng.permutation(n)[: int(n * lmbd)])


def fixed_bin(key: Key, lmbd: float = LMBD) -> int:
    """The segment arm's single target bin for the whole text (no chain): first valid bin of the seed-0 mask."""
    return valid_mask(key, 0, lmbd)[0]


def target_direction(planes: Hyperplanes, signs: np.ndarray, active: np.ndarray) -> np.ndarray | None:
    """Unit direction toward the octant ``signs`` along the planes still ``active`` (None when all are satisfied)."""
    if not np.any(active):
        return None
    u = (planes.unit * (signs * active)[:, None]).sum(axis=0)
    n = np.linalg.norm(u)
    return u / n if n > 0 else None


def signed_distance(proj: np.ndarray, bins: tuple[int, ...], planes: Hyperplanes) -> float:
    """Depth inside the valid region: max over valid bins of the smallest signed projection; > 0 iff inside one."""
    return max(float(np.min(planes.bin_signs(b) * proj)) for b in bins)


# --------------------------------------------------------------------------- #
# statistics (text + key only)
# --------------------------------------------------------------------------- #

def chain_z(bins: list[int], key: Key, lmbd: float = LMBD) -> float | None:
    """SemStamp ``detect_lsh``: sentence 1 seeds; count sentences 2..n whose bin is in the mask of the previous one."""
    n_test = len(bins) - 1
    if n_test < 1:
        return None
    n_wm = sum(1 for i in range(1, len(bins)) if bins[i] in valid_mask(key, bins[i - 1], lmbd))
    return (n_wm - lmbd * n_test) / math.sqrt(n_test * lmbd * (1.0 - lmbd))


def cont_stat(projs: np.ndarray, key: Key, planes: Hyperplanes, lmbd: float = LMBD, clip: float = 0.05) -> float | None:
    """Continuous chain statistic: clipped depth inside the previous sentence's mask, summed over sentences 2..n, / sqrt(n-1)."""
    n_test = len(projs) - 1
    if n_test < 1:
        return None
    bins = [planes.bin_of(p) for p in projs]
    s = [max(-clip, min(clip, signed_distance(projs[i], valid_mask(key, bins[i - 1], lmbd), planes))) for i in range(1, len(projs))]
    return float(sum(s) / math.sqrt(n_test))


def seg_stat(projs: np.ndarray, target: int, planes: Hyperplanes, clip: float = 0.05) -> float | None:
    """Segment statistic: clipped depth inside ONE fixed bin, every unit counts, exactly invariant to unit order."""
    if len(projs) < 1:
        return None
    signs = planes.bin_signs(target)
    s = np.sort(np.clip(np.min(signs * projs, axis=1), -clip, clip))     # sorted sum: bit-identical under any unit order
    return float(s.sum() / math.sqrt(len(projs)))


def entropy_gate(h: float, h_min: float = 0.2, h_ref: float = 2.0) -> float:
    """The generator's own gate (wmworld/runtime.py): no push where the next token is already decided."""
    return 0.0 if h <= h_min else min(1.0, (h - h_min) / max(1e-6, h_ref - h_min))


# --------------------------------------------------------------------------- #
# embedder and token table
# --------------------------------------------------------------------------- #

class Embedder:
    def __init__(self, path: str, device: str = "cuda", batch_size: int = 64, prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.path, self.prefix, self.batch_size = path, prefix, batch_size
        self.model = SentenceTransformer(path, device=device)
        self.device = device
        self.dim = int(self.model.get_embedding_dimension())
        self.n_calls = 0

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-norm embeddings [n, dim] as float32 numpy (empty input -> [0, dim])."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.n_calls += 1
        v = self.model.encode([self.prefix + t for t in texts], batch_size=self.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def token_table(embedder: Embedder, tokenizer, cache_path: str | Path, log=print) -> torch.Tensor:
    """[V, dim] unit embeddings of each lexical token's surface string (zero rows for specials, byte fragments, punctuation)."""
    cache_path = Path(cache_path)
    V = len(tokenizer)
    if cache_path.exists():
        z = np.load(cache_path)
        if int(z["vocab"]) == V and str(z["embedder"]) == embedder.path:
            return torch.as_tensor(z["table"], device=embedder.device)
    specials = set(int(i) for i in tokenizer.all_special_ids)
    strings, ids = [], []
    for v in range(V):
        if v in specials:
            continue
        s = tokenizer.decode([v])
        if "�" in s or not any(c.isalnum() for c in s):
            continue
        strings.append(s.strip()); ids.append(v)
    table = np.zeros((V, embedder.dim), dtype=np.float32)
    for s0 in range(0, len(strings), 4096):
        table[ids[s0:s0 + 4096]] = embedder.encode(strings[s0:s0 + 4096])
        if log:
            log(f"[token-table] {min(s0 + 4096, len(strings))}/{len(strings)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, table=table, vocab=V, embedder=embedder.path, n_lexical=len(ids))
    return torch.as_tensor(table, device=embedder.device)


# --------------------------------------------------------------------------- #
# steering (generation side)
# --------------------------------------------------------------------------- #

class SteerBiasProcessor(LogitsProcessor):
    """Logit bias toward a target region of the unit being written.  Prefix-pure: its only state is a cache keyed by
    the token prefix of the current unit, so the drafter's and the verifier's calls on the same prefix see the same
    bias (transformers forwards custom processors to the assistant model).

    ``seg_W``: segment arm -- the unit is the current partial W-word segment of the text since ``L0`` (the same
    segmentation the detector applies to the finished text); otherwise the unit is everything since ``L0`` (the
    sentence being written)."""

    def __init__(self, etok: torch.Tensor, planes: Hyperplanes, embedder: Embedder, tokenizer, L0: int, target_bin: int,
                 beta: float, k: int = 8, margin: float = MARGIN, clip: float = 2.0, eos_ids: list[int] | None = None,
                 gate: tuple[float, float] = (0.2, 2.0), zero_ids: list[int] | None = None, seg_W: int | None = None):
        self.etok, self.planes, self.embedder, self.tokenizer = etok, planes, embedder, tokenizer
        self.L0, self.target, self.beta, self.k, self.margin, self.clip, self.gate = L0, int(target_bin), float(beta), int(k), margin, clip, gate
        self.seg_W = seg_W
        self.signs = planes.bin_signs(self.target)
        self.device = etok.device
        self.eos_ids = [int(e) for e in (eos_ids or [])]
        self.zero_ids = torch.tensor(sorted(set(self.eos_ids + [int(z) for z in (zero_ids or [])])), dtype=torch.long, device=self.device)
        self._active: dict[tuple[int, ...], tuple[bool, ...]] = {}       # block prefix -> planes still to push
        self._bias: dict[tuple[bool, ...], torch.Tensor] = {}            # active set -> [V] bias (before gate/beta)
        self._lexical = (etok.norm(dim=1) > 0)
        self.stats = {"n_embed_calls": 0, "n_calls": 0, "planes_dropped": 0, "all_satisfied": 0}

    # ---- active planes for a prefix (embedding-derived, cached) ---------------------------------- #
    def _active_for(self, block: tuple[int, ...]) -> tuple[bool, ...]:
        a = self._active.get(block)
        if a is not None:
            return a
        if not block:
            a = tuple([True] * self.planes.d)
        else:
            ids = [t for t in block if t not in self.eos_ids and t < len(self.tokenizer)]
            text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            if self.seg_W:
                words = text.split()
                text = " ".join(words[(len(words) // self.seg_W) * self.seg_W:])
            if not text:
                a = tuple([True] * self.planes.d)
            else:
                proj = self.planes.projections(self.embedder.encode([text]))[0]
                self.stats["n_embed_calls"] += 1
                a = tuple(bool(not (s * p >= self.margin)) for s, p in zip(self.signs, proj))
                self.stats["planes_dropped"] += int(self.planes.d - sum(a))
        self._active[block] = a
        return a

    def _bias_for(self, active: tuple[bool, ...]) -> torch.Tensor:
        b = self._bias.get(active)
        if b is not None:
            return b
        u = target_direction(self.planes, self.signs, np.array(active, dtype=bool))
        if u is None:
            b = torch.zeros(self.etok.shape[0], device=self.device)
        else:
            cos = self.etok @ torch.as_tensor(u, dtype=self.etok.dtype, device=self.device)
            sd = float(cos[self._lexical].std().item()) or 1.0
            b = torch.clamp(cos / sd, -self.clip, self.clip)
            b[self.zero_ids] = 0.0
        self._bias[active] = b
        return b

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.stats["n_calls"] += 1
        if self.beta == 0.0:
            return scores
        out = scores.clone()
        for r in range(input_ids.shape[0]):
            prefix = tuple(int(t) for t in input_ids[r, self.L0:].tolist())
            block = prefix[: len(prefix) - (len(prefix) % self.k)]
            active = self._active_for(block)
            if not any(active):
                self.stats["all_satisfied"] += 1
                continue
            p = torch.softmax(scores[r].float(), dim=-1)
            h = float(-(p * torch.log(p + 1e-30)).sum().item())
            g = entropy_gate(h, *self.gate)
            if g > 0:
                out[r] = out[r] + (self.beta * g) * self._bias_for(active).to(out.dtype)
        return out


def boundary_token_mask(tokenizer, eos_ids: list[int], device, vocab_size: int | None = None) -> torch.Tensor:
    """[V] bool: tokens that carry structure, not content (EOS, specials, and surface strings without a letter or digit:
    punctuation, whitespace, newlines).  The nucleus steerer never biases them and keeps their total mass unchanged.
    vocab_size = the model's logits width when it exceeds the tokenizer (OPT 50272 vs 50265, Qwen3.5 248320 vs 248077):
    the extra ids are not tokens and count as boundary (never pushed, mass unchanged); without it, a nucleus that reaches
    them (top-p 1.0) indexes past the mask (CUDA device-side assert, 2026-09-30)."""
    V_tok = len(tokenizer)
    V = max(V_tok, int(vocab_size or 0))
    m = torch.zeros(V, dtype=torch.bool)
    m[V_tok:] = True
    for v in set(int(i) for i in tokenizer.all_special_ids) | set(int(e) for e in eos_ids):
        if 0 <= v < V:
            m[v] = True
    for v in range(V_tok):
        if not m[v]:
            s = tokenizer.decode([v])
            m[v] = ("�" not in s) and not any(c.isalnum() for c in s)
    return m.to(device)


class NucleusSteerProcessor(LogitsProcessor):
    """Round-2 steering: push only among the model's own plausible tokens, never change how likely a boundary is.

    Per row and step (before temperature / top-p, like every custom processor):
      1. nucleus = top-p set of the UNBIASED tempered distribution; tokens outside get -inf (so the push can never pull a
         tail token in -- the "SQLite SQLite" failure); the bias itself goes to the ``topk`` most likely lexical members.
      2. gain per candidate v: ``table`` = round-1 token table ``etok[v] . u`` (u toward the planes not yet ``target_depth``
         deep); ``marginal`` = the exact per-unit score of the unit being written with v appended, score =
         min(target_depth, depth) where depth is the min signed projection over the bin's planes (``stat='seg'``) or the
         projection on the target direction (``stat='dir'``).  Once every candidate clears target_depth the gains are
         equal and nothing is pushed.
      3. gains standardised over the lexical nucleus (weights p), clamped to +-clip_sd, times beta * entropy gate:
         peak |bias| <= beta * clip_sd raw logits.
      4. repetition guard: a token that would complete an n-gram already present in the generated text gets no positive bias.
      5. mass-preserving tilt: the boundary class (EOS, punctuation, whitespace) is not biased and the lexical class is
         shifted so that its total tempered mass -- hence P(boundary) -- equals the model's own.
    At beta == 0 the scores are returned untouched.  ``L0`` = start of the unit stream (whole answer for the segment arm,
    chunk start for best-of-N); ``rep_L0`` = start of the generated text (repetition guard); ``seg_W`` slices the current
    W-word unit exactly as ``split_segments`` would."""

    def __init__(self, planes: Hyperplanes, embedder: Embedder, tokenizer, L0: int, target_bin: int, beta: float,
                 boundary: torch.Tensor, temperature: float, top_p: float, mode: str = "marginal", etok: torch.Tensor | None = None,
                 stat: str = "seg", topk: int = 20, clip_sd: float = 1.0, target_depth: float = 0.05, rep_n: int = 4,
                 rep_L0: int | None = None, gate: tuple[float, float] = (0.2, 2.0), seg_W: int | None = None, k: int = 8,
                 eos_ids: list[int] | None = None, target_bins: tuple[int, ...] | None = None):
        assert mode in ("marginal", "table") and stat in ("seg", "dir")
        self.planes, self.embedder, self.tokenizer = planes, embedder, tokenizer
        self.L0, self.rep_L0 = int(L0), int(L0 if rep_L0 is None else rep_L0)
        self.target, self.beta, self.mode, self.stat = int(target_bin), float(beta), mode, stat
        self.boundary, self.T, self.top_p, self.topk = boundary, float(temperature), float(top_p), int(topk)
        self.clip_sd, self.target_depth, self.rep_n, self.gate, self.seg_W = float(clip_sd), float(target_depth), int(rep_n), gate, seg_W
        self.eos_ids = [int(e) for e in (eos_ids or [])]
        self.signs = planes.bin_signs(self.target)
        # chain arm: the target is a mask of several bins; a unit's depth is its depth in the nearest of them
        self.bins = tuple(int(b) for b in target_bins) if target_bins else (self.target,)
        self.signs_all = np.stack([planes.bin_signs(b) for b in self.bins])
        self.u = target_direction(planes, self.signs, np.ones(planes.d, dtype=bool))
        self._table = None
        if mode == "table":
            assert etok is not None
            self._table = SteerBiasProcessor(etok, planes, embedder, tokenizer, L0, target_bin, 1.0, k, target_depth,
                                             eos_ids=eos_ids, seg_W=seg_W)
        self.stats = {"n_calls": 0, "n_pushed": 0, "n_embed_calls": 0, "bias_abs_sum": 0.0, "bias_abs_max": 0.0,
                      "p_boundary_before": 0.0, "p_boundary_after": 0.0, "rep_guard_hits": 0, "n_strings": 0}

    # ---- helpers ---------------------------------------------------------------------------------- #
    def _score(self, emb: np.ndarray) -> np.ndarray:
        e = np.asarray(emb, dtype=np.float64)
        if self.stat == "seg":
            P = self.planes.projections(e)
            s = np.max(np.min(self.signs_all[None, :, :] * P[:, None, :], axis=2), axis=1)     # max over bins, min over planes
        else:
            s = e @ self.u
        return np.minimum(self.target_depth, s)

    def _unit_texts(self, prefix: list[int], cands: list[int]) -> list[str]:
        """Text of the unit being written if each candidate were appended (decoded as it would really read)."""
        dec = lambda ids: self.tokenizer.decode([t for t in ids if t not in self.eos_ids], skip_special_tokens=True)
        text = dec(prefix)
        tail = prefix[-8:]
        tail_text = dec(tail)
        words = text.split()
        start = ((len(words) - 1) // self.seg_W) * self.seg_W if (self.seg_W and words) else 0
        out = []
        for v in cands:
            full = dec(tail + [v])
            t = text + full[len(tail_text):] if full.startswith(tail_text) else dec(prefix + [v])
            w = t.split()[start:]
            out.append(" ".join(w[: self.seg_W] if self.seg_W else w))
        return out

    def _repeat_banned(self, gen: list[int]) -> set[int]:
        n = self.rep_n
        if n < 2 or len(gen) < n:
            return set()
        tail = gen[-(n - 1):]
        return {gen[i + n - 1] for i in range(len(gen) - n + 1) if gen[i:i + n - 1] == tail}

    def _nucleus(self, s: torch.Tensor) -> torch.Tensor:
        """Indices of the top-p set of softmax(s / T), same rule as transformers' TopPLogitsWarper."""
        p = torch.softmax(s / self.T, dim=-1)
        sp, idx = torch.sort(p, descending=False)
        keep = torch.cumsum(sp, dim=-1) > (1.0 - self.top_p)
        keep[-1] = True
        return idx[keep]

    # ---- main ----------------------------------------------------------------------------------- #
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.stats["n_calls"] += 1
        if self.beta == 0.0:
            return scores
        out = scores.clone()
        jobs = []                                                        # (row, lexical ids, p_lex, gate, strings|gains)
        for r in range(input_ids.shape[0]):
            s = scores[r].float()
            p_raw = torch.softmax(s, dim=-1)
            h = float(-(p_raw * torch.log(p_raw + 1e-30)).sum().item())
            g = entropy_gate(h, *self.gate)
            if g <= 0:
                continue
            nuc = self._nucleus(s)
            lex = nuc[~self.boundary[nuc]]
            if lex.numel() < 2:
                continue
            p_nuc = torch.softmax(s[nuc] / self.T, dim=-1)
            p_lex = torch.softmax(s[lex] / self.T, dim=-1)
            order = torch.argsort(p_lex, descending=True)[: self.topk]
            push = lex[order]                                            # the lexical tokens that receive a bias
            prefix = [int(t) for t in input_ids[r, self.L0:].tolist()]
            if self.mode == "table":
                block = tuple(prefix[: len(prefix) - (len(prefix) % self._table.k)])
                active = self._table._active_for(block)
                u = target_direction(self.planes, self.signs, np.array(active, dtype=bool))
                if u is None:
                    continue
                gains = (self._table.etok[push] @ torch.as_tensor(u, dtype=self._table.etok.dtype, device=self._table.etok.device)).double().cpu().numpy()
                jobs.append([r, s, nuc, lex, push, p_lex[order], p_nuc, g, None, gains])
            else:
                strings = self._unit_texts(prefix, [int(v) for v in push.tolist()])
                jobs.append([r, s, nuc, lex, push, p_lex[order], p_nuc, g, strings, None])
        if not jobs:
            return out
        if self.mode == "marginal":
            flat = [t for j in jobs for t in j[8]]
            e = self.embedder.encode(flat)
            self.stats["n_embed_calls"] += 1; self.stats["n_strings"] += len(flat)
            i = 0
            for j in jobs:
                n = len(j[8])
                j[9] = self._score(e[i:i + n]); i += n
        for r, s, nuc, lex, push, w, p_nuc, g, _, gains in jobs:
            w = w.double().cpu().numpy(); w = w / w.sum()
            mu = float((w * gains).sum()); sd = float(np.sqrt((w * (gains - mu) ** 2).sum()))
            if sd < 1e-9:
                continue
            z = np.clip((gains - mu) / sd, -self.clip_sd, self.clip_sd)
            b = torch.as_tensor(self.beta * g * z, dtype=torch.float32, device=scores.device)
            prefix_gen = [int(t) for t in input_ids[r, self.rep_L0:].tolist()]
            banned = self._repeat_banned(prefix_gen)
            if banned:
                for q, v in enumerate(push.tolist()):
                    if v in banned and b[q] > 0:
                        b[q] = 0.0; self.stats["rep_guard_hits"] += 1
            new = torch.full_like(s, float("-inf"))
            new[nuc] = s[nuc]
            new[push] = s[push] + b
            # mass-preserving tilt: shift the lexical class so its tempered mass is unchanged
            c = self.T * (torch.logsumexp(new[lex] / self.T, 0) - torch.logsumexp(s[lex] / self.T, 0))
            new[lex] = new[lex] - c
            pb0 = float(p_nuc[self.boundary[nuc]].sum())
            pb1 = float(torch.softmax(new[nuc] / self.T, dim=-1)[self.boundary[nuc]].sum())
            out[r] = new.to(out.dtype)
            self.stats["n_pushed"] += 1; self.stats["bias_abs_sum"] += float(b.abs().mean())
            self.stats["bias_abs_max"] = max(self.stats["bias_abs_max"], float(b.abs().max()))
            self.stats["p_boundary_before"] += pb0; self.stats["p_boundary_after"] += pb1
        return out

    def summary(self) -> dict:
        n = max(1, self.stats["n_pushed"])
        return {"steer_calls": self.stats["n_calls"], "steer_pushed_frac": self.stats["n_pushed"] / max(1, self.stats["n_calls"]),
                "steer_bias_abs_mean": self.stats["bias_abs_sum"] / n, "steer_bias_abs_max": self.stats["bias_abs_max"],
                "steer_p_boundary_before": self.stats["p_boundary_before"] / n, "steer_p_boundary_after": self.stats["p_boundary_after"] / n,
                "steer_rep_guard_hits": self.stats["rep_guard_hits"], "steer_embed_calls": self.stats["n_embed_calls"],
                "steer_strings": self.stats["n_strings"]}


class SentenceEndBatch(StoppingCriteria):
    """Stop a row once its new text contains a sentence boundary (one token past it, as SemStamp does).  Returns
    BoolTensor[B] as transformers 5.x expects; rows already finished are left to the generator's own bookkeeping."""

    def __init__(self, tokenizer, L0: int, eos_ids: list[int], min_new: int = 4):
        self.tokenizer, self.L0, self.eos_ids, self.min_new = tokenizer, L0, set(int(e) for e in eos_ids), min_new

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        done = []
        for r in range(input_ids.shape[0]):
            new = [int(t) for t in input_ids[r, self.L0:].tolist()]
            if len(new) < self.min_new:
                done.append(False); continue
            if new[-1] in self.eos_ids:
                done.append(True); continue
            text = self.tokenizer.decode([t for t in new if t not in self.eos_ids], skip_special_tokens=True)
            done.append(first_boundary(text) is not None)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def trim_candidate(tokenizer, new_ids: list[int], eos_ids: list[int]) -> tuple[list[int], str, bool]:
    """Keep the tokens through the first sentence boundary (dropping the overshoot token): (ids, text, boundary_found)."""
    eos = set(int(e) for e in eos_ids)
    ids = []
    for t in new_ids:
        if int(t) in eos:
            break
        ids.append(int(t))
    full = tokenizer.decode(ids, skip_special_tokens=True)
    cut = first_boundary(full)
    if cut is None:
        return ids, full.strip(), False
    for n in range(1, len(ids) + 1):                       # smallest prefix whose decoding reaches the boundary
        if len(tokenizer.decode(ids[:n], skip_special_tokens=True)) >= cut:
            return ids[:n], tokenizer.decode(ids[:n], skip_special_tokens=True).strip(), True
    return ids, full.strip(), True


# --------------------------------------------------------------------------- #
# speculative verification of an externally produced draft
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _processed_logprobs(model, ctx_ids: list[int], draft_ids: list[int], processors, warpers, L0: int) -> torch.Tensor:
    """Log-probs [T, V] of the model's (bias + temperature/top-p processed) next-token distribution at each draft position."""
    seq = torch.tensor([ctx_ids + draft_ids], device=model.device)
    logits = model(input_ids=seq, use_cache=False).logits[0].float()      # [L, V]
    out = []
    for t in range(len(draft_ids)):
        pos = len(ctx_ids) - 1 + t
        prefix = seq[:, : len(ctx_ids) + t]
        s = logits[pos:pos + 1].clone()
        for p in processors:
            s = p(prefix, s)
        for w in warpers:
            s = w(prefix, s)
        out.append(torch.log_softmax(s[0], dim=-1))
    return torch.stack(out) if out else torch.zeros((0, logits.shape[-1]), device=model.device)


@torch.no_grad()
def verify_draft(target, drafter, ctx_ids: list[int], draft_ids: list[int], processors, warpers, L0: int,
                 gen: torch.Generator) -> tuple[list[int], dict]:
    """Speculative sampling (Leviathan et al.) of a GIVEN draft: token t is kept with probability min(1, p/q); at the
    first rejection a token is drawn from norm(max(p - q, 0)) and the rest of the draft is discarded.  Both p and q
    are the models' processed distributions (same bias, temperature, top-p) so the result is the target's steered
    distribution, up to the draft being region-filtered upstream."""
    if not draft_ids:
        return [], {"n_draft": 0, "n_kept": 0, "resampled": False}
    lp = _processed_logprobs(target, ctx_ids, draft_ids, processors, warpers, L0)
    lq = _processed_logprobs(drafter, ctx_ids, draft_ids, processors, warpers, L0)
    kept: list[int] = []
    for t, x in enumerate(draft_ids):
        ratio = math.exp(min(0.0, float(lp[t, x] - lq[t, x])))
        if torch.rand(1, generator=gen, device=gen.device).item() < ratio:
            kept.append(x)
            continue
        resid = torch.clamp(lp[t].exp() - lq[t].exp(), min=0.0)
        if float(resid.sum()) <= 0:
            resid = lp[t].exp()
        y = int(torch.multinomial(resid / resid.sum(), 1, generator=gen).item())
        kept.append(y)
        return kept, {"n_draft": len(draft_ids), "n_kept": t, "resampled": True}
    return kept, {"n_draft": len(draft_ids), "n_kept": len(draft_ids), "resampled": False}


# --------------------------------------------------------------------------- #
# detection (text + key only)
# --------------------------------------------------------------------------- #

def score_text(text: str, key: Key, embedder: Embedder, lmbd: float = LMBD, margin: float = MARGIN, clip: float = 0.05,
               W: int = 16, planes: Hyperplanes | None = None, sent_emb: np.ndarray | None = None,
               seg_emb: np.ndarray | None = None) -> dict:
    """Every statistic a detector can form from the text and the key: chain z, continuous chain, segment statistic."""
    planes = planes or Hyperplanes.for_key(key, embedder.dim)
    sents, segs = split_sentences(text), split_segments(text, W)
    se = sent_emb if sent_emb is not None else embedder.encode(sents)
    ge = seg_emb if seg_emb is not None else embedder.encode(segs)
    ps, pg = planes.projections(se), planes.projections(ge)
    bins = [planes.bin_of(p) for p in ps]
    in_mask = [bins[i] in valid_mask(key, bins[i - 1], lmbd) for i in range(1, len(bins))]
    return {"n_sent": len(sents), "n_seg": len(segs), "bins": bins, "in_mask": in_mask,
            "margin_ok": [planes.margin_ok(p, margin) for p in ps],
            "z_chain": chain_z(bins, key, lmbd), "t_cont": cont_stat(ps, key, planes, lmbd, clip),
            "t_seg": seg_stat(pg, fixed_bin(key, lmbd), planes, clip) if len(pg) else None,
            # the same chain over W-word segments (bounded-chain arm with --unit seg)
            "z_segchain": chain_z([planes.bin_of(p) for p in pg], key, lmbd), "t_segcont": cont_stat(pg, key, planes, lmbd, clip)}


# --------------------------------------------------------------------------- #
# prompts of the held-out run (generation side only)
# --------------------------------------------------------------------------- #

def held_out_prompts(cfg, run: str = "runs/conf_cross"):
    """Exactly the prompts of the held-out run, so every arm answers the same questions (and keeps its key assignment)."""
    from .data.prepare import load_prepared
    want = {}
    for f in Path(run).glob("generations*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            if g["arm"] == "base":
                want[g["prompt_id"]] = g
    rows = [p for p in load_prepared(cfg.path_of("data_prepared"), partition="positive_test") if p.prompt_id in want]
    seen, uniq = set(), []
    for p in sorted(rows, key=lambda p: p.prompt_id):
        if p.prompt_id not in seen:
            seen.add(p.prompt_id); uniq.append(p)
    return uniq, want


# --------------------------------------------------------------------------- #
# best-of-N selection over fixed-length word chunks (no logit bias involved)
# --------------------------------------------------------------------------- #

class WordCountStop(StoppingCriteria):
    """Stop a row once its new text holds more than W whitespace words (one word past, so the W-th word is complete)."""

    def __init__(self, tokenizer, L0: int, eos_ids: list[int], W: int):
        self.tokenizer, self.L0, self.eos_ids, self.W = tokenizer, L0, set(int(e) for e in eos_ids), int(W)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        done = []
        for r in range(input_ids.shape[0]):
            new = [int(t) for t in input_ids[r, self.L0:].tolist()]
            if new and new[-1] in self.eos_ids:
                done.append(True); continue
            text = self.tokenizer.decode([t for t in new if t not in self.eos_ids], skip_special_tokens=True)
            done.append(len(text.split()) > self.W)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def trim_to_words(tokenizer, ids: list[int], W: int) -> tuple[list[int], str, bool]:
    """Smallest token prefix whose decoding holds W whole words; continuation pieces of the W-th word are kept so the
    chunk ends at a word boundary.  Returns (ids, text, complete) where complete is False if fewer than W words exist."""
    ids = [int(t) for t in ids]
    n_words = lambda n: len(tokenizer.decode(ids[:n], skip_special_tokens=True).split())
    if n_words(len(ids)) < W:
        return ids, tokenizer.decode(ids, skip_special_tokens=True).strip(), False
    n = next(n for n in range(1, len(ids) + 1) if n_words(n) >= W)
    while n < len(ids) and n_words(n + 1) == W:                 # the next token continues the W-th word
        n += 1
    return ids[:n], tokenizer.decode(ids[:n], skip_special_tokens=True).strip(), True


def chunk_depth(text: str, target: int, planes: Hyperplanes, embedder: Embedder) -> float:
    """How deep the chunk sits inside the target bin (what the segment statistic rewards, before clipping)."""
    proj = planes.projections(embedder.encode([text]))[0]
    return float(np.min(planes.bin_signs(target) * proj))
