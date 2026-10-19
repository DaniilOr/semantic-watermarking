"""Frozen deduplication procedure (spec Section 7).

1. Exact duplicates: normalized prompt core (lowercase, whitespace-collapsed).
2. Near-duplicates: word 5-gram MinHash (128 permutations, 16 bands x 8 rows)
   followed by exact Jaccard of the 5-gram sets >= 0.85 on candidate pairs.
Groups are the connected components; each group is one source group.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

DEDUP_VERSION = "dedup_v1_minhash128_jaccard085"
N_PERM = 128
BANDS = 16
ROWS = N_PERM // BANDS
JACCARD_THRESHOLD = 0.85
_PRIME = 4294967311  # smallest prime > 2^32; a*x < 2^64 for a, x < 2^32 so uint64 arithmetic is exact


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, n: int = 5) -> set[int]:
    """Word 5-gram shingles hashed to 32 bits (blake2b)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        words = words + ["<pad>"] * (n - len(words))
    out = set()
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        out.add(int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "little"))
    return out


class MinHasher:
    def __init__(self, seed: int = 20260911):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 32, size=N_PERM, dtype=np.int64).astype(np.uint64)
        self.b = rng.integers(0, 1 << 32, size=N_PERM, dtype=np.int64).astype(np.uint64)

    def signature(self, sh: set[int]) -> np.ndarray:
        if not sh:
            return np.full(N_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        ax = (self.a[:, None] * x[None, :]) % np.uint64(_PRIME)          # (128, S), each < 2^33
        h = (ax + self.b[:, None]) % np.uint64(_PRIME)
        return h.min(axis=1).astype(np.uint64)


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def group_duplicates(prompt_cores: list[str], doc_texts: list[str], seed: int = 20260911) -> list[int]:
    """Return a group index per item (0..G-1) under the frozen procedure."""
    n = len(doc_texts)
    parent = list(range(n))

    def union(i, j):
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # exact duplicates on the prompt core
    seen: dict[str, int] = {}
    for i, p in enumerate(prompt_cores):
        k = normalized_key(p)
        if k in seen:
            union(i, seen[k])
        else:
            seen[k] = i
    # MinHash LSH on document shingles
    hasher = MinHasher(seed)
    shingle_sets = [shingles(t) for t in doc_texts]
    sigs = np.stack([hasher.signature(s) for s in shingle_sets]) if n else np.zeros((0, N_PERM), dtype=np.uint64)
    candidates: set[tuple[int, int]] = set()
    for b in range(BANDS):
        band = sigs[:, b * ROWS:(b + 1) * ROWS]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n):
            buckets.setdefault(band[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) > 1:
                for a in range(len(members)):
                    for c in range(a + 1, len(members)):
                        candidates.add((members[a], members[c]))
    for i, j in candidates:
        si, sj = shingle_sets[i], shingle_sets[j]
        inter = len(si & sj)
        union_size = len(si) + len(sj) - inter
        if union_size and inter / union_size >= JACCARD_THRESHOLD:
            union(i, j)
    roots = [_find(parent, i) for i in range(n)]
    remap: dict[int, int] = {}
    out = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return out
