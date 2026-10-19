"""Text normalization policy (spec Section 2).

Policy version 1:
  * text is UTF-8 (str in Python); invalid surrogates are replaced;
  * line endings are normalized to LF exactly once (CRLF and CR -> LF);
  * all other whitespace is preserved verbatim (no strip, no collapsing);
  * no Unicode normalization (NFC/NFKC) is applied.

The identical function is applied during candidate scoring, saved-output
serialization, attacks, and detection.  Its version is recorded in every
artifact that depends on it.
"""

from __future__ import annotations

import hashlib

TEXTNORM_VERSION = "textnorm_v1_lf"


def normalize_text(text: str) -> str:
    """Apply the versioned normalization policy.  Idempotent."""
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text).__name__}")
    # Replace lone surrogates so the text is valid UTF-8 when serialized.
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def text_hash(text: str) -> str:
    """Stable content hash of the *normalized* text (sha256 hex, 16 chars)."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def stable_hash(*parts: object, length: int = 16) -> str:
    """Deterministic hash of a tuple of printable parts (used for ids/seeds)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def seed_from(*parts: object) -> int:
    """Deterministic 63-bit integer seed derived from printable parts."""
    return int(stable_hash(*parts, length=16), 16) & ((1 << 63) - 1)
