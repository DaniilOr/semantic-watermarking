"""Record schemas and JSONL utilities with idempotent, resumable writes.

Raw outputs (generations, trajectories, references, attacks) are immutable
append-only JSONL files.  Every record carries a stable ``id``; a restart
re-reads the file, collects existing ids, and skips them (spec Section 12:
"Resume at record boundaries with stable RNG state and idempotent output IDs").
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .textnorm import TEXTNORM_VERSION, stable_hash


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #

def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:  # a truncated last line after a crash
                raise ValueError(f"{path}:{line_no}: corrupt JSONL line ({e}); "
                                 f"repair with `wmworld.records.repair_jsonl`") from e


def load_jsonl(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def existing_ids(path: str | Path, key: str = "id") -> set[str]:
    return {r[key] for r in read_jsonl(path) if key in r}


def repair_jsonl(path: str | Path) -> int:
    """Drop a trailing corrupt line (crash mid-write).  Returns kept line count."""
    path = Path(path)
    if not path.exists():
        return 0
    good: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
                good.append(s)
            except json.JSONDecodeError:
                break
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for s in good:
            f.write(s + "\n")
    os.replace(tmp, path)
    return len(good)


class JsonlWriter:
    """Append-only writer that skips ids already present (idempotent resume)."""

    def __init__(self, path: str | Path, id_key: str = "id"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.id_key = id_key
        self._ids = existing_ids(self.path, id_key) if self.path.exists() else set()
        self._fh = self.path.open("a", encoding="utf-8")

    def __contains__(self, rid: str) -> bool:
        return rid in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def write(self, record: dict | Any) -> bool:
        if dataclasses.is_dataclass(record):
            record = asdict(record)
        rid = record[self.id_key]
        if rid in self._ids:
            return False
        self._fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._ids.add(rid)
        return True

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(o: Any):
    """Serialize numpy scalars/arrays and paths; anything else raises as usual."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return asdict(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    # the temp name must be unique per writer: parallel shards of one run write the same manifest at the same moment,
    # and with a shared temp name one shard's rename finds its file already moved by the other
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def content_hash(obj: Any, length: int = 16) -> str:
    """Hash of the canonical JSON serialization of ``obj``."""
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return stable_hash(s, length=length)


# --------------------------------------------------------------------------- #
# Record schemas (spec Section 13)
# --------------------------------------------------------------------------- #

def output_id(stage: str, arm: str, group_id: str, key_id: str | None, seed: int, config_hash: str) -> str:
    """Idempotent output id: stable across restarts for the same work item."""
    return f"{stage}.{arm}.{group_id}.{key_id or 'nokey'}.s{seed}." + stable_hash(stage, arm, group_id, key_id, seed, config_hash, length=8)


@dataclass
class CostRecord:
    proposal_tokens: int = 0          # every generated candidate token, incl. discarded ones
    proposal_forward_calls: int = 0
    scoring_tokens: int = 0           # input tokens processed by the world scorer (keyed + neutral, wrappers included)
    scoring_forward_calls: int = 0
    scored_sequences: int = 0
    discarded_tail_tokens: int = 0    # sentence-unit proposals beyond the committed boundary
    wall_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def add(self, other: "CostRecord") -> "CostRecord":
        for f in dataclasses.fields(CostRecord):
            if f.name == "peak_memory_bytes":
                setattr(self, f.name, max(getattr(self, f.name), getattr(other, f.name)))
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self


@dataclass
class GenerationRecord:
    id: str
    stage: str
    arm: str
    group_id: str
    prompt_id: str
    domain: str
    key_id: str | None
    seed: int
    text: str                       # visible normalized text (the detector's only input)
    text_hash: str
    accepted_tokens: int            # generator tokens accepted (trace count)
    visible_tokens: int             # text-only retokenized count under the pinned tokenizer
    stop_reason: str                # eos | max_new_tokens | error | budget
    ends_mid_sentence: bool
    config_hash: str
    textnorm_version: str = TEXTNORM_VERSION
    status: str = "ok"              # ok | failed
    error: str | None = None
    cost: dict = field(default_factory=dict)
    n_decisions: int = 0
    n_active_decisions: int = 0
    n_inactive_no_signal: int = 0
    shard: int | None = None
    assigned_key_id: str | None = None   # the schedule's key for this prompt (used for paired comparisons; None-key arms too)
    final_score_trace: float | None = None  # F_k(final) as seen by the generator (diagnostic; detector recomputes from text)
    arm_config: dict = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    """Private research diagnostics; never read by the detector."""
    id: str                          # f"{output_id}.d{decision}"
    output_id: str
    decision: int
    accepted_prefix_hash: str
    accepted_len: int
    unit: str
    active: bool
    scale_status: str                # ok | inactive_no_signal | inactive_schedule | uniform_arm
    sigma: float | None
    length_bin: str | None
    candidate_token_ids: list[list[int]]
    candidate_texts: list[str]
    candidate_scores: list[float | None]      # F_k(y||s_j) (global) or F_k(s_j) (local)
    prefix_score: float | None                 # F_k(y) before the decision
    gains: list[float | None]
    incremental_contrast: list[float | None]   # c_j = [ell_k - ell_0](s_j | y): raw sum contrast of the unit given prefix
    probabilities: list[float]
    chosen: int
    chosen_ended: bool
    cost: dict


@dataclass
class ReferenceRecord:
    id: str
    stage: str
    group_id: str
    prompt_id: str
    domain: str
    text: str
    text_hash: str
    visible_tokens: int
    source: str
    textnorm_version: str = TEXTNORM_VERSION


@dataclass
class AttackRecord:
    id: str
    parent_output_id: str
    stage: str
    attack: str
    params: dict
    seed: int
    text: str
    text_hash: str
    visible_tokens: int
    donor_ids: list[str]
    status: str                      # ok | failed | inapplicable
    error: str | None
    raw_response: str | None
    metrics: dict
    textnorm_version: str = TEXTNORM_VERSION


@dataclass
class DetectionRecord:
    """Detection output: contains no prompt, label, arm or trajectory information."""
    id: str
    text_hash: str
    key_id: str
    scorer_id: str
    calibration_id: str | None
    score: float | None
    visible_tokens: int
    p_generator: float | None
    p_human: float | None
    p_final: float | None
    decision: str                    # detected | not_detected | abstain_short | abstain_overflow | error
    status: str
    components: dict = field(default_factory=dict)
