"""``prepare-data``: fixed filters -> dedup -> grouped, disjoint partitions -> manifest.

Partitions (per domain budgets from the config): smoke, scale_fit, dev,
null_calibration, null_test, positive_test, plus ``donor_pool`` (dilution
donors, spec Section 10) and an ``ablation`` subset marked inside ``dev``.
WritingPrompts honours source splits (train -> smoke/scale_fit/dev/donor,
valid -> null_calibration, test -> null_test/positive_test).  Dolly and C4
realnewslike get deterministic custom partitions (not official benchmark splits).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..records import content_hash, write_json
from ..textnorm import TEXTNORM_VERSION, normalize_text, seed_from, stable_hash, text_hash
from .dedup import DEDUP_VERSION, group_duplicates
from .sources import (PREPROCESSING_VERSION, SOURCES, SourceExample, iter_c4_realnewslike, iter_dolly,
                      iter_writingprompts)

PARTITION_ORDER = ["smoke", "scale_fit", "dev", "null_calibration", "null_test", "positive_test", "donor_pool", "code"]
WP_SPLIT_FOR_PARTITION = {"smoke": "train", "scale_fit": "train", "dev": "train", "donor_pool": "train",
                          "null_calibration": "valid", "null_test": "test", "positive_test": "test", "code": "train"}


@dataclass
class PreparedExample:
    prompt_id: str
    group_id: str
    source_id: str
    domain: str
    source_split: str
    partition: str
    ablation_subset: bool
    prompt: str
    prompt_hash: str
    reference: str | None
    reference_hash: str | None
    reference_tokens: int | None
    reference_eligible: bool
    meta: dict
    preprocessing_version: str = PREPROCESSING_VERSION
    textnorm_version: str = TEXTNORM_VERSION


def _cap_reference(ref: str | None, encode, decode, cap_tokens: int, min_tokens: int) -> tuple[str | None, int | None, bool]:
    if not ref:
        return None, None, False
    ref = normalize_text(ref)
    toks = encode(ref)
    if len(toks) > cap_tokens:
        ref = decode(toks[:cap_tokens])
        toks = encode(ref)
    return ref, len(toks), len(toks) >= min_tokens


def load_domain_examples(domain: str, raw_root: Path, encode, decode, candidate_cap: int, seed: int) -> list[SourceExample]:
    """Load a deterministic candidate pool per domain (documented sub-sampling for the large WP train split)."""
    if domain == "writingprompts":
        root = raw_root / "writingprompts" / "writingPrompts"
        out: list[SourceExample] = []
        for split in ("train", "valid", "test"):
            exs = list(iter_writingprompts(root, split))
            # deterministic sub-sample of the candidate pool by hash order (train is ~272k lines)
            exs.sort(key=lambda e: stable_hash("wp_pool", seed, e.source_id))
            out.extend(exs[:candidate_cap])
        return out
    if domain == "dolly":
        return list(iter_dolly(raw_root / "dolly" / "databricks-dolly-15k.jsonl"))
    if domain == "c4_realnewslike":
        return list(iter_c4_realnewslike(raw_root / "c4_realnewslike" / "realnewslike" / "c4-validation.00000-of-00001.json.gz",
                                         encode, tokenizer_decode=decode))
    if domain == "mbpp":
        from .sources import iter_mbpp
        return list(iter_mbpp(raw_root / "mbpp"))
    raise ValueError(f"unknown domain {domain}")


def prepare_domain(domain: str, examples: list[SourceExample], budgets: dict[str, int], encode, decode, seed: int,
                   reference_cap_tokens: int = 512, min_reference_tokens: int = 32,
                   allow_shortfall: bool = False) -> tuple[list[PreparedExample], dict]:
    report: dict = {"domain": domain, "n_loaded": len(examples)}
    kept = [e for e in examples if not e.meta.get("filter_reasons")]
    reasons: dict[str, int] = {}
    for e in examples:
        for r in e.meta.get("filter_reasons", []):
            reasons[r] = reasons.get(r, 0) + 1
    report["filter_reasons"] = reasons
    report["n_after_filters"] = len(kept)
    # dedup -> groups
    gidx = group_duplicates([e.prompt_core for e in kept], [e.doc_text for e in kept], seed=seed)
    groups: dict[int, list[SourceExample]] = {}
    for e, g in zip(kept, gidx):
        groups.setdefault(g, []).append(e)
    report["n_groups"] = len(groups)
    report["n_multi_member_groups"] = sum(1 for g in groups.values() if len(g) > 1)
    # one representative per group (smallest source id by hash for determinism)
    reps = []
    for g, members in groups.items():
        members.sort(key=lambda e: stable_hash("rep", e.source_id))
        reps.append(members[0])
    # deterministic shuffle
    rng = np.random.default_rng(seed_from("partition", domain, seed))
    order = np.arange(len(reps))
    rng.shuffle(order)
    reps = [reps[i] for i in order]
    # assign partitions with split preference
    assigned: dict[str, list[SourceExample]] = {p: [] for p in PARTITION_ORDER}
    used: set[str] = set()
    shortfall: dict[str, int] = {}
    for part in PARTITION_ORDER:
        need = int(budgets.get(part, 0))
        if need <= 0:
            continue
        if domain == "writingprompts":
            want_split = WP_SPLIT_FOR_PARTITION[part]
            pool = [e for e in reps if e.source_split == want_split and e.source_id not in used]
        else:
            pool = [e for e in reps if e.source_id not in used]
        take = pool[:need]
        if len(take) < need:
            shortfall[part] = need - len(take)
        for e in take:
            used.add(e.source_id)
        assigned[part] = take
    report["shortfall"] = shortfall
    if shortfall and not allow_shortfall:
        raise RuntimeError(f"{domain}: not enough eligible groups for {shortfall}; "
                           f"use an explicitly versioned smaller profile (allow_shortfall) instead of broadening the task mix")
    # ablation subset: first budgets['ablation'] dev prompts by hash
    abl_n = int(budgets.get("ablation", 0))
    dev_sorted = sorted(assigned["dev"], key=lambda e: stable_hash("ablation", seed, e.source_id))
    abl_ids = {e.source_id for e in dev_sorted[:abl_n]}
    prepared: list[PreparedExample] = []
    ref_elig = 0
    for part, exs in assigned.items():
        for e in exs:
            ref, ntok, elig = _cap_reference(e.reference, encode, decode, reference_cap_tokens, min_reference_tokens)
            ref_elig += int(elig)
            gid = "g_" + stable_hash(domain, e.source_id, length=12)
            prepared.append(PreparedExample(
                prompt_id=f"{domain}:{e.source_id}", group_id=gid, source_id=e.source_id, domain=domain,
                source_split=e.source_split, partition=part, ablation_subset=e.source_id in abl_ids,
                prompt=normalize_text(e.prompt), prompt_hash=text_hash(e.prompt),
                reference=ref, reference_hash=text_hash(ref) if ref else None, reference_tokens=ntok,
                reference_eligible=elig, meta={k: v for k, v in e.meta.items() if k != "filter_reasons"},
            ))
    report["n_assigned"] = {p: len(v) for p, v in assigned.items()}
    report["reference_eligible_total"] = ref_elig
    report["reference_eligible_by_partition"] = {p: sum(1 for x in prepared if x.partition == p and x.reference_eligible)
                                                 for p in PARTITION_ORDER}
    return prepared, report


def write_prepared(out_dir: Path, prepared: list[PreparedExample], reports: list[dict], config_summary: dict,
                   backend_info: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_domain: dict[str, list[PreparedExample]] = {}
    for p in prepared:
        by_domain.setdefault(p.domain, []).append(p)
    files = {}
    for domain, rows in by_domain.items():
        path = out_dir / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda r: (r.partition, r.prompt_id)):
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        files[domain] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
    manifest = {
        "preprocessing_version": PREPROCESSING_VERSION, "dedup_version": DEDUP_VERSION, "textnorm_version": TEXTNORM_VERSION,
        "sources": SOURCES, "files": files, "reports": reports, "config": config_summary, "tokenizer": backend_info,
        "rows": [{"prompt_id": p.prompt_id, "group_id": p.group_id, "domain": p.domain, "source_split": p.source_split,
                  "partition": p.partition, "ablation_subset": p.ablation_subset, "prompt_hash": p.prompt_hash,
                  "reference_hash": p.reference_hash, "reference_tokens": p.reference_tokens,
                  "reference_eligible": p.reference_eligible} for p in prepared],
    }
    manifest["content_hash"] = content_hash({k: v for k, v in manifest.items() if k != "rows"} | {"rows": manifest["rows"]})
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_prepared(prepared_dir: Path, partition: str | None = None, domains: list[str] | None = None) -> list[PreparedExample]:
    out = []
    for path in sorted(prepared_dir.glob("*.jsonl")):
        if domains and path.stem not in domains:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if partition and d["partition"] != partition:
                    continue
                out.append(PreparedExample(**d))
    return out
