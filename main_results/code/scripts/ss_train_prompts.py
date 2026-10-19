"""Training prompts for the bin adapters (amendment 11): fresh prompts, disjoint from every prepared partition.

Same loaders, filters and near-duplicate grouping as ``prepare-data`` (wmworld/data/prepare.py, data/dedup.py).  A
dedup group is excluded when any of its members is a prepared prompt (any partition: smoke, scale_fit, dev,
null_calibration, null_test, positive_test, donor_pool); for writingprompts only the train split is used.  A
deterministic sample (salt "ss_train") of --train + --val prompts per domain is written to data/ss_train/prompts.jsonl
with a report proving zero overlap (prepared groups hit, exact prompt hashes shared with the prepared partitions).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.cli import _tokenizer_funcs  # noqa: E402
from wmworld.config import load_config  # noqa: E402
from wmworld.data.dedup import group_duplicates  # noqa: E402
from wmworld.data.prepare import load_domain_examples, load_prepared  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import normalize_text, stable_hash, text_hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--train", type=int, default=400, help="train prompts per domain")
    ap.add_argument("--val", type=int, default=50, help="validation prompts per domain")
    ap.add_argument("--out", default="data/ss_train/prompts.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    _, encode, decode = _tokenizer_funcs(cfg)
    prepared = load_prepared(cfg.path_of("data_prepared"))
    prep_ids = {(p.domain, p.source_id) for p in prepared}
    prep_hashes = {p.prompt_hash for p in prepared}
    rows, report = [], {"n_prepared": len(prepared), "domains": {}}
    for domain in cfg["data"]["domains"]:
        exs = load_domain_examples(domain, cfg.path_of("data_raw"), encode, decode,
                                   int(cfg["data"].get("writingprompts_candidate_cap", 6000)), cfg.seed)
        kept = [e for e in exs if not e.meta.get("filter_reasons")]
        gidx = group_duplicates([e.prompt_core for e in kept], [e.doc_text for e in kept], seed=cfg.seed)
        groups: dict[int, list] = {}
        for e, g in zip(kept, gidx):
            groups.setdefault(g, []).append(e)
        hit = {g for g, mem in groups.items() if any((domain, e.source_id) in prep_ids for e in mem)}
        free = []
        for g, mem in groups.items():
            if g in hit:
                continue
            mem.sort(key=lambda e: stable_hash("rep", e.source_id))
            e = mem[0]
            if domain == "writingprompts" and e.source_split != "train":
                continue
            if text_hash(e.prompt) in prep_hashes:
                continue
            free.append(e)
        free.sort(key=lambda e: stable_hash("ss_train", cfg.seed, domain, e.source_id))
        need = a.train + a.val
        if len(free) < need:
            raise SystemExit(f"{domain}: only {len(free)} free groups for {need} prompts")
        for i, e in enumerate(free[:need]):
            rows.append({"prompt_id": f"sstrain:{domain}:{e.source_id}", "group_id": "gt_" + stable_hash(domain, e.source_id, length=12),
                         "source_id": e.source_id, "domain": domain, "split": "train" if i < a.train else "val",
                         "prompt": normalize_text(e.prompt), "prompt_hash": text_hash(e.prompt)})
        report["domains"][domain] = {"n_loaded": len(exs), "n_after_filters": len(kept), "n_groups": len(groups),
                                     "groups_with_a_prepared_member": len(hit), "free_groups": len(free),
                                     "taken": dict(Counter(r["split"] for r in rows if r["domain"] == domain))}
    # proofs of disjointness
    report["overlap_prepared_source_ids"] = sum((r["domain"], r["source_id"]) in prep_ids for r in rows)
    report["overlap_prepared_prompt_hashes"] = sum(r["prompt_hash"] in prep_hashes for r in rows)
    report["n_rows"] = len(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out.with_name("prompts_report.json"), report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
