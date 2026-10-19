"""Test benchmark after OpenStamp (arXiv 2608.27899): C4 RealNewsLike, ArXiv, BookSum, Wikipedia.

The paper's recipe (Sec. 4.2): 50-token prompts, 200-token completions, 500 completions; unwatermarked negatives =
the document's own next 200 tokens (human text).  It gives no construction details for ArXiv / BookSum / Wikipedia,
so the same recipe is applied to each.  Tokens = the generator's (Llama-3.2) tokenizer.  Per dataset:
  test  500 documents: prompt (first 50 tokens, wrapped in a continuation instruction for the instruct generator)
        and the human continuation (next 200 tokens) = the paper's negatives
  null  2,000 further documents: human continuations only, so 1% and 0.1% FPR thresholds are resolvable
Sources (pinned revisions, downloaded to data/raw/ob_bench/):
  c4        allenai/c4 realnewslike validation (already in data/raw/c4_realnewslike); rows used by any prepared
            partition or by the bin-adapter training prompts are excluded, as are their URLs and exact duplicates
  arxiv     ccdv/arxiv-summarization, config "document", test split, field "article" (Cohan et al. 2018)
  booksum   kmfoda/booksum, field "chapter" (Kryscinski et al. 2022): test prompts from test.csv (761 distinct
            chapters); nulls from dev.csv then train.csv (BookSum splits by book, so nulls never share a book)
  wikipedia wikimedia/wikipedia 20231101.en, shard train-00000-of-00041, field "text"
Documents are taken in a fixed hash order (salt "ob_bench") among those with >= 250 tokens.
Writes data/ob_bench/prompts.jsonl and data/ob_bench/report.json.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.data.prepare import load_prepared  # noqa: E402
from wmworld.config import load_config  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import normalize_text, stable_hash, text_hash  # noqa: E402

RAW = Path("data/raw/ob_bench")
SOURCES = {
    "arxiv": ("ccdv/arxiv-summarization", "240aaf1a969b", "document/test-00000-of-00001.parquet", "article"),
    "booksum": ("kmfoda/booksum", "c62321036e56", "test.csv", "chapter"),
    "wikipedia": ("wikimedia/wikipedia", "b04c8d1ceb2f", "20231101.en/train-00000-of-00041.parquet", "text"),
}
WRAP = {
    "c4": "Continue the following news article. Do not repeat the given text; write what comes next.\n\n{excerpt}",
    "arxiv": "Continue the following scientific article. Do not repeat the given text; write what comes next.\n\n{excerpt}",
    "booksum": "Continue the following book chapter. Do not repeat the given text; write what comes next.\n\n{excerpt}",
    "wikipedia": "Continue the following Wikipedia article. Do not repeat the given text; write what comes next.\n\n{excerpt}",
}
N_TEST, N_NULL, P_TOK, C_TOK = 500, 2000, 50, 200


def full_sha(repo: str, short: str) -> str:
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(repo).sha
    assert sha.startswith(short), f"{repo}: hub revision {sha} != pinned {short}"
    return sha


def load_docs(name: str, fname_override: str | None = None) -> tuple[list[tuple[str, str]], dict]:
    """(doc_id, text, url) triples and provenance."""
    if name == "c4":
        path = Path("data/raw/c4_realnewslike/realnewslike/c4-validation.00000-of-00001.json.gz")
        docs = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    d = json.loads(line); docs.append((f"c4rnl:val:{i}", d.get("text", ""), d.get("url")))
        return docs, {"source": str(path)}
    repo, short, fname, field = SOURCES[name]
    fname = fname_override or fname
    sha = full_sha(repo, short)
    local = hf_hub_download(repo, fname, repo_type="dataset", revision=sha, local_dir=RAW / name)
    df = pd.read_parquet(local) if fname.endswith(".parquet") else pd.read_csv(local)
    key = "id" if "id" in df.columns else ("bid" if "bid" in df.columns else None)
    tag = Path(fname).stem.split("-")[0]
    docs = [(f"{name}:{tag}:{(row[key] if key else i)}:{i}", str(row[field] if isinstance(row[field], str) else ""), None) for i, row in df.iterrows()]
    return docs, {"repo": repo, "revision": sha, "file": fname, "field": field, "rows": len(df)}


def main() -> None:
    cfg = load_config("configs/pilot.yaml")
    tok = AutoTokenizer.from_pretrained(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"])
    prepared = load_prepared(cfg.path_of("data_prepared"))
    train_rows = [json.loads(l) for l in open("data/ss_train/prompts.jsonl")]
    excl_ids = {p.source_id for p in prepared if p.domain == "c4_realnewslike"} | {r["source_id"] for r in train_rows if r["domain"] == "c4_realnewslike"}
    rows, report = [], {"recipe": {"prompt_tokens": P_TOK, "continuation_tokens": C_TOK, "test": N_TEST, "null": N_NULL}, "datasets": {}}
    for name in ("c4", "arxiv", "booksum", "wikipedia"):
        docs, prov = load_docs(name)
        excl_urls, excl_hash = set(), set()
        if name == "c4":
            for did, text, url in docs:
                if did in excl_ids:
                    excl_urls.add(url); excl_hash.add(text_hash(normalize_text(text)))
        seen, cand, n_short, n_excl = set(), [], 0, 0
        ordered = sorted(docs, key=lambda d: stable_hash("ob_bench", name, d[0]))
        if name == "booksum":                                            # test prompts from test.csv, nulls from other books
            extra = []
            for f in ("dev.csv", "train.csv"):
                d2, p2 = load_docs(name, f); extra += sorted(d2, key=lambda d: stable_hash("ob_bench", name, d[0])); prov[f] = p2
        for did, text, url in ordered + (extra if name == "booksum" else []):
            if name == "booksum" and len(cand) < N_TEST and not did.startswith("booksum:test:"):
                break                                                    # fewer than N_TEST test chapters: fails below
            if name == "booksum" and len(cand) >= N_TEST and did.startswith("booksum:test:"):
                continue                                                 # nulls only from other books (dev/train)
            text = normalize_text(text).strip()
            h = text_hash(text)
            if name == "c4" and (did in excl_ids or url in excl_urls or h in excl_hash):
                n_excl += 1; continue
            if h in seen:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) < P_TOK + C_TOK:
                n_short += 1; continue
            seen.add(h)
            excerpt = tok.decode(ids[:P_TOK]); cont = tok.decode(ids[P_TOK:P_TOK + C_TOK])
            cand.append((did, excerpt, cont))
            if len(cand) >= N_TEST + N_NULL:
                break
        if len(cand) < N_TEST + N_NULL:
            raise SystemExit(f"{name}: only {len(cand)} eligible documents")
        for k, (did, excerpt, cont) in enumerate(cand):
            rows.append({"prompt_id": f"ob:{did}", "group_id": "ob_" + stable_hash(name, did, length=12), "domain": f"ob_{name}",
                         "split": "test" if k < N_TEST else "null", "prompt": WRAP[name].format(excerpt=excerpt),
                         "excerpt": excerpt, "human_continuation": cont})
        report["datasets"][name] = prov | {"docs": len(docs), "excluded_overlap": n_excl, "too_short_seen": n_short,
                                           "taken": dict(Counter(r["split"] for r in rows if r["domain"] == f"ob_{name}"))}
        print(name, json.dumps(report["datasets"][name]), flush=True)
    out = Path("data/ob_bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ex = {r["prompt_id"].split(":", 1)[1] for r in rows if r["domain"] == "ob_c4"}
    report["c4_overlap_with_prepared_or_training_rows"] = len(ex & excl_ids)
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
