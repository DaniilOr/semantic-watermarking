"""Benchmark with SemStamp's datasets and protocol (amendment 22): RealNews, BookSum, Reddit-TIFU, plus MBPP.

SemStamp (Hou et al., NAACL 2024) evaluates on the RealNews subset of C4, BookSum and Reddit-TIFU with 32-token
prompts, 200-token generations and 1,000 random samples; this builds the same for our instruct generators (the excerpt
is wrapped in a continuation instruction, as in data/ob_bench).  Tokens = the Llama-3.2 tokenizer.  Per text dataset:
  test  1,000 documents: prompt (first 32 tokens, wrapped) and the human continuation (next 200 tokens)
  null  2,000 further documents: human continuations only (1% and 0.1% FPR thresholds)
Sources:
  realnews  allenai/c4 realnewslike validation (data/raw/c4_realnewslike); rows of any prepared partition, of the
            bin-adapter training prompts and of data/ob_bench are excluded (id, URL, exact text)
  booksum   kmfoda/booksum (rev c62321036e56, field "chapter"): test prompts from test.csv then dev.csv (test.csv has
            only 761 distinct chapters), nulls from train.csv (BookSum splits by book, so nulls never share a book)
  tifu      ctr4si/reddit_tifu (rev 68cc1f53a9e3, Kim et al. 2019), field "selftext" cut at the first "TL;DR"
  mbpp      data/raw/mbpp (all 974 problems, train+test+validation+prompt): prompt = task + its tests (the template of
            wmworld/data/sources.py); human null = the 974 reference solutions (only 974 exist, so a 0.1% FPR
            threshold cannot be resolved for MBPP)
Documents are taken in a fixed hash order (salt "sp_bench") among those with >= 232 tokens.
Writes data/sp_bench/prompts.jsonl, data/sp_bench/report.json and the human-null records runs/sp_bench_null/generations.jsonl.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.data.prepare import load_prepared  # noqa: E402
from wmworld.data.sources import PROMPT_TEMPLATES  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import normalize_text, stable_hash, text_hash  # noqa: E402

RAW = Path("data/raw/sp_bench")
N_TEST, N_NULL, P_TOK, C_TOK = 1000, 2000, 32, 200
WRAP = {
    "realnews": "Continue the following news article. Do not repeat the given text; write what comes next.\n\n{excerpt}",
    "booksum": "Continue the following book chapter. Do not repeat the given text; write what comes next.\n\n{excerpt}",
    "tifu": "Continue the following Reddit post. Do not repeat the given text; write what comes next.\n\n{excerpt}",
}
TLDR = re.compile(r"\bt\s*l\s*;?\s*d\s*r\b", re.I)


def docs_realnews():
    path = Path("data/raw/c4_realnewslike/realnewslike/c4-validation.00000-of-00001.json.gz")
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                d = json.loads(line); out.append((f"c4rnl:val:{i}", d.get("text", ""), d.get("url")))
    return out


def docs_booksum(fname):
    local = hf_hub_download("kmfoda/booksum", fname, repo_type="dataset", revision="c62321036e56", local_dir=RAW / "booksum")
    df = pd.read_csv(local)
    tag = Path(fname).stem
    return [(f"booksum:{tag}:{row['bid'] if 'bid' in df.columns else i}:{i}", str(row["chapter"]) if isinstance(row["chapter"], str) else "", None)
            for i, row in df.iterrows()]


def docs_tifu():
    p = hf_hub_download("ctr4si/reddit_tifu", "data/tifu_all_tokenized_and_filtered.json.gz", repo_type="dataset",
                        revision="68cc1f53a9e3", local_dir=RAW / "tifu")
    out = []
    with gzip.open(p, "rt") as f:
        for ln in f:
            r = json.loads(ln)
            t = r.get("selftext") or ""
            m = TLDR.search(t)
            out.append((f"tifu:{r['id']}", t[: m.start()] if m else t, r.get("url")))
    return out


def take(name, ordered, tok, excl=lambda d: False, is_test=None, n_test=N_TEST, n_null=N_NULL):
    """First n_test eligible documents with is_test(d) become test rows, then n_null documents with not is_test(d)."""
    seen, test, null, n_short, n_excl = set(), [], [], 0, 0
    for d in ordered:
        did, text, url = d
        want_test = len(test) < n_test
        if is_test is not None and want_test != is_test(d):
            continue
        if not want_test and len(null) >= n_null:
            break
        if excl(d):
            n_excl += 1; continue
        text = normalize_text(text).strip()
        h = text_hash(text)
        if h in seen:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < P_TOK + C_TOK:
            n_short += 1; continue
        seen.add(h)
        (test if want_test else null).append((did, tok.decode(ids[:P_TOK]), tok.decode(ids[P_TOK:P_TOK + C_TOK])))
    if len(test) < n_test or len(null) < n_null:
        raise SystemExit(f"{name}: only {len(test)} test / {len(null)} null eligible documents")
    rows = []
    for split, items in (("test", test), ("null", null)):
        for did, excerpt, cont in items:
            rows.append({"prompt_id": f"sp:{did}", "group_id": "sp_" + stable_hash(name, did, length=12), "domain": f"sp_{name}",
                         "split": split, "prompt": WRAP[name].format(excerpt=excerpt), "excerpt": excerpt, "human_continuation": cont})
    return rows, {"excluded_overlap": n_excl, "too_short_seen": n_short}


def main():
    cfg = load_config("configs/pilot.yaml")
    tok = AutoTokenizer.from_pretrained(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"])
    rows, report = [], {"recipe": {"prompt_tokens": P_TOK, "continuation_tokens": C_TOK, "test": N_TEST, "null": N_NULL}, "datasets": {}}
    order = lambda docs, name: sorted(docs, key=lambda d: stable_hash("sp_bench", name, d[0]))
    # realnews: exclude prepared / training / ob_bench rows (id, url, text)
    prepared = load_prepared(cfg.path_of("data_prepared"))
    excl_ids = {p.source_id for p in prepared if p.domain == "c4_realnewslike"}
    excl_ids |= {r["source_id"] for r in map(json.loads, open("data/ss_train/prompts.jsonl")) if r["domain"] == "c4_realnewslike"}
    excl_ids |= {r["prompt_id"].split(":", 1)[1] for r in map(json.loads, open("data/ob_bench/prompts.jsonl")) if r["domain"] == "ob_c4"}
    docs = docs_realnews()
    bad_url = {u for d, _, u in docs if d in excl_ids}; bad_hash = {text_hash(normalize_text(t)) for d, t, _ in docs if d in excl_ids}
    r, st = take("realnews", order(docs, "realnews"), tok,
                 excl=lambda d: d[0] in excl_ids or d[2] in bad_url or text_hash(normalize_text(d[1])) in bad_hash)
    rows += r; report["datasets"]["realnews"] = st | {"docs": len(docs)}
    # booksum: test from test.csv then dev.csv, nulls from train.csv
    bt, bd, btr = docs_booksum("test.csv"), docs_booksum("dev.csv"), docs_booksum("train.csv")
    r, st = take("booksum", order(bt, "booksum") + order(bd, "booksum") + order(btr, "booksum"), tok,
                 is_test=lambda d: not d[0].startswith("booksum:train:"))
    rows += r; report["datasets"]["booksum"] = st | {"docs": {"test.csv": len(bt), "dev.csv": len(bd), "train.csv": len(btr)}}
    # tifu
    docs = docs_tifu()
    r, st = take("tifu", order(docs, "tifu"), tok)
    rows += r; report["datasets"]["tifu"] = st | {"docs": len(docs)}
    # mbpp: all problems as test prompts, reference solutions as the human null
    mb = pd.concat([pd.read_parquet(f) for f in sorted(Path("data/raw/mbpp").glob("full-*.parquet"))]).drop_duplicates("task_id")
    for _, p in mb.sort_values("task_id").iterrows():
        tid = int(p["task_id"])
        prompt = PROMPT_TEMPLATES["mbpp"].format(text=p["text"], tests="\n".join(p["test_list"]))
        rows.append({"prompt_id": f"sp:mbpp:{tid}", "group_id": f"sp_mbpp_{tid}", "domain": "sp_mbpp", "split": "test",
                     "prompt": prompt, "excerpt": p["text"], "human_continuation": p["code"]})
        rows.append({"prompt_id": f"sp:mbpp:null:{tid}", "group_id": f"sp_mbpp_{tid}", "domain": "sp_mbpp", "split": "null",
                     "prompt": prompt, "excerpt": p["text"], "human_continuation": p["code"]})
    report["datasets"]["mbpp"] = {"problems": int(len(mb))}
    for name in ("realnews", "booksum", "tifu", "mbpp"):
        report["datasets"][name]["taken"] = dict(Counter(r["split"] for r in rows if r["domain"] == f"sp_{name}"))
        print(name, json.dumps(report["datasets"][name]), flush=True)
    out = Path("data/sp_bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "report.json", report)
    nul = Path("runs/sp_bench_null"); nul.mkdir(parents=True, exist_ok=True)
    with (nul / "generations.jsonl").open("w") as f:
        for r in rows:
            if r["split"] != "null":
                continue
            text = normalize_text(r["human_continuation"])
            f.write(json.dumps({"id": f"human.{r['prompt_id']}", "stage": nul.name, "arm": "base", "group_id": r["group_id"],
                                "prompt_id": r["prompt_id"], "domain": r["domain"], "key_id": None, "seed": 0, "text": text,
                                "text_hash": text_hash(text), "visible_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                                "status": "ok"}, ensure_ascii=False) + "\n")
    write_json(nul / "manifest.json", {"partition": "sp_bench_null", "note": "human continuations of the null documents (MBPP: reference solutions)"})
    print(f"wrote data/sp_bench/prompts.jsonl ({len(rows)} rows) and runs/sp_bench_null/generations.jsonl")


if __name__ == "__main__":
    main()
