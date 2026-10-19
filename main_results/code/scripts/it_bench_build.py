"""Instruction-task benchmark for the chat models (2026-10-02): Dolly, WritingPrompts, IFEval, GSM8K.

Per dataset, test prompts (the chat model's instruction) and human texts for the false-alarm thresholds (null split),
in the format of data/sp_bench (prompt_id, group_id, domain, split, prompt, excerpt, human_continuation, + extras):
  it_dolly           databricks-dolly-15k, categories open_qa / general_qa / brainstorming / creative_writing;
                     1,000 test prompts (instruction, plus its context when present); null = human answers of >= 100
                     words in those categories from other prompts
  it_writingprompts  WritingPrompts test split; 1,000 test prompts, null = 2,000 human stories of other prompts
  it_ifeval          all 541 IFEval prompts (instruction ids and kwargs kept for scoring); no human answers exist
  it_gsm8k           all 1,319 GSM8K test problems (gold answer kept); null = 2,000 human train solutions
Prompts are taken in a fixed hash order (salt "it_bench").  Human texts are cut to 512 tokens (Llama-3.2 tokenizer), the
answer cap of the generation runs.  Writes data/it_bench/prompts.jsonl, data/it_bench/report.json and
runs/it_bench_null/generations.jsonl (human texts as arm "base", like runs/sp_bench_null).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.data.sources import PROMPT_TEMPLATES  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import normalize_text, stable_hash, text_hash  # noqa: E402

CAP = 512
DOLLY_CATS = ("open_qa", "general_qa", "brainstorming", "creative_writing")
GSM_PROMPT = ("Solve the following math problem. Reason step by step, then give the final answer on the last line "
              "as: The answer is <number>.\n\nProblem: {question}")


def order(items, name):
    return sorted(items, key=lambda x: stable_hash("it_bench", name, x[0]))


def cut(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= CAP else tok.decode(ids[:CAP])


def row(name, pid, split, prompt, human, **extra):
    return {"prompt_id": f"it:{name}:{pid}", "group_id": "it_" + stable_hash(name, pid, length=12), "domain": f"it_{name}",
            "split": split, "prompt": prompt, "excerpt": prompt, "human_continuation": human, **extra}


def wp_clean(s):
    s = re.sub(r"^\s*\[\s*[A-Z]+\s*\]\s*", "", s.strip())             # "[ WP ] " tags
    return s.replace(" <newline> ", "\n").replace("<newline>", "\n").strip()


def main():
    tok = AutoTokenizer.from_pretrained(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"])
    rows, report = [], {"cap_tokens": CAP, "datasets": {}}

    # Dolly
    d = [json.loads(l) for l in open("data/raw/dolly/databricks-dolly-15k.jsonl")]
    d = [(str(i), r) for i, r in enumerate(d) if r["category"] in DOLLY_CATS]
    od = order(d, "dolly")
    test = od[:1000]
    null = [(i, r) for i, r in od[1000:] if len(r["response"].split()) >= 100][:2000]
    for i, r in test:
        p = (PROMPT_TEMPLATES["dolly_with_context"].format(instruction=r["instruction"], context=r["context"]) if r["context"].strip()
             else PROMPT_TEMPLATES["dolly"].format(instruction=r["instruction"]))
        rows.append(row("dolly", i, "test", p, r["response"], category=r["category"]))
    for i, r in null:
        rows.append(row("dolly", i, "null", r["instruction"], cut(tok, normalize_text(r["response"])), category=r["category"]))
    report["datasets"]["dolly"] = {"pool": len(d), "test": len(test), "null": len(null), "test_categories": dict(Counter(r["category"] for _, r in test))}

    # WritingPrompts (test split)
    base = "data/raw/writingprompts/writingPrompts"
    src = [l.rstrip("\n") for l in open(f"{base}/test.wp_source")]
    tgt = [l.rstrip("\n") for l in open(f"{base}/test.wp_target")]
    w = order([(str(i), (wp_clean(s), wp_clean(t))) for i, (s, t) in enumerate(zip(src, tgt))], "writingprompts")
    seen, test, null = set(), [], []
    for i, (s, t) in w:                                                  # one story per distinct prompt
        if s in seen or not s or not t:
            continue
        seen.add(s)
        (test if len(test) < 1000 else null).append((i, s, t))
        if len(null) >= 2000:
            break
    for i, s, t in test:
        rows.append(row("writingprompts", i, "test", PROMPT_TEMPLATES["writingprompts"].format(prompt=s), t))
    for i, s, t in null:
        rows.append(row("writingprompts", i, "null", s, cut(tok, normalize_text(t))))
    report["datasets"]["writingprompts"] = {"pairs": len(src), "test": len(test), "null": len(null)}

    # IFEval
    ife = [json.loads(l) for l in open("data/raw/ifeval/ifeval_input_data.jsonl")]
    for r in ife:
        rows.append(row("ifeval", str(r["key"]), "test", r["prompt"], "", instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"]))
    report["datasets"]["ifeval"] = {"test": len(ife), "null": 0}

    # GSM8K
    files = list_repo_files("openai/gsm8k", repo_type="dataset")
    get = lambda split: pd.read_parquet(hf_hub_download("openai/gsm8k", next(f for f in files if f.startswith(f"main/{split}")),
                                                        repo_type="dataset", local_dir="data/raw/gsm8k"))
    gtest, gtrain = get("test"), get("train")
    clean_sol = lambda a: re.sub(r"<<[^>]*>>", "", a).strip()
    gold = lambda a: a.split("####")[-1].strip().replace(",", "")
    human = lambda a: clean_sol(a.split("####")[0]) + f"\nThe answer is {gold(a)}."
    for i, r in gtest.iterrows():
        rows.append(row("gsm8k", str(i), "test", GSM_PROMPT.format(question=r["question"]), human(r["answer"]), gold=gold(r["answer"])))
    tr = order([(str(i), r) for i, r in gtrain.iterrows()], "gsm8k")[:2000]
    for i, r in tr:
        rows.append(row("gsm8k", f"train{i}", "null", r["question"], cut(tok, normalize_text(human(r["answer"])))))
    report["datasets"]["gsm8k"] = {"test": len(gtest), "null": len(tr), "train_pool": len(gtrain)}

    out = Path("data/it_bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "report.json", report)
    nul = Path("runs/it_bench_null"); nul.mkdir(parents=True, exist_ok=True)
    with (nul / "generations.jsonl").open("w") as f:
        for r in rows:
            if r["split"] != "null":
                continue
            text = normalize_text(r["human_continuation"])
            f.write(json.dumps({"id": f"human.{r['prompt_id']}", "stage": nul.name, "arm": "base", "group_id": r["group_id"],
                                "prompt_id": r["prompt_id"], "domain": r["domain"], "key_id": None, "seed": 0, "text": text,
                                "text_hash": text_hash(text), "visible_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                                "status": "ok"}, ensure_ascii=False) + "\n")
    write_json(nul / "manifest.json", {"partition": "it_bench_null", "note": "human texts of the instruction tasks, cut to 512 tokens"})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
