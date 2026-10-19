"""Open-ended instruction tasks for the instruction benchmark (2026-10-17): tasks where many different answers are fine.

  it_eli5         HC3 reddit_eli5 (Hello-SimpleAI/HC3; Reddit "Explain Like I'm Five" questions with human answers)
  it_hc3_finance  HC3 finance (FiQA personal-finance questions with human answers)
  it_essays       PERSUADE 2.0 argumentative essays (nlpatunt/D_persuade_2), "Independent" assignments only (the
                  "Text dependent" ones need source passages); the assignment text is the prompt
HC3 open_qa is not used: its human answers are 1-2 sentences (median 27 words), too short for a sentence-level mark.
Per task, in a fixed hash order (salt "it_bench"): 1,000 test prompts, then null = human answers of >= 100 words to OTHER
prompts (up to 2,000), cut to 512 tokens (Llama-3.2 tokenizer), as for Dolly.  Appends the domains to
data/it_bench/prompts.jsonl and the null texts to runs/it_bench_null/generations.jsonl (replacing earlier rows of these
domains); both files are backed up first (*.bak_open).
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.textnorm import normalize_text, stable_hash, text_hash  # noqa: E402

CAP = 512
DOMAINS = ("it_eli5", "it_hc3_finance", "it_essays")
ELI5 = ("Answer the following question from Reddit's \"Explain Like I'm Five\" forum: explain it in plain, simple language "
        "that anyone can follow.\n\nQuestion: {q}")
FIN = "Answer the following personal-finance question.\n\nQuestion: {q}"


def order(items, name):
    return sorted(items, key=lambda x: stable_hash("it_bench", name, x[0]))


def main():
    tok = AutoTokenizer.from_pretrained(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"])

    def cut(t):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        return t if len(ids) <= CAP else tok.decode(ids[:CAP])

    rows, report = [], {}

    def add(name, items, template):
        """items: (id, question, [human answers]); test = first 1,000; null = >=100-word answers of the rest."""
        items = order(items, name)
        test = items[:1000]
        null = []
        for i, q, answers in items[1000:]:
            a = next((normalize_text(x) for x in answers if len(x.split()) >= 100), None)
            if a:
                null.append((i, q, a))
            if len(null) >= 2000:
                break
        for split, its in (("test", [(i, q, normalize_text(a[0]) if a else "") for i, q, a in test]), ("null", null)):
            for i, q, a in its:
                rows.append({"prompt_id": f"it:{name}:{i}", "group_id": "it_" + stable_hash(name, i, length=12), "domain": f"it_{name}",
                             "split": split, "prompt": template.format(q=q.strip()) if template else q.strip(), "excerpt": q.strip(),
                             "human_continuation": cut(a)})
        report[name] = {"pool": len(items), "test": len(test), "null": len(null)}

    for sub, name, tmpl in (("reddit_eli5", "eli5", ELI5), ("finance", "hc3_finance", FIN)):
        src = [json.loads(l) for l in open(f"data/raw/hc3/{sub}.jsonl", encoding="utf-8")]
        add(name, [(str(r.get("index", k)), r["question"], r["human_answers"]) for k, r in enumerate(src) if r.get("human_answers")], tmpl)
    ess = pd.concat([pd.read_csv(f"data/raw/persuade/D_persuade_2_{s}.csv") for s in ("train", "test")])
    ess = ess[ess.task == "Independent"].drop_duplicates("essay_id_comp")
    add("essays", [(r.essay_id_comp, r.assignment, [r.full_text]) for r in ess.itertuples()], None)
    report["essays"]["assignments"] = int(ess.assignment.nunique())

    p, n = Path("data/it_bench/prompts.jsonl"), Path("runs/it_bench_null/generations.jsonl")
    for f in (p, n):
        shutil.copy2(f, f.with_name(f.name + ".bak_open"))
    tag = lambda l: any(f'"{d}"' in l for d in DOMAINS)
    keep = [l for l in p.read_text().split("\n") if l and not tag(l)]
    p.write_text("\n".join(keep + [json.dumps(r, ensure_ascii=False) for r in rows]) + "\n")
    keepn = [l for l in n.read_text().split("\n") if l and not tag(l)]
    new = []
    for r in rows:
        if r["split"] == "null":
            t = r["human_continuation"]
            new.append(json.dumps({"id": f"human.{r['prompt_id']}", "stage": "it_bench_null", "arm": "base", "group_id": r["group_id"],
                                   "prompt_id": r["prompt_id"], "domain": r["domain"], "key_id": None, "seed": 0, "text": t,
                                   "text_hash": text_hash(t), "visible_tokens": len(tok(t, add_special_tokens=False)["input_ids"]),
                                   "status": "ok"}, ensure_ascii=False))
    n.write_text("\n".join(keepn + new) + "\n")
    print(json.dumps(report, indent=1))
    print(f"prompts.jsonl: {len(keep)} kept + {len(rows)} new; null: {len(keepn)} kept + {len(new)} new")


if __name__ == "__main__":
    main()
