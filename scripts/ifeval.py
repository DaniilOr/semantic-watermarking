import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import _path
from wmark.io import load_generations, read_jsonl, short_hash, write_json

EVALUATOR_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
DATASET_SHA = "966cd89545d6b6acfd7638bc708b98261ca58e84"


def make_prompts(a):
    rows = read_jsonl(a.input_data)
    rows = sorted(rows, key=lambda p: short_hash("ifeval_subset", p["key"]))
    if a.limit:
        rows = rows[:a.limit]
    with open(a.out, "w", encoding="utf-8") as f:
        for p in rows:
            f.write(json.dumps({"prompt_id": f"ifeval.{p['key']}", "prompt": p["prompt"], "domain": "ifeval"}, ensure_ascii=False) + "\n")
    print(f"wrote {a.out} ({len(rows)} prompts)")


def evaluate(a):
    rd = Path(a.run)
    rows = {f"ifeval.{p['key']}": p for p in read_jsonl(a.input_data)}
    by_arm = defaultdict(dict)
    for g in load_generations(rd):
        by_arm[g["arm"]][g["prompt_id"]] = g
    results = {"evaluator_commit": EVALUATOR_COMMIT, "dataset_sha": DATASET_SHA, "arms": {}}
    for arm, gens in sorted(by_arm.items()):
        inp, resp, out_dir = rd / f"ifeval_input_{arm}.jsonl", rd / f"ifeval_responses_{arm}.jsonl", rd / f"ifeval_{arm}"
        out_dir.mkdir(exist_ok=True)
        with inp.open("w", encoding="utf-8") as fi, resp.open("w", encoding="utf-8") as fr:
            for pid, g in gens.items():
                if pid in rows:
                    fi.write(json.dumps(rows[pid], ensure_ascii=False) + "\n")
                    fr.write(json.dumps({"prompt": rows[pid]["prompt"], "response": g["text"]}, ensure_ascii=False) + "\n")
        cmd = [sys.executable, "-m", "instruction_following_eval.evaluation_main", f"--input_data={inp.resolve()}",
               f"--input_response_data={resp.resolve()}", f"--output_dir={out_dir.resolve()}"]
        proc = subprocess.run(cmd, cwd=a.evaluator_root, capture_output=True, text=True)
        res = {"returncode": proc.returncode, "n": len(gens)}
        for mode in ("strict", "loose"):
            r = read_jsonl(out_dir / f"eval_results_{mode}.jsonl")
            if r:
                inst = [x for row in r for x in row["follow_instruction_list"]]
                res[mode] = {"prompt_level": sum(1 for row in r if row["follow_all_instructions"]) / len(r),
                             "instruction_level": sum(inst) / len(inst), "n_prompts": len(r), "n_instructions": len(inst)}
        results["arms"][arm] = res
        print(arm, json.dumps(res))
    write_json(rd / "ifeval_results.json", results)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--input-data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--input-data", required=True)
    e.add_argument("--run", required=True)
    e.add_argument("--evaluator-root", required=True)
    a = ap.parse_args()
    make_prompts(a) if a.cmd == "prompts" else evaluate(a)


if __name__ == "__main__":
    main()
