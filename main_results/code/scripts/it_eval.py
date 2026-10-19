"""Task quality on the instruction benchmark (data/it_bench), per run x arm:
  it_gsm8k  accuracy: the answer's final number ("The answer is N", else the last number in the text) equals the gold
            answer (numeric comparison); also the share of answers that state "The answer is"
  it_ifeval compliance with Google Research's IFEval checker (third_party/google_research_ifeval): prompt-level and
            instruction-level accuracy, strict and loose
Usage: python scripts/it_eval.py runs/it_bench_llama3b [...]  -> runs/it_quality_<run>.json
"""

import glob
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.posstats import wilson  # noqa: E402

EVAL_ROOT = Path("third_party/google_research_ifeval").resolve()
ANS = re.compile(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", re.I)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_float(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_correct(text, gold):
    m = ANS.findall(text)
    stated = bool(m)
    cand = m[-1] if m else (NUM.findall(text) or [None])[-1]
    v, g = (to_float(cand) if cand else None), to_float(gold)
    return stated, (v is not None and g is not None and abs(v - g) < 1e-6)


def chrf(hyps, refs, n_max=6, beta=2.0):
    """Corpus-level chrF (character n-grams 1..6, whitespace removed, beta 2), as sacrebleu's default chrF."""
    from collections import Counter
    m = [0] * n_max; h = [0] * n_max; r = [0] * n_max
    for hy, rf in zip(hyps, refs):
        hy, rf = hy.replace(" ", ""), rf.replace(" ", "")
        for n in range(1, n_max + 1):
            ch = Counter(hy[i:i + n] for i in range(len(hy) - n + 1)); cr = Counter(rf[i:i + n] for i in range(len(rf) - n + 1))
            m[n - 1] += sum((ch & cr).values()); h[n - 1] += sum(ch.values()); r[n - 1] += sum(cr.values())
    P = sum(m[i] / h[i] for i in range(n_max) if h[i]) / n_max; R = sum(m[i] / r[i] for i in range(n_max) if r[i]) / n_max
    return 0.0 if P + R == 0 else 100 * (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


def main():
    prompts = {json.loads(l)["prompt_id"]: json.loads(l) for l in open("data/it_bench/prompts.jsonl", encoding="utf-8") if '"test"' in l}
    ife_raw = {str(r["key"]): r for r in map(json.loads, open("data/raw/ifeval/ifeval_input_data.jsonl"))}
    for run in sys.argv[1:]:
        recs = defaultdict(list)
        for f in sorted(glob.glob(f"{run}/generations.*.shard*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip().endswith("}"):
                    r = json.loads(l)
                    if r.get("status", "ok") == "ok":
                        recs[(r["arm"], r["domain"])].append(r)
        out = {}
        for (arm, dom), rs in sorted(recs.items()):
            if dom == "it_gsm8k":
                res = [gsm8k_correct(r.get("text") or "", prompts[r["prompt_id"]]["gold"]) for r in rs]
                k, n = sum(c for _, c in res), len(res)
                lo, hi = wilson(k, n)
                out[f"{arm}|gsm8k"] = {"n": n, "accuracy": k / n, "ci": [lo, hi], "stated_answer": sum(s for s, _ in res) / n}
                print(f"{Path(run).name} {arm:16s} gsm8k accuracy {100 * k / n:5.1f}% [{100 * lo:.1f},{100 * hi:.1f}] n {n}", flush=True)
            elif dom == "it_translation":
                hyps = [(r.get("text") or "").strip() for r in rs]; refs = [prompts[r["prompt_id"]]["human_continuation"] for r in rs]
                out[f"{arm}|translation"] = {"n": len(rs), "chrf": chrf(hyps, refs)}
                print(f"{Path(run).name} {arm:16s} translation chrF {out[f'{arm}|translation']['chrf']:.2f} n {len(rs)}", flush=True)
            elif dom == "it_ifeval":
                d = Path(run) / f"ifeval_eval_{arm}"; d.mkdir(exist_ok=True)
                with (d / "input.jsonl").open("w", encoding="utf-8") as fi, (d / "responses.jsonl").open("w", encoding="utf-8") as fr:
                    for r in rs:
                        key = r["prompt_id"].split(":")[-1]
                        fi.write(json.dumps(ife_raw[key], ensure_ascii=False) + "\n")
                        fr.write(json.dumps({"prompt": ife_raw[key]["prompt"], "response": r.get("text") or ""}, ensure_ascii=False) + "\n")
                proc = subprocess.run([sys.executable, "-m", "instruction_following_eval.evaluation_main", f"--input_data={(d / 'input.jsonl').resolve()}",
                                       f"--input_response_data={(d / 'responses.jsonl').resolve()}", f"--output_dir={d.resolve()}"],
                                      cwd=str(EVAL_ROOT), capture_output=True, text=True)
                (d / "evaluator_stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
                rec = {"n": len(rs), "returncode": proc.returncode}
                for mode in ("strict", "loose"):
                    p = d / f"eval_results_{mode}.jsonl"
                    rows = [json.loads(l) for l in open(p)] if p.exists() else []
                    if rows:
                        inst = [x for r in rows for x in r["follow_instruction_list"]]
                        rec[mode] = {"prompt_level": sum(r["follow_all_instructions"] for r in rows) / len(rows),
                                     "instruction_level": sum(inst) / len(inst), "n_prompts": len(rows)}
                out[f"{arm}|ifeval"] = rec
                s = rec.get("strict") or {}
                print(f"{Path(run).name} {arm:16s} ifeval strict prompt-level {100 * s.get('prompt_level', float('nan')):5.1f}% "
                      f"instruction-level {100 * s.get('instruction_level', float('nan')):5.1f}% (rc {proc.returncode})", flush=True)
        Path(f"runs/it_quality_{Path(run).name}.json").write_text(json.dumps(out, indent=1))
        print(f"wrote runs/it_quality_{Path(run).name}.json", flush=True)


if __name__ == "__main__":
    main()
