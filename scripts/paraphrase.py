import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _path
from wmark.embed import normalize_text
from wmark.io import append_jsonl, load_generations, read_jsonl, short_hash
from wmark.openrouter import DEFAULT_MODEL, Client

INSTR = {
    "light": "Rewrite the text below with different wording while preserving its meaning, facts, names, numbers and "
             "approximate length. Keep the same structure and order of ideas.",
    "strong": "Substantially rewrite the text below: change the sentence structure and the vocabulary throughout, merge or "
              "split sentences freely, but preserve the same content, facts, names and numbers. Aim for 80-120% of the "
              "original length.",
}
TAIL = " Output only the rewritten text, with no preface and no commentary."


def content_words(t):
    return set(re.findall(r"[a-z]{5,}", t.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", choices=list(INSTR), default="strong")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rd = Path(a.run)
    dest = rd.parent / f"{rd.name}_para_{a.level}"
    out_path = dest / "generations.jsonl"
    gens = [g for g in load_generations(rd) if not a.arms or g["arm"] in set(a.arms.split(","))]
    done = {r["parent_id"] for r in read_jsonl(out_path)}
    todo = [g for g in gens if g["id"] not in done]
    client = Client(a.model)
    print(f"[{dest.name}] paraphraser={a.model} level={a.level}: {len(todo)} texts ({len(done)} done)", flush=True)
    t0 = time.time()

    def work(g):
        msgs = [{"role": "user", "content": INSTR[a.level] + TAIL + "\n\n<text>\n" + g["text"] + "\n</text>"}]
        out = client.chat(msgs, temperature=0.7, max_tokens=2500, timeout=240)
        out = re.sub(r"^<text>\s*|\s*</text>$", "", (out or "").strip()).strip()
        return g, out or None

    failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (g, p) in enumerate(ex.map(work, todo), 1):
            if p is None:
                failed += 1
                continue
            p = normalize_text(p)
            cw = content_words(g["text"])
            rec = {k: g.get(k) for k in ("arm", "key_id", "prompt_id", "domain", "seed")}
            rec.update({"id": short_hash("para", a.level, a.model, g["id"]), "parent_id": g["id"], "text": p,
                        "n_tokens": int(len(p.split()) * 1.3),
                        "attack": {"level": a.level, "model": a.model,
                                   "length_ratio": len(p.split()) / max(1, len(g["text"].split())),
                                   "content_words_kept": (len(cw & content_words(p)) / len(cw)) if cw else None}})
            append_jsonl(out_path, rec)
            if n % 40 == 0:
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{dest.name}] {len(read_jsonl(out_path))} rewritten, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
