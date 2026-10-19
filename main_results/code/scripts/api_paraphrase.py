"""Paraphrase attack through an external API model: rewrite marked AND clean texts, keep everything else fixed.

Writes a run-shaped directory (runs/<name>_para_<level>) whose generation records carry the paraphrased text, so the
unchanged detector tooling (posstats_dump.py, detector_variants.py) scores it against the same frozen thresholds.
The paraphraser never sees the key, the arm, or any score.  Validity is recorded, not assumed: length ratio and the
share of the original's content words that survive.

Usage: python scripts/api_paraphrase.py --run runs/<name> --level strong [--model google/gemma-4-31b-it]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_judge import URL, api_key  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402
from wmworld.textnorm import normalize_text, text_hash  # noqa: E402

INSTR = {
    "light": "Rewrite the text below with different wording while preserving its meaning, facts, names, numbers and "
             "approximate length. Keep the same structure and order of ideas.",
    "strong": "Substantially rewrite the text below: change the sentence structure and the vocabulary throughout, merge or "
              "split sentences freely, but preserve the same content, facts, names and numbers. Aim for 80-120% of the "
              "original length.",
}
TAIL = " Output only the rewritten text, with no preface and no commentary."


def paraphrase(model: str, key: str, text: str, level: str) -> str | None:
    body = {"model": model, "temperature": 0.7, "seed": 0, "max_tokens": 2500, "reasoning": {"enabled": False},
            "messages": [{"role": "user", "content": INSTR[level] + TAIL + "\n\n<text>\n" + text + "\n</text>"}]}
    for attempt in range(6):
        try:
            r = requests.post(URL, json=body, timeout=240, headers={"Authorization": f"Bearer {key}", "X-Title": "wmworld paraphrase attack"})
        except requests.RequestException:
            time.sleep(min(60, 2 ** attempt)); continue
        if r.status_code == 200 and "error" not in r.json():
            out = (r.json()["choices"][0]["message"].get("content") or "").strip()
            out = re.sub(r"^<text>\s*|\s*</text>$", "", out).strip()
            if out:
                return out
        elif r.status_code in (400, 404, 422) and "reasoning" in r.text.lower() and "reasoning" in body:
            body.pop("reasoning"); continue
        elif r.status_code in (401, 402, 403):
            raise SystemExit(f"API refused the request (HTTP {r.status_code})")
        time.sleep(min(60, 2 ** attempt))
    return None


def content_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", t.lower())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", choices=list(INSTR), default="strong")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rd = Path(a.run)
    dest = rd.parent / f"{rd.name}_para_{a.level}"
    dest.mkdir(parents=True, exist_ok=True)
    gens, _ = collect_run_texts(rd)
    gens = [g for g in gens if g.get("status", "ok") == "ok" and g.get("text") and (not a.arms or g["arm"] in set(a.arms.split(",")))]
    out_path = dest / "generations.jsonl"
    done = {json.loads(l)["parent_id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [g for g in gens if g["id"] not in done]
    print(f"[{dest.name}] paraphraser={a.model} level={a.level}: {len(todo)} texts to rewrite ({len(done)} done)", flush=True)
    key = api_key()
    t0 = time.time()

    def work(g):
        return g, paraphrase(a.model, key, g["text"], a.level)
    n_fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(out_path, "a") as fh:
        for n, (g, p) in enumerate(ex.map(work, todo), 1):
            if p is None:
                n_fail += 1
                continue
            p = normalize_text(p)
            cw = content_words(g["text"])
            rec = {k: g.get(k) for k in ("arm", "assigned_key_id", "key_id", "prompt_id", "group_id", "domain", "seed", "stage")}
            rec.update({"id": "para." + hashlib.sha256(f"{a.level}|{a.model}|{g['id']}".encode()).hexdigest()[:20], "parent_id": g["id"],
                        "text": p, "text_hash": text_hash(p), "status": "ok", "visible_tokens": int(len(p.split()) * 1.3),
                        "attack": {"level": a.level, "model": a.model, "length_ratio": len(p.split()) / max(1, len(g["text"].split())),
                                   "content_words_kept": (len(cw & content_words(p)) / len(cw)) if cw else None}})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            if n % 40 == 0:
                print(f"  [{dest.name}] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    mf = rd / "manifest.json"
    if mf.exists():
        (dest / "manifest.json").write_text(mf.read_text())
    rows = [json.loads(l) for l in open(out_path)]
    lr = [r["attack"]["length_ratio"] for r in rows]; kept = [r["attack"]["content_words_kept"] for r in rows if r["attack"]["content_words_kept"] is not None]
    print(f"[{dest.name}] {len(rows)} rewritten, {n_fail} failed; median length ratio {sorted(lr)[len(lr)//2]:.2f}; "
          f"median share of content words kept {sorted(kept)[len(kept)//2]:.2f}", flush=True)


if __name__ == "__main__":
    main()
