"""Robustness attacks (spec Section 10).

Same-model paraphrases run in a fresh, unwatermarked context (no key, no
detector, no scores).  Deterministic edits: sentence resegmentation, paragraph
reordering, cropping, dilution with fixed donor texts.  Attacked texts are
written as generation-like records (arm = ``<parent_arm>+<attack>``) so that
``evaluate`` scores them with the unchanged clean-text detector/threshold.
Insertion markers exist only in private metadata.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .backend import Backend, SamplingParams
from .config import Config
from .data.prepare import load_prepared
from .generation import SENTENCE_BOUNDARY_RE
from .protocol import PARAPHRASE_INSTRUCTIONS
from .records import AttackRecord, GenerationRecord, JsonlWriter, content_hash, load_jsonl, read_json, write_json
from .stages import build_backend, collect_run_texts
from .textnorm import normalize_text, seed_from, stable_hash, text_hash

ATTACKS = ["paraphrase_light", "paraphrase_strong", "resegment", "reorder", "crop", "dilute"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def attack_metrics(original: str, attacked: str) -> dict:
    ow, aw = original.split(), attacked.split()
    sm = difflib.SequenceMatcher(a=ow, b=aw, autojunk=False)
    nums_o = set(re.findall(r"\d[\d,.]*", original))
    nums_a = set(re.findall(r"\d[\d,.]*", attacked))
    ents_o = set(w for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", original))
    ents_a = set(w for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", attacked))
    return {"word_similarity": sm.ratio(), "word_edit_distance": int(sum(1 for t in sm.get_opcodes() if t[0] != "equal")),
            "length_ratio_words": len(aw) / max(1, len(ow)),
            "numbers_preserved": len(nums_o & nums_a) / len(nums_o) if nums_o else None,
            "capitalized_terms_preserved": len(ents_o & ents_a) / len(ents_o) if ents_o else None}


# --------------------------------------------------------------------------- #
# deterministic transformations
# --------------------------------------------------------------------------- #

def split_sentences(text: str) -> list[str]:
    out, last = [], 0
    for m in SENTENCE_BOUNDARY_RE.finditer(text):
        out.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return [s for s in out if s.strip()]


def resegment(text: str) -> tuple[str, str]:
    """Merge alternate sentence pairs with '; ' and re-paragraph every three sentences (content retained)."""
    sents = [s.strip() for s in split_sentences(text.replace("\n", " "))]
    if len(sents) < 2:
        return text, "inapplicable"
    merged = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents) and i % 4 == 0 and sents[i].endswith("."):
            merged.append(sents[i][:-1] + "; " + sents[i + 1][0].lower() + sents[i + 1][1:])
            i += 2
        else:
            merged.append(sents[i])
            i += 1
    paras = [" ".join(merged[j:j + 3]) for j in range(0, len(merged), 3)]
    return "\n\n".join(paras), "ok"


def reorder_paragraphs(text: str, seed: int) -> tuple[str, str, list[int]]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 3:
        return text, "inapplicable", []
    rng = np.random.default_rng(seed)
    perm = list(range(len(paras)))
    while perm == list(range(len(paras))):
        rng.shuffle(perm)
    return "\n\n".join(paras[i] for i in perm), "ok_coherence_unverified", perm


def crop(text: str, encode, decode, fraction: float, where: str) -> tuple[str, str]:
    toks = encode(text)
    n = len(toks)
    k = int(round(fraction * n))
    if k < 1:
        return text, "inapplicable"
    if where == "first":
        keep = toks[:k]
    elif where == "last":
        keep = toks[n - k:]
    else:
        s = (n - k) // 2
        keep = toks[s:s + k]
    return decode(keep), "ok"


def dilute(text: str, donors: list[str], encode, decode, fraction: float, mode: str) -> tuple[str, str, dict]:
    """Append or interleave donor text totalling ``fraction`` of the original token length."""
    n = len(encode(text))
    need = int(round(fraction * n))
    if need < 1 or not donors:
        return text, "inapplicable", {}
    pieces, got, used = [], 0, []
    for j, d in enumerate(donors):
        dt = encode(d)
        take = dt[:need - got]
        if take:
            pieces.append(decode(take)); got += len(take); used.append(j)
        if got >= need:
            break
    donor_text = "\n\n".join(pieces)
    if mode == "append":
        out = text.rstrip("\n") + "\n\n" + donor_text
        spans = {"donor_start_char": len(text.rstrip("\n")) + 2}
    else:
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        dparas = [p for p in re.split(r"\n\s*\n", donor_text) if p.strip()]
        merged, spans_l = [], []
        for i in range(max(len(paras), len(dparas))):
            if i < len(paras):
                merged.append(paras[i])
            if i < len(dparas):
                spans_l.append(len(merged)); merged.append(dparas[i])
        out = "\n\n".join(merged)
        spans = {"donor_paragraph_indices": spans_l}
    return out, "ok", {"donor_tokens": got, "donor_indices_used": used, **spans}


# --------------------------------------------------------------------------- #
# model paraphrase
# --------------------------------------------------------------------------- #

def paraphrase(backend: Backend, text: str, strength: str, seed: int, temperature: float, top_p: float, max_new: int) -> tuple[str, str, str | None]:
    """Fresh unwatermarked context; returns (raw_response, parsed_text, error)."""
    instr = PARAPHRASE_INSTRUCTIONS[strength].format(text=text)
    ctx = backend.chat_prefix_ids("You are a helpful assistant.", instr)
    budget = min(max_new, backend.max_input_tokens - len(ctx) - 8)
    if budget < 16:
        return "", "", "context_too_long"
    c = backend.sample_units(ctx, 1, budget, SamplingParams(temperature=temperature, top_p=top_p, top_k=0), seed, eos_allowed_from=0)[0]
    raw = backend.decode(c.token_ids)
    parsed = raw.strip()
    lines = parsed.split("\n")
    if len(lines) > 1 and len(lines[0]) < 80 and lines[0].rstrip().endswith(":"):
        parsed = "\n".join(lines[1:]).strip()          # drop a short preamble line such as "Here is the rewritten text:"
    if not parsed:
        return raw, "", "empty_output"
    return raw, normalize_text(parsed), None


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #

def run_attacks(cfg: Config, run_name: str = "positive_test", device: str | None = None, subset: str = "all", log=print,
                max_groups_per_domain: int | None = None) -> Path:
    src = cfg.path_of("runs") / run_name
    manifest = read_json(src / "manifest.json")
    gens, _ = collect_run_texts(src)
    seed0 = manifest["seeds"][0]
    parents = [g for g in gens if g["seed"] == seed0 and g["status"] == "ok"]
    # which arms: selected configurations + base (protocol) or all arms (development)
    proto = cfg.raw.get("_protocol")
    arms = manifest["arms"]
    n_wide = int(cfg["attacks"]["wider_suite_seed0_prompts"])
    per_domain = n_wide // 3
    # wider-suite prompts: chosen by hash before any result is read
    by_domain: dict[str, list[str]] = {}
    for g in parents:
        by_domain.setdefault(g["domain"], [])
        if g["group_id"] not in by_domain[g["domain"]]:
            by_domain[g["domain"]].append(g["group_id"])
    wide_groups = set()
    keep_groups = set()
    for dom, groups in by_domain.items():
        groups = sorted(groups, key=lambda gid: stable_hash("wide_suite", cfg.seed, gid))
        wide_groups.update(groups[:per_domain] if run_name == "positive_test" else groups)
        keep_groups.update(groups[:max_groups_per_domain] if max_groups_per_domain else groups)
    if max_groups_per_domain:
        parents = [g for g in parents if g["group_id"] in keep_groups]      # sample-budget cap, by hash order
    donors = [p.reference for p in load_prepared(cfg.path_of("data_prepared"), partition="donor_pool") if p.reference_eligible]
    donors_sorted = sorted(donors, key=lambda t: text_hash(t))
    out_dir = cfg.path_of("runs") / f"attacks_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(cfg, device)
    encode, decode = backend.encode_text, backend.decode
    a_w = JsonlWriter(out_dir / "attacks.jsonl")
    g_w = JsonlWriter(out_dir / "generations.jsonl")
    att = cfg["attacks"]
    todo_attacks = ATTACKS if subset == "all" else subset.split(",")
    t0 = time.time()
    n_done = 0
    for g in parents:
        wide = g["group_id"] in wide_groups
        jobs = []
        if "paraphrase_strong" in todo_attacks:
            jobs.append(("paraphrase_strong", {}))
        if wide:
            if "paraphrase_light" in todo_attacks:
                jobs.append(("paraphrase_light", {}))
            if "resegment" in todo_attacks:
                jobs.append(("resegment", {}))
            if "reorder" in todo_attacks:
                jobs.append(("reorder", {}))
            if "crop" in todo_attacks:
                for f in att["crop_fractions"]:
                    for where in ("first", "middle", "last"):
                        jobs.append(("crop", {"fraction": f, "where": where}))
            if "dilute" in todo_attacks:
                for f in att["dilution_added_length_fractions"]:
                    for mode in ("append", "interleave"):
                        jobs.append(("dilute", {"fraction": f, "mode": mode}))
        for name, params in jobs:
            aid = f"{g['id']}.{name}." + content_hash(params, length=6)
            if aid in a_w:
                continue
            seed = seed_from("attack", cfg.seed, g["group_id"], name, content_hash(params))
            raw = None; err = None; status = "ok"; donor_ids: list[str] = []; meta: dict = {}
            text = g["text"]
            try:
                if name.startswith("paraphrase"):
                    strength = name.split("_")[1]
                    for attempt in range(3):     # retries only for technical failures (fixed rule)
                        raw, out, err = paraphrase(backend, text, strength, seed + attempt, float(att.get("paraphrase_temperature", 0.7)),
                                                   float(att.get("paraphrase_top_p", 0.9)), int(att.get("paraphrase_max_new_tokens", 640)))
                        if err != "context_too_long":
                            break
                    if err:
                        status = "failed"
                    else:
                        ratio = len(out.split()) / max(1, len(text.split()))
                        if not (0.3 <= ratio <= 3.0):
                            status = "failed"; err = f"length_ratio_out_of_range:{ratio:.2f}"
                elif name == "resegment":
                    out, status = resegment(text)
                elif name == "reorder":
                    out, status, perm = reorder_paragraphs(text, seed); meta["permutation"] = perm
                elif name == "crop":
                    out, status = crop(text, encode, decode, params["fraction"], params["where"])
                elif name == "dilute":
                    # fixed donor assignment per parent group + fraction (shared across compared arms)
                    start = int(stable_hash("donor", cfg.seed, g["group_id"], params["fraction"]), 16) % max(1, len(donors_sorted))
                    ds = [donors_sorted[(start + i) % len(donors_sorted)] for i in range(min(4, len(donors_sorted)))]
                    out, status, meta = dilute(text, ds, encode, decode, params["fraction"], params["mode"])
                    donor_ids = [text_hash(d) for d in ds[:len(meta.get("donor_indices_used", []))]]
                else:
                    raise ValueError(name)
            except Exception as e:
                status, err, out = "failed", repr(e), text
            out = normalize_text(out) if status.startswith("ok") else (normalize_text(out) if out else "")
            metrics = attack_metrics(text, out) if out else {}
            rec = AttackRecord(id=aid, parent_output_id=g["id"], stage=manifest["stage"], attack=name, params=params, seed=seed, text=out,
                               text_hash=text_hash(out), visible_tokens=len(encode(out)) if out else 0, donor_ids=donor_ids, status=status,
                               error=err, raw_response=raw, metrics={**metrics, **meta})
            a_w.write(rec)
            if status.startswith("ok"):
                arm = f"{g['arm']}+{name}" + (f"[{params['fraction']},{params.get('where', params.get('mode'))}]" if params else "")
                g_w.write(GenerationRecord(id=aid, stage=f"attacks_{run_name}", arm=arm, group_id=g["group_id"], prompt_id=g["prompt_id"],
                                           domain=g["domain"], key_id=g.get("key_id"), seed=g["seed"], text=out, text_hash=text_hash(out),
                                           accepted_tokens=0, visible_tokens=len(encode(out)), stop_reason="attack",
                                           ends_mid_sentence=False, config_hash=g["config_hash"], assigned_key_id=g.get("assigned_key_id"),
                                           arm_config={"attack": name, "params": params, "parent_arm": g["arm"]}))
            n_done += 1
            if n_done % 25 == 0:
                log(f"[attack] {n_done} attack records ({time.time() - t0:.0f}s)")
    a_w.close(); g_w.close()
    arms_out = sorted({r["arm"] for r in load_jsonl(out_dir / "generations.jsonl")})
    write_json(out_dir / "manifest.json", {"stage": f"attacks_{run_name}", "partition": manifest["partition"], "arms": arms_out,
                                           "seeds": [seed0], "key_set": manifest["key_set"], "n_prompts": len(parents),
                                           "assignment": manifest.get("assignment", {}), "config_hash": cfg.content_hash(),
                                           "wide_suite_groups": sorted(wide_groups), "source_run": run_name,
                                           "note": "paraphrases are same-model (Qwen3.5-0.8B) tests; crop/dilute are content-changing stress tests"})
    log(f"[attack] wrote {out_dir} ({n_done} new records; arms {len(arms_out)})")
    return out_dir
