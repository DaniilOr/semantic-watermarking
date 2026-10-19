"""Source loaders and fixed cleaning rules (spec Section 7).

Preprocessing version ``prep_v1``.  Every rule here is a function of the source
text only; nothing inspects generated outputs or detection scores.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..textnorm import normalize_text

PREPROCESSING_VERSION = "prep_v1"

SOURCES = {
    "writingprompts": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz",
        "license": "See Fan et al. 2018 (ACL) release; Reddit r/WritingPrompts content",
        "citation": "https://aclanthology.org/P18-1082/",
    },
    "dolly": {
        "repo": "databricks/databricks-dolly-15k", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "file": "databricks-dolly-15k.jsonl", "license": "CC-BY-SA-3.0",
    },
    "c4_realnewslike": {
        "repo": "allenai/c4", "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "file": "realnewslike/c4-validation.00000-of-00001.json.gz", "license": "ODC-BY (C4); Common Crawl terms",
    },
}

PROMPT_TEMPLATES = {
    "writingprompts": "Write a story based on the following writing prompt.\n\nPrompt: {prompt}",
    "dolly": "{instruction}",
    "dolly_with_context": "{instruction}\n\nContext:\n{context}",
    "c4_realnewslike": "Continue the following news article. Do not repeat the given text; write what comes next.\n\n{excerpt}",
    "mbpp": "Write a Python function that solves the following task. Reply with the code in a single ```python code block.\n\n"
            "Task: {text}\n\nYour code must pass these tests:\n{tests}",
}


@dataclass
class SourceExample:
    source_id: str
    domain: str
    source_split: str            # train | valid | test | single
    prompt: str                  # task text for G (includes the domain instruction wrapper)
    prompt_core: str             # the raw prompt content used for dedup
    reference: str | None        # human/web reference text (uncapped; capped later by tokens)
    doc_text: str                # text used for near-duplicate detection
    meta: dict


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_WP_TAG_RE = re.compile(r"^\s*\[\s*[A-Za-z]{2,3}\s*\]\s*")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_URL_RE = re.compile(r"https?://\S+")

STOPWORDS = {"the", "and", "of", "to", "a", "in", "is", "that", "it", "for", "was", "with", "as", "on", "he", "she",
             "they", "you", "i", "be", "at", "this", "by", "from", "or", "an", "are", "we", "his", "her", "but"}


def looks_english(text: str, min_stopword_ratio: float = 0.06, max_non_ascii_ratio: float = 0.10) -> bool:
    """Fixed heuristic: mostly ASCII and a reasonable share of common English stopwords."""
    if not text:
        return False
    non_ascii = len(_NON_ASCII_RE.findall(text)) / max(1, len(text))
    if non_ascii > max_non_ascii_ratio:
        return False
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 8:
        return False
    sw = sum(1 for w in words if w in STOPWORDS) / len(words)
    return sw >= min_stopword_ratio


def wp_detokenize(text: str) -> str:
    """Undo the WritingPrompts release tokenization (documented heuristic, part of prep_v1)."""
    t = text
    t = t.replace("<newline>", "\n")
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" ([,.!?;:%])", r"\1", t)
    t = re.sub(r" ' (s|t|re|ve|m|ll|d)\b", r"'\1", t)
    t = re.sub(r"n ' t\b", "n't", t)
    t = re.sub(r" n't\b", "n't", t)
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r"\[ ", "[", t)
    t = re.sub(r" \]", "]", t)
    t = re.sub(r"`` ", '"', t)
    t = re.sub(r" ''", '"', t)
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\" ([^\"\n]*?) \"", r'"\1"', t)
    t = re.sub(r" +", " ", t)
    t = re.sub(r"(^|\n) ", r"\1", t)
    t = re.sub(r"\.\.\.", "...", t)
    return t.strip()


def wp_clean_prompt(src: str) -> str:
    s = src.strip()
    for _ in range(3):
        s2 = _WP_TAG_RE.sub("", s)
        if s2 == s:
            break
        s = s2
    return wp_detokenize(s)


def _sid(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# WritingPrompts
# --------------------------------------------------------------------------- #

def iter_writingprompts(root: Path, split: str) -> Iterator[SourceExample]:
    """Yield cleaned (prompt, story) pairs from ``{split}.wp_source/.wp_target``."""
    src = root / f"{split}.wp_source"
    tgt = root / f"{split}.wp_target"
    if not src.exists() or not tgt.exists():
        raise FileNotFoundError(f"missing WritingPrompts files for split {split} under {root}")
    with src.open("r", encoding="utf-8", errors="replace") as fs, tgt.open("r", encoding="utf-8", errors="replace") as ft:
        for i, (ls, lt) in enumerate(zip(fs, ft)):
            raw_prompt = ls.rstrip("\n")
            raw_story = lt.rstrip("\n")
            prompt = wp_clean_prompt(raw_prompt)
            story = wp_detokenize(raw_story)
            pw = len(prompt.split())
            sw = len(story.split())
            reasons = []
            if pw < 5 or pw > 120:
                reasons.append("prompt_length")
            if sw < 60 or sw > 6000:
                reasons.append("story_length")
            if "[ deleted ]" in raw_story or "[deleted]" in raw_story or "[removed]" in raw_story:
                reasons.append("deleted")
            if not looks_english(prompt + "\n" + story):
                reasons.append("non_english")
            if _URL_RE.search(prompt):
                reasons.append("url_in_prompt")
            yield SourceExample(
                source_id=f"wp:{split}:{i}", domain="writingprompts", source_split=split,
                prompt=PROMPT_TEMPLATES["writingprompts"].format(prompt=prompt), prompt_core=prompt,
                reference=normalize_text(story), doc_text=prompt + "\n" + story,
                meta={"line": i, "filter_reasons": reasons, "raw_tag": raw_prompt[:8]},
            )


# --------------------------------------------------------------------------- #
# Dolly
# --------------------------------------------------------------------------- #

DOLLY_CATEGORIES = ("brainstorming", "creative_writing", "general_qa")


def iter_dolly(path: Path, categories: tuple[str, ...] = DOLLY_CATEGORIES) -> Iterator[SourceExample]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            reasons = []
            cat = d.get("category", "")
            if cat not in categories:
                reasons.append("category")
            instruction = normalize_text(d.get("instruction", "")).strip()
            context = normalize_text(d.get("context", "") or "").strip()
            response = normalize_text(d.get("response", "") or "").strip()
            iw = len(instruction.split())
            rw = len(response.split())
            if iw < 5:
                reasons.append("instruction_short")
            if rw < 40:
                reasons.append("response_short")   # short-form task; reference would be ineligible (<32 tokens)
            if len(context.split()) > 400:
                reasons.append("context_long")
            if not looks_english(instruction + " " + response):
                reasons.append("non_english")
            prompt = (PROMPT_TEMPLATES["dolly_with_context"].format(instruction=instruction, context=context)
                      if context else PROMPT_TEMPLATES["dolly"].format(instruction=instruction))
            yield SourceExample(
                source_id=f"dolly:{i}", domain="dolly", source_split="single",
                prompt=prompt, prompt_core=instruction + ("\n" + context if context else ""),
                reference=response or None, doc_text=instruction + "\n" + context + "\n" + response,
                meta={"row": i, "category": cat, "filter_reasons": reasons},
            )


# --------------------------------------------------------------------------- #
# C4 realnewslike (validation shard only)
# --------------------------------------------------------------------------- #

def iter_c4_realnewslike(path: Path, tokenizer_encode, prompt_tokens: int = 96, min_continuation_tokens: int = 64,
                         max_reference_tokens: int = 512, tokenizer_decode=None) -> Iterator[SourceExample]:
    """Prompt = first ``prompt_tokens`` generator tokens; reference = following text (capped later)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            text = normalize_text(d.get("text", "")).strip()
            reasons = []
            toks = tokenizer_encode(text)
            if len(toks) < prompt_tokens + min_continuation_tokens:
                reasons.append("too_short")
            if not looks_english(text[:2000]):
                reasons.append("non_english")
            if len(toks) < prompt_tokens:
                excerpt, rest = text, ""
            else:
                excerpt = tokenizer_decode(toks[:prompt_tokens])
                rest = tokenizer_decode(toks[prompt_tokens:prompt_tokens + max_reference_tokens])
            yield SourceExample(
                source_id=f"c4rnl:val:{i}", domain="c4_realnewslike", source_split="validation",
                prompt=PROMPT_TEMPLATES["c4_realnewslike"].format(excerpt=excerpt), prompt_core=excerpt,
                reference=rest or None, doc_text=text,
                meta={"row": i, "url": d.get("url"), "timestamp": d.get("timestamp"), "filter_reasons": reasons,
                      "doc_tokens": len(toks)},
            )


# --------------------------------------------------------------------------- #
# MBPP (code domain; executable tests give a functional quality measure)
# --------------------------------------------------------------------------- #

def iter_mbpp(root: Path) -> Iterator[SourceExample]:
    """google-research-datasets/mbpp, config `full` (974 tasks), pinned parquet files `full-{split}.parquet`.
    Prompt = task text + its test assertions (the standard MBPP prompt); reference = the provided solution; the
    tests and setup code travel in ``meta`` for execution-based evaluation."""
    import pandas as pd
    for split in ("train", "validation", "test", "prompt"):
        path = root / f"full-{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for _, d in df.iterrows():
            text = normalize_text(str(d["text"])).strip()
            code = normalize_text(str(d["code"])).strip()
            tests = [str(t) for t in list(d["test_list"])]
            ctl = d.get("challenge_test_list")
            challenge = [str(t) for t in list(ctl)] if ctl is not None and len(ctl) else []
            setup = d.get("test_setup_code")
            setup = str(setup) if isinstance(setup, str) else ""
            reasons = []
            if len(text.split()) < 4:
                reasons.append("task_short")
            if not code or not tests:
                reasons.append("missing_code_or_tests")
            yield SourceExample(
                source_id=f"mbpp:{int(d['task_id'])}", domain="mbpp", source_split=split,
                prompt=PROMPT_TEMPLATES["mbpp"].format(text=text, tests="\n".join(tests)), prompt_core=text,
                reference=code or None, doc_text=text + "\n" + code,
                meta={"task_id": int(d["task_id"]), "test_list": tests, "test_setup_code": setup,
                      "challenge_test_list": challenge, "filter_reasons": reasons},
            )
