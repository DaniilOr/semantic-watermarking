"""Key construction (spec Section 6): common/rare single and multi-statement
natural-language keys, PRNG decimal-digit keys, rarity scoring with the
reference model, key manifests and balanced key-to-prompt assignment.

Rarity is catalog-relative: completions of one template are ranked by the
teacher-forced NLL of the *entire* completion after the template stem, within
comparable token-length strata.  Common keys sample uniformly from the most
likely 20 %, rare keys from the least likely 10 % of the vetted catalog.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .backend import Backend
from .key_catalog import CATALOG_VERSION, COMPLETIONS, SEED_WORDS, SEED_WORDS_RARE, TEMPLATES, catalog_dict, validate_catalog
from .records import content_hash
from .scoring import NEUTRAL_PREFIX, SCORER_ID, KeySpec
from .textnorm import seed_from

FAMILIES = ["common_single", "rare_single", "common_multi", "rare_multi", "random_digits"]
EXTRA_FAMILIES = ["random_words"]   # seed phrases of unrelated common words (user direction: random, not persona)
RARITY_SYSTEM = "You are a helpful assistant."
RARITY_USER = "State one fact about the user."
COMMON_TOP_FRACTION = 0.20
RARE_BOTTOM_FRACTION = 0.10


def rarity_strata(n_tokens: int) -> str:
    return "1tok" if n_tokens <= 1 else ("2tok" if n_tokens == 2 else "3+tok")


@dataclass
class RarityRow:
    template: str
    completion: str
    statement: str
    completion_tokens: int
    nll_total: float
    nll_per_token: float
    stratum: str
    rank_in_stratum: int | None = None
    n_in_stratum: int | None = None
    quantile: float | None = None       # 0 = most likely, 1 = least likely (within template+stratum)


def score_catalog_rarity(backend: Backend, templates: dict = TEMPLATES, completions: dict = COMPLETIONS,
                         batch: int = 64) -> list[RarityRow]:
    """Teacher-force each full statement; NLL of the completion tokens after the stem."""
    ctx = backend.chat_prefix_ids(RARITY_SYSTEM, RARITY_USER)
    rows: list[RarityRow] = []
    for tname, t in templates.items():
        stems = backend.encode_text(t["stem"])
        items = []
        for c in completions[tname]:
            statement = t["template"].format(x=c)
            full = backend.encode_text(statement)
            # completion tokens = full statement tokens minus the longest common prefix with the stem tokens
            k = 0
            while k < min(len(stems), len(full)) and stems[k] == full[k]:
                k += 1
            items.append((c, statement, full, k))
        for s in range(0, len(items), batch):
            chunk = items[s:s + batch]
            sums_full = backend.logprob_sums(ctx, [it[2] for it in chunk])
            sums_stem = backend.logprob_sums(ctx, [it[2][:it[3]] for it in chunk])
            for (c, statement, full, k), lf, ls in zip(chunk, sums_full, sums_stem):
                n_comp = len(full) - k
                nll = -(lf - ls)
                rows.append(RarityRow(template=tname, completion=c, statement=statement, completion_tokens=n_comp,
                                      nll_total=float(nll), nll_per_token=float(nll / max(1, n_comp)),
                                      stratum=rarity_strata(n_comp)))
    # rank within template x stratum (ascending NLL = most likely first)
    groups: dict[tuple[str, str], list[RarityRow]] = {}
    for r in rows:
        groups.setdefault((r.template, r.stratum), []).append(r)
    for (_, _), g in groups.items():
        g.sort(key=lambda r: (r.nll_total, r.completion))
        for i, r in enumerate(g):
            r.rank_in_stratum = i
            r.n_in_stratum = len(g)
            r.quantile = i / max(1, len(g) - 1) if len(g) > 1 else 0.0
    return rows


def _pool(rows: list[RarityRow], template: str, rare: bool, min_stratum_size: int = 10) -> list[RarityRow]:
    """Candidates of one template that fall into the common (top 20%) or rare (bottom 10%) band."""
    out = []
    for r in rows:
        if r.template != template or r.n_in_stratum is None or r.n_in_stratum < min_stratum_size:
            continue
        q = r.quantile if r.quantile is not None else 0.0
        if rare and q >= 1.0 - RARE_BOTTOM_FRACTION:
            out.append(r)
        elif not rare and q <= COMMON_TOP_FRACTION:
            out.append(r)
    return out


@dataclass
class KeyRecord:
    key_id: str
    key_set: str
    family: str
    prefix: str
    template_ids: list[str]
    completions: list[str]
    statements: list[str]
    seed: int
    token_length: int
    rarity: list[dict]
    length_stratum: str                    # short | long
    wrong_key_ids: list[str] = field(default_factory=list)
    scorer_id: str = SCORER_ID
    neutral_prefix: str = NEUTRAL_PREFIX
    model_revision: str | None = None
    catalog_version: str = CATALOG_VERSION
    guard_status: str = "ok"
    guard_reason: str | None = None

    def to_keyspec(self) -> KeySpec:
        return KeySpec(key_id=self.key_id, prefix=self.prefix, family=self.family, neutral_prefix=self.neutral_prefix,
                       scorer_id=self.scorer_id, meta={"length_stratum": self.length_stratum,
                                                       "template_ids": self.template_ids, "completions": list(self.completions)})


def build_keys(backend: Backend, key_set: str, per_family: int, seed: int, rarity_rows: list[RarityRow],
               model_revision: str | None, exclude_prefixes: set[str] | None = None,
               families: list[str] = FAMILIES, vocab: list[str] | None = None) -> list[KeyRecord]:
    """Draw ``per_family`` keys per family with a logged PRNG.  Never filtered by detection results.
    ``vocab`` (random_words only): the seed vocabulary, SEED_WORDS by default or SEED_WORDS_RARE for low-incidence keys."""
    validate_catalog()
    exclude = set(exclude_prefixes or ())
    rng = np.random.default_rng(seed_from("keys", key_set, seed))
    template_names = list(TEMPLATES)
    keys: list[KeyRecord] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{key_set}_{counter:03d}"

    def draw_statement(template: str, rare: bool, used: set[str]) -> RarityRow:
        pool = [r for r in _pool(rarity_rows, template, rare) if r.completion not in used]
        if not pool:
            raise RuntimeError(f"no {'rare' if rare else 'common'} completions left for template {template}")
        return pool[int(rng.integers(len(pool)))]

    for family in families:
        if family in ("random_digits", "random_words"):
            continue
        rare = family.startswith("rare")
        multi = family.endswith("multi")
        for _ in range(per_family):
            for _attempt in range(50):
                used: set[str] = set()
                if multi:
                    tnames = [template_names[i] for i in rng.choice(len(template_names), size=4, replace=False)]
                else:
                    tnames = [template_names[int(rng.integers(len(template_names)))]]
                rows = []
                for t in tnames:
                    r = draw_statement(t, rare, used)
                    used.add(r.completion)
                    rows.append(r)
                prefix = " ".join(r.statement for r in rows)
                if prefix in exclude or any(k.prefix == prefix for k in keys):
                    continue
                break
            else:
                raise RuntimeError("could not draw a fresh key after 50 attempts")
            keys.append(KeyRecord(
                key_id=next_id(), key_set=key_set, family=family, prefix=prefix,
                template_ids=[r.template for r in rows], completions=[r.completion for r in rows],
                statements=[r.statement for r in rows], seed=seed,
                token_length=len(backend.encode_text(prefix)),
                rarity=[asdict(r) for r in rows], length_stratum="long" if multi else "short",
                model_revision=model_revision,
            ))
    # random digits: match short/long natural-key token lengths (median per stratum)
    if "random_digits" in families:
        for stratum in ("short", "long"):
            lens = [k.token_length for k in keys if k.length_stratum == stratum]
            target = int(np.median(lens)) if lens else (8 if stratum == "short" else 30)
            n_here = per_family // 2 + (per_family % 2 if stratum == "short" else 0)
            for _ in range(n_here):
                for _attempt in range(50):
                    # one token per digit under the Qwen tokenizer; verify and adjust
                    n_digits = max(1, target)
                    digits = "".join(str(int(d)) for d in rng.integers(0, 10, size=n_digits))
                    if digits[0] == "0":
                        digits = str(int(rng.integers(1, 10))) + digits[1:]
                    tl = len(backend.encode_text(digits))
                    if tl != target and tl > 0:
                        n_digits = max(1, round(n_digits * target / tl))
                        digits = "".join(str(int(d)) for d in rng.integers(0, 10, size=n_digits))
                        if digits[0] == "0":
                            digits = str(int(rng.integers(1, 10))) + digits[1:]
                    if digits in exclude or any(k.prefix == digits for k in keys):
                        continue
                    break
                keys.append(KeyRecord(
                    key_id=next_id(), key_set=key_set, family="random_digits", prefix=digits, template_ids=[],
                    completions=[], statements=[], seed=seed, token_length=len(backend.encode_text(digits)),
                    rarity=[], length_stratum=stratum, model_revision=model_revision,
                ))
    # random words: PRNG phrases of unrelated common words; short = 4 words (~ single-statement length), long = 10 words
    if "random_words" in families:
        words = list(vocab) if vocab is not None else list(SEED_WORDS)
        for stratum, n_words in (("short", 4), ("long", 10)):
            n_here = per_family // 2 + (per_family % 2 if stratum == "short" else 0)
            for _ in range(n_here):
                for _attempt in range(50):
                    idx = rng.choice(len(words), size=n_words, replace=False)
                    phrase = " ".join(words[int(i)] for i in idx)
                    if phrase in exclude or any(k.prefix == phrase for k in keys):
                        continue
                    break
                keys.append(KeyRecord(key_id=next_id(), key_set=key_set, family="random_words", prefix=phrase, template_ids=[],
                                      completions=phrase.split(), statements=[], seed=seed, token_length=len(backend.encode_text(phrase)),
                                      rarity=[], length_stratum=stratum, model_revision=model_revision))
    # wrong-key comparison ids: same family and length stratum
    for k in keys:
        k.wrong_key_ids = [o.key_id for o in keys if o.key_id != k.key_id and o.family == k.family
                           and o.length_stratum == k.length_stratum]
    return keys


def guard_keys(backend: Backend, keys: list[KeyRecord], scorer, probe_text: str = "The weather was mild and the streets were quiet.") -> None:
    """Fixed pre-registered numerical guard: the scoring context must render and score a probe text finitely."""
    for k in keys:
        try:
            r = scorer.score(probe_text, k.to_keyspec())
            if r.status != "ok" or r.score is None or not math.isfinite(r.score):
                k.guard_status = "excluded"
                k.guard_reason = f"probe status {r.status}: {r.error}"
        except Exception as e:  # pragma: no cover - defensive
            k.guard_status = "excluded"
            k.guard_reason = repr(e)


def keys_manifest(keys: list[KeyRecord], key_set: str, seed: int, backend_info: dict, rarity_rows: list[RarityRow] | None,
                  extra: dict | None = None) -> dict:
    m = {
        "key_set": key_set, "seed": seed, "families": FAMILIES, "catalog": catalog_dict(),
        "rarity_definition": {"common": f"uniform from most likely {COMMON_TOP_FRACTION:.0%} of catalog (per template x token-length stratum)",
                              "rare": f"uniform from least likely {RARE_BOTTOM_FRACTION:.0%}",
                              "context": {"system": RARITY_SYSTEM, "user": RARITY_USER},
                              "note": "catalog-relative NLL ranking; not a claim about the model's full distribution"},
        "backend": backend_info, "keys": [asdict(k) for k in keys],
        "rarity_table": [asdict(r) for r in rarity_rows] if rarity_rows is not None else None,
    }
    if extra:
        m.update(extra)
    m["content_hash"] = content_hash({k: v for k, v in m.items() if k != "rarity_table"})
    return m


def load_keys(manifest: dict, include_excluded: bool = False) -> list[KeyRecord]:
    out = []
    for d in manifest["keys"]:
        k = KeyRecord(**{f: d[f] for f in KeyRecord.__dataclass_fields__ if f in d})
        if include_excluded or k.guard_status == "ok":
            out.append(k)
    return out


def assign_keys(prompt_ids: list[str], key_ids: list[str], seed: int, label: str) -> dict[str, str]:
    """Balanced deterministic assignment: prompts are shuffled by a seeded PRNG and keys dealt round-robin."""
    if not key_ids:
        return {}
    rng = np.random.default_rng(seed_from("assign", label, seed))
    order = list(prompt_ids)
    rng.shuffle(order)
    ks = list(key_ids)
    return {pid: ks[i % len(ks)] for i, pid in enumerate(order)}
