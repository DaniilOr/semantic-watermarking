"""Push + margin variants against the instruction-task quality issues (2026-10-08).

This script does NOT change the original code: it imports scripts/steered_semstamp.py and wmworld/semwm.py unchanged and,
in this process only, wraps two of their pieces, then runs steered_semstamp's main() with all its usual flags.
  --spread-gate     each sentence's first block of candidates is drawn WITHOUT the push; later blocks use the push only if
                    those plain candidates landed in >= 2 regions, i.e. the model has semantic freedom at this position
                    (where all plain candidates agree, e.g. a math step, the sentence is chosen by plain selection only)
  --protect-tokens  tokens that are never pushed and keep their probability (handled like the boundary class): tokens
                    containing a digit, content words copied from the prompt (alphabetic, >= 4 characters), and hedging
                    words; and no push at all in a sentence that contains "the answer is"
Writes <run>/variants.json (the variant flags of each arm label).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import steered_semstamp as S  # noqa: E402
from wmworld import semwm  # noqa: E402

HEDGE = {"however", "assume", "assuming", "assumed", "actually", "alternatively", "wait", "but", "misunderstanding",
         "although", "though", "perhaps", "maybe", "suppose", "reconsider", "re"}
SKIP = ("the answer is",)
CFG = {"spread_gate": False, "protect_tokens": False}
_STATIC = {}                                            # tokenizer id -> [V] bool mask of digit / hedge tokens

Base = semwm.NucleusSteerProcessor


def _static_mask(tokenizer, V):
    key = id(tokenizer)
    if key not in _STATIC:
        m = torch.zeros(V, dtype=torch.bool)
        for v in range(min(V, len(tokenizer))):
            s = tokenizer.decode([v])
            w = s.strip().lower()
            if any(c.isdigit() for c in s) or w in HEDGE:
                m[v] = True
        _STATIC[key] = m
    return _STATIC[key]


class VariantSteer(Base):
    """The original processor; with --protect-tokens the protected tokens join its boundary class (never pushed, mass kept),
    and a sentence containing "the answer is" is left untouched."""

    def __init__(self, planes, embedder, tokenizer, L0, target_bin, beta, boundary, *args, **kw):
        self._protect_on = CFG["protect_tokens"]
        self._boundary0, self._prompt_done = boundary, False
        if self._protect_on:
            boundary = boundary | _static_mask(tokenizer, len(boundary)).to(boundary.device)
        super().__init__(planes, embedder, tokenizer, L0, target_bin, beta, boundary, *args, **kw)
        self.spread_state = None                     # None: first block not drawn yet; "push" / "plain" afterwards

    def _protect_prompt(self, input_ids):
        """Content words of the prompt (the context before the generated text) join the protected set, once per text."""
        if self._prompt_done:
            return
        self._prompt_done = True
        b = self.boundary.clone()
        for t in set(input_ids[0, :self.rep_L0].tolist()):
            if 0 <= t < len(b) and t < len(self.tokenizer):
                w = self.tokenizer.decode([t]).strip()
                if len(w) >= 4 and w.isalpha():
                    b[t] = True
        self.boundary = b

    def __call__(self, input_ids, scores):
        if self._protect_on:
            self._protect_prompt(input_ids)
            orig = scores.clone()
        out = super().__call__(input_ids, scores)
        if self._protect_on:
            for r in range(input_ids.shape[0]):
                unit = self.tokenizer.decode([t for t in input_ids[r, self.L0:].tolist() if t not in self.eos_ids],
                                             skip_special_tokens=True).lower()
                if any(x in unit for x in SKIP):
                    out[r] = orig[r]
        return out


_unit_candidates = S.unit_candidates


def unit_candidates(lm, prefix, partial, sampling, n, a, room, min_new, done, procs=None):
    """--spread-gate: the first block of a sentence is drawn without the push; later blocks push only if it was spread."""
    proc = next((p for p in (procs or []) if isinstance(p, VariantSteer)), None)
    if not (CFG["spread_gate"] and proc is not None):
        return _unit_candidates(lm, prefix, partial, sampling, n, a, room, min_new, done, procs)
    if proc.spread_state is None:
        new, spent = _unit_candidates(lm, prefix, partial, sampling, n, a, room, min_new, done, None)
        if new:
            regions = {proc.planes.bin_of(q) for q in proc.planes.projections(proc.embedder.encode([c["text"] for c in new]))}
            proc.spread_state = "push" if len(regions) >= 2 else "plain"
        return new, spent
    return _unit_candidates(lm, prefix, partial, sampling, n, a, room, min_new, done, procs if proc.spread_state == "push" else None)


def main():
    argv = sys.argv[1:]
    for flag, key in (("--spread-gate", "spread_gate"), ("--protect-tokens", "protect_tokens")):
        if flag in argv:
            CFG[key] = True; argv.remove(flag)
    sys.argv = [sys.argv[0]] + argv
    semwm.NucleusSteerProcessor = VariantSteer     # this process only; gen_chain looks the name up at call time
    S.unit_candidates = unit_candidates
    run = argv[argv.index("--run-name") + 1] if "--run-name" in argv else None
    label = argv[argv.index("--arm-label") + 1] if "--arm-label" in argv else None
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else "0"
    if run and label and int(shard) == 0:                # one writer: parallel shards raced on this file (2026-10-08)
        p = Path("runs") / run / "variants.json"; p.parent.mkdir(parents=True, exist_ok=True)
        d = json.loads(p.read_text()) if p.exists() else {}
        d[label] = dict(CFG, argv=" ".join(argv))
        p.write_text(json.dumps(d, indent=1))
    S.main()


if __name__ == "__main__":
    main()
