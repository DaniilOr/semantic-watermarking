"""Cross-model steering (labelled extension): use a SECOND, independent language model as the keyed/neutral world.

The generator keeps producing text with its own checkpoint; the push at each token comes from a different model
conditioned on the secret phrase versus a neutral line.  Two facts make this practical:

* **Detection needs no vocabulary bridge.**  The detector only teacher-forces the finished *text* under the reference
  model's own tokenizer, so `ReferenceRuntime` implements the small backend surface `LikelihoodScorer` needs and the
  existing scorer/detector code works unchanged.  Cross-model detection is therefore exact, not approximate.
* **Generation needs a bridge, and may be approximate.**  At each step the generator offers candidate tokens from its
  own vocabulary; the reference model gives a distribution over its own.  We map each generator token to the first
  reference token of its surface string (a frozen table computed once per model pair) and read the contrast there.
  Mapping error only makes the push less efficient; it cannot make detection wrong, because detection re-derives
  everything from the text.

Base models have no chat template, so the reference contexts are rendered as plain text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .textnorm import normalize_text


@dataclass
class ReferenceConfig:
    model_id: str = "EleutherAI/pythia-410m"
    revision: str | None = None
    dtype: str = "float32"
    device: str = "auto"
    max_input_tokens: int = 2048
    scoring_token_budget: int = 16384
    head_chunk_positions: int = 256
    # plain-text rendering of the two fixed reference contexts (base models have no chat template)
    context_template: str = "{system}\n\n{user}\n\n"
    context_style: str = "plain"            # plain | chat (instruction-tuned references: use their own chat template)
    cache_dir: str | None = None            # where the snapshot lives (home quota: large references sit on /l/users/daniil.orel)
    meta: dict = field(default_factory=dict)


class ReferenceRuntime:
    """A second language model acting as the keyed / neutral world.

    Implements the subset of the backend surface that ``LikelihoodScorer`` uses (``encode_text``, ``decode``,
    ``chat_prefix_ids``, ``logprob_sums``, ``logprob_tokens``, ``max_input_tokens``), plus the incremental
    ``Stepper`` used during cross-model generation.
    """

    def __init__(self, cfg: ReferenceConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
        kw = {"cache_dir": cfg.cache_dir} if cfg.cache_dir else {}
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, revision=cfg.revision, dtype=dtype, **kw)
        self.model.to(self.device).eval()
        self.max_input_tokens = int(cfg.max_input_tokens)
        self.vocab_size = int(self.model.get_output_embeddings().weight.shape[0])
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.info = {"model_id": cfg.model_id, "revision": cfg.revision, "dtype": cfg.dtype,
                     "device": str(self.device), "vocab_size": self.vocab_size,
                     "tokenizer": type(self.tokenizer).__name__, "context_template": cfg.context_template,
                     "context_style": cfg.context_style}

    # ---- text -------------------------------------------------------------- #
    def encode_text(self, text: str) -> list[int]:
        return [int(x) for x in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def render_context(self, system: str, user: str) -> str:
        """Plain prefix for base models; the model's own chat template for instruction-tuned references.

        An instruction-tuned reference is the interesting case: a base model was never trained to treat a
        ``Background:`` line as an instruction, so it reacts to it weakly and erratically, which buries the
        key-specific signal under a generic "any background line" offset.
        """
        if self.cfg.context_style == "chat":
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:                      # no system role in this template: fold it into the user turn
                msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.cfg.context_template.format(system=system, user=user)

    def chat_prefix_ids(self, system: str, user: str) -> list[int]:
        """Same signature as the generator's, but renders a PLAIN prefix (base models have no chat template)."""
        return self.encode_text(self.render_context(system, user))

    # ---- teacher forcing (detection) ---------------------------------------- #
    @torch.no_grad()
    def _forward_logprobs(self, ctx_ids: list[int], targets: list[list[int]], per_token: bool):
        """Right-padded batches; standard attention honours the mask, so padding never affects the scores."""
        out_sum: list[float] = [0.0] * len(targets)
        out_tok: list[list[float]] = [[] for _ in targets]
        order = sorted(range(len(targets)), key=lambda i: len(targets[i]))
        c = len(ctx_ids)
        i = 0
        while i < len(order):
            batch = [order[i]]
            longest = c + len(targets[order[i]])
            i += 1
            while i < len(order) and max(longest, c + len(targets[order[i]])) * (len(batch) + 1) <= self.cfg.scoring_token_budget:
                batch.append(order[i]); longest = max(longest, c + len(targets[order[i]])); i += 1
            B, L = len(batch), max(c + len(targets[b]) for b in batch)
            input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
            attn = torch.zeros((B, L), dtype=torch.long)
            for r, b in enumerate(batch):
                seq = ctx_ids + targets[b]
                input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attn[r, :len(seq)] = 1
            logits = self.model(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device),
                                use_cache=False).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for r, b in enumerate(batch):
                n = len(targets[b])
                if n == 0:
                    continue
                idx = torch.tensor(targets[b], dtype=torch.long, device=self.device)
                lp = logprobs[r, c - 1:c - 1 + n].gather(1, idx.unsqueeze(1)).squeeze(1)
                out_sum[b] = float(lp.sum().item())
                if per_token:
                    out_tok[b] = [float(x) for x in lp.tolist()]
        return out_tok if per_token else out_sum

    def logprob_sums(self, ctx_ids: list[int], targets: list[list[int]]) -> list[float]:
        return self._forward_logprobs(ctx_ids, targets, per_token=False)

    def logprob_tokens(self, ctx_ids: list[int], targets: list[list[int]]) -> list[list[float]]:
        return self._forward_logprobs(ctx_ids, targets, per_token=True)

    # ---- per-position statistics (detector research) -------------------------- #
    @torch.no_grad()
    def _target_logprobs(self, ctx_ids: list[int], targets: list[list[int]], nmax: int) -> torch.Tensor:
        """Full next-token log-prob rows at every target position: [B, nmax, V] (rows past a text's end are zero)."""
        c = len(ctx_ids)
        B, L = len(targets), c + max(len(t) for t in targets)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for r, t in enumerate(targets):
            seq = ctx_ids + t
            input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[r, :len(seq)] = 1
        logits = self.model(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device), use_cache=False).logits
        out = torch.zeros((B, nmax, logits.shape[-1]), dtype=torch.float32, device=self.device)
        for r, t in enumerate(targets):
            n = len(t)
            out[r, :n] = torch.log_softmax(logits[r, c - 1:c - 1 + n].float(), dim=-1)
        return out

    @torch.no_grad()
    def position_stats(self, ctx_neutral: list[int], ctx_keys: list[list[int]], targets: list[list[int]],
                       token_budget: int = 3072) -> list[dict]:
        """For every text and key, what a text-only detector can know about each position.

        Per text: ``h0`` [n], the neutral world's next-token entropy.  Per text and key ([K, n] each):
        ``d`` -- the realized token's keyed-minus-neutral log-prob (the detector's raw evidence);
        ``mu`` / ``var`` -- mean and variance of that contrast over the neutral world's whole next-token
        distribution, i.e. what the contrast would look like at this position if nobody had steered.
        Everything is a function of the text and the key alone.
        """
        out: list[dict | None] = [None] * len(targets)
        order = [i for i in sorted(range(len(targets)), key=lambda i: len(targets[i])) if targets[i]]
        cmax = max([len(ctx_neutral)] + [len(c) for c in ctx_keys])
        i = 0
        while i < len(order):
            batch = [order[i]]
            longest = cmax + len(targets[order[i]])
            i += 1
            while i < len(order) and max(longest, cmax + len(targets[order[i]])) * (len(batch) + 1) <= token_budget:
                batch.append(order[i]); longest = max(longest, cmax + len(targets[order[i]])); i += 1
            tg = [targets[b] for b in batch]
            nmax = max(len(t) for t in tg)
            idx = torch.zeros((len(tg), nmax), dtype=torch.long, device=self.device)
            for r, t in enumerate(tg):
                idx[r, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
            L0 = self._target_logprobs(ctx_neutral, tg, nmax)
            P0 = L0.exp()
            H0 = -(P0 * L0).sum(-1)
            ds, mus, vs = [], [], []
            for ck in ctx_keys:
                D = self._target_logprobs(ck, tg, nmax).sub_(L0)          # vocabulary-wide contrast at each position
                ds.append(D.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
                PD = P0 * D
                mu = PD.sum(-1)
                mus.append(mu)
                vs.append((PD.mul_(D)).sum(-1) - mu * mu)
                del D, PD
            d, mu, var = torch.stack(ds, 1).cpu().numpy(), torch.stack(mus, 1).cpu().numpy(), torch.stack(vs, 1).cpu().numpy()
            h0 = H0.cpu().numpy()
            for r, b in enumerate(batch):
                n = len(tg[r])
                out[b] = {"h0": h0[r, :n].astype(np.float32), "d": d[r, :, :n].astype(np.float32),
                          "mu": mu[r, :, :n].astype(np.float32), "var": np.maximum(var[r, :, :n], 0.0).astype(np.float32)}
            del L0, P0, H0
        return out  # type: ignore[return-value]

    @torch.no_grad()
    def position_stats_windowed(self, ctx_neutral: list[int], ctx_keys: list[list[int]], targets: list[list[int]],
                                window: int, stride: int = 0, token_budget: int = 8192) -> list[dict]:
        """``position_stats`` under a bounded context: block k (positions k*stride ..) sees tokens from
        max(0, k*stride - window) on -- the rule of the windowed ``Stepper``.  Returns ``h0`` and ``d`` only
        (``mu``/``var`` are zero-filled; the clipped gated statistic does not use them)."""
        stride = int(stride or max(1, window // 2))
        out = [{"h0": np.zeros(len(t), dtype=np.float32), "d": np.zeros((len(ctx_keys), len(t)), dtype=np.float32),
                "mu": np.zeros((len(ctx_keys), len(t)), dtype=np.float32), "var": np.zeros((len(ctx_keys), len(t)), dtype=np.float32)}
               if len(t) >= 2 else None for t in targets]
        groups: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for i, t in enumerate(targets):
            if len(t) < 2:
                continue
            s = 0
            while s < len(t):
                e = min(s + stride, len(t)); a = max(0, s - window)
                groups.setdefault((e - 1 - a, e - s), []).append((i, a, s, e)); s = e

        def realised(ctx, blocks, k, want_entropy):
            ids = torch.tensor([ctx + targets[i][a:e - 1] for i, a, s, e in blocks], device=self.device)
            lp = torch.log_softmax(self.model(input_ids=ids, use_cache=False, logits_to_keep=k).logits.float(), dim=-1)
            tok = torch.tensor([targets[i][s:e] for i, a, s, e in blocks], device=self.device)
            real = lp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            return real, (-(lp.exp() * lp).sum(-1) if want_entropy else None)

        cmax = max([len(ctx_neutral)] + [len(c) for c in ctx_keys])
        for (n_in, k), blocks in groups.items():
            per = max(1, token_budget // (cmax + n_in))
            for b0 in range(0, len(blocks), per):
                bl = blocks[b0:b0 + per]
                r0, h0 = realised(ctx_neutral, bl, k, True)
                for r, (i, a, s, e) in enumerate(bl):
                    out[i]["h0"][s:e] = h0[r].cpu().numpy()
                for j, ck in enumerate(ctx_keys):
                    d = (realised(ck, bl, k, False)[0] - r0).cpu().numpy()
                    for r, (i, a, s, e) in enumerate(bl):
                        out[i]["d"][j, s:e] = d[r]
        return out  # type: ignore[return-value]

    # ---- generator -> reference token map ------------------------------------ #
    def token_map(self, gen_backend, cache_path: Path | None = None, log=print) -> np.ndarray:
        """For every generator token id, the FIRST reference token id of its surface string (-1 when unmappable).

        Frozen on disk per (generator, reference) pair; the mapping is a property of the two tokenizers only.
        """
        if cache_path is not None and Path(cache_path).exists():
            d = np.load(cache_path)
            if int(d["ref_vocab"]) == self.vocab_size and int(d["gen_vocab"]) == len(d["map"]):
                return d["map"]
        gen_tok = getattr(gen_backend, "tokenizer", None)
        n = int(len(gen_tok)) if gen_tok is not None else int(getattr(gen_backend, "VOCAB", getattr(gen_backend, "vocab_size", 0)))
        specials = set(int(x) for x in (getattr(gen_tok, "all_special_ids", None) or [])) if gen_tok is not None else set()
        out = np.full(n, -1, dtype=np.int64)
        step = 8192
        for s0 in range(0, n, step):
            ids = list(range(s0, min(s0 + step, n)))
            if gen_tok is not None and hasattr(gen_tok, "batch_decode"):
                strings = gen_tok.batch_decode([[i] for i in ids], skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False)
            else:
                strings = [gen_backend.decode([i]) for i in ids]
            enc = self.tokenizer(strings, add_special_tokens=False)["input_ids"]
            for i, e in zip(ids, enc):
                if e and i not in specials:
                    out[i] = int(e[0])
            if log:
                log(f"[token-map] {min(s0 + step, n)}/{n}")
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, map=out, ref_vocab=self.vocab_size, gen_vocab=n,
                     meta=json.dumps({"reference": self.info, "generator": getattr(gen_backend, "model_id", "")}))
        return out

    # ---- incremental stepping (generation) ----------------------------------- #
    class Stepper:
        """Keeps the reference model positioned at the end of ``context + text`` and returns next-token log-probs.

        The text is re-tokenized after every committed generator token (tokenizations are not incremental in
        general); the cache is cropped to the longest common reference-token prefix, so usually one or two tokens
        are recomputed per step.
        """

        def __init__(self, rt: "ReferenceRuntime", ctx_ids: list[int], window: int = 0, stride: int = 0):
            self.rt = rt
            self.ctx = list(ctx_ids)
            self.ids: list[int] = []
            self.cache = None
            self.logits: torch.Tensor | None = None
            self.forward_calls = 0
            self.recomputed_tokens = 0
            # bounded context (labelled extension): positions are grouped in blocks of ``stride`` reference tokens; every
            # position of block k sees the text from token max(0, k*stride - window) on.  The key's influence on the
            # reference model decays with the amount of visible text, so a bounded view keeps it at the opening level.
            # ``position_stats_windowed`` applies the identical rule on the detection side.
            self.window, self.stride = int(window), int(stride or max(1, int(window) // 2))
            self.start = 0

        @torch.no_grad()
        def logprobs_after(self, text: str) -> torch.Tensor:
            new = self.rt.encode_text(text) if text else []
            if self.window > 0:
                start = max(0, (len(new) // self.stride) * self.stride - self.window)
                if start != self.start:                     # the window moved: rebuild the state from the new start
                    self.start, self.cache, self.ids, self.logits = start, None, [], None
                new = new[start:]
            if self.cache is None:
                seq = self.ctx + new
                out = self.rt.model(input_ids=torch.tensor([seq], device=self.rt.device), use_cache=True)
                self.cache, self.logits = out.past_key_values, out.logits[0, -1, :].float()
                self.ids = new
                self.forward_calls += 1
                self.recomputed_tokens += len(seq)
                return torch.log_softmax(self.logits, dim=-1)
            L = 0
            while L < min(len(new), len(self.ids)) and new[L] == self.ids[L]:
                L += 1
            if L == len(new) and len(new) == len(self.ids):
                return torch.log_softmax(self.logits, dim=-1)           # text unchanged in reference tokens
            self.cache.crop(len(self.ctx) + L)
            tail = new[L:]
            if not tail:                                                 # text got shorter (rare retokenisation)
                out = self.rt.model(input_ids=torch.tensor([[self.ctx[-1]]], device=self.rt.device),
                                    past_key_values=self.cache, use_cache=True)
            else:
                out = self.rt.model(input_ids=torch.tensor([tail], device=self.rt.device),
                                    past_key_values=self.cache, use_cache=True)
            self.cache, self.logits = out.past_key_values, out.logits[0, -1, :].float()
            self.ids = new
            self.forward_calls += 1
            self.recomputed_tokens += max(1, len(tail))
            return torch.log_softmax(self.logits, dim=-1)

    def stepper(self, ctx_ids: list[int], window: int = 0, stride: int = 0) -> "ReferenceRuntime.Stepper":
        return ReferenceRuntime.Stepper(self, ctx_ids, window=window, stride=stride)


def build_reference(cfg) -> ReferenceRuntime:
    """Construct the reference runtime from the experiment config's ``reference:`` block."""
    r = dict(cfg["reference"]) if "reference" in cfg.raw else {}
    rc = ReferenceConfig(model_id=r.get("id", ReferenceConfig.model_id), revision=r.get("revision"),
                         dtype=r.get("dtype", "float32"), device=r.get("device", "auto"),
                         max_input_tokens=int(r.get("max_input_tokens", 2048)),
                         context_template=r.get("context_template", ReferenceConfig.context_template),
                         context_style=r.get("context_style", "plain"), cache_dir=r.get("cache_dir"))
    return ReferenceRuntime(rc)
