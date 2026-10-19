"""Shared Qwen3.5-0.8B runtime (spec Section 3).

One weight load serves every role (generator, keyed world, neutral world,
key-rarity scorer, paraphraser, judge).  Roles differ only in the chat context
they render; caches, recurrent states and RNG streams are never shared between
roles or between candidate branches.

Correctness policy:
* the multimodal checkpoint is loaded into the text-only ``Qwen3_5ForCausalLM``
  and the loading report is asserted (no missing language/lm_head weights);
* the chat template is rendered with ``enable_thinking=False`` and
  ``add_generation_prompt=True``; the fixed empty thinking block is context;
* teacher-forced likelihoods use right padding only (linear-attention layers
  process tokens sequentially, so padding must never precede content) and are
  validated against single-sequence scoring;
* the LM head and log-softmax are computed in float32 in position chunks so the
  full-vocabulary logits (248,320 wide) are never materialized for a batch;
* candidate proposals are sampled by an explicit loop with a private
  ``torch.Generator``; when ``fork_prefix_cache`` is enabled, the accepted-prefix
  cache is deep-copied and expanded with ``reorder_cache`` (attention KV,
  convolution and recurrent states alike); the fork path is gated by an
  equality test against fresh prefill (spec Sections 3 and 5).
"""

from __future__ import annotations

from collections import deque

import copy
import hashlib
import importlib
import math
import platform
import time
from dataclasses import dataclass

import torch

from .backend import BackendInfo, Candidate, SamplingParams

ALLOWED_MODEL_ID = "Qwen/Qwen3.5-0.8B"
GENERATION_PREFIX_TEXT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class RuntimeConfig:
    model_id: str = ALLOWED_MODEL_ID
    revision: str | None = None
    dtype: str = "bfloat16"                 # bfloat16 on CUDA, float32 otherwise
    device: str = "auto"
    attn_implementation: str | None = None   # None -> transformers default (sdpa)
    max_input_tokens: int = 4096
    head_chunk_positions: int = 256
    scoring_token_budget: int = 24576        # max tokens per teacher-forcing forward
    fork_prefix_cache: bool = False          # enable only after the fork equality test passes
    use_kernels: bool = False                # Hub kernels (kernels package); off by default
    kernels: str = "torch"                   # auto | torch | fla  -- DeltaNet implementation policy
    allow_tf32: bool = False                 # TF32 matmul/conv would break float32 batched-vs-single equality


def apply_kernel_policy(policy: str, device_type: str) -> str:
    """Decide whether the optional ``fla`` package may be used and block it otherwise.

    Transformers resolves the Gated DeltaNet implementation at import time by
    trying ``import fla``; blocking the import (``sys.modules['fla'] = None``)
    before the modeling module is imported forces the reference PyTorch path.
    Must run before ``transformers.models.qwen3_5`` is imported.
    """
    import sys
    if policy not in ("auto", "torch", "fla"):
        raise ValueError(f"unknown kernels policy {policy!r}")
    want_fla = (policy == "fla") or (policy == "auto" and device_type == "cuda")
    if not want_fla:
        if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules and sys.modules.get("fla") is not None:
            raise RuntimeError("kernels policy 'torch' requested after the Qwen3.5 modeling module was imported with fla")
        sys.modules["fla"] = None  # type: ignore[assignment]
        return "torch"
    try:
        importlib.import_module("fla")
        return "fla"
    except Exception:
        return "torch"


def _kernel_availability() -> dict:
    out = {}
    for mod in ("fla", "causal_conv1d", "kernels", "flash_attn"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception:
            out[mod] = None
    return out


def running_t(s1: float, s2: float, n: int, min_n: int = 8) -> float:
    """Within-text t-statistic of the accepted tokens' raw contrasts from running sums (0 until ``min_n`` tokens)."""
    if n < min_n:
        return 0.0
    mean = s1 / n
    var = max(0.0, s2 / n - mean * mean) * n / (n - 1)
    sd = max(1e-6, var ** 0.5)
    return mean / (sd / n ** 0.5)


def adaptive_beta(beta: float, adapt: str, t_run: float, t_target: float, beta_floor: float) -> float:
    """adapt='tstat': beta * (floor + (1 - floor) * clip((t_target - t_run) / t_target, 0, 1)) (linear ramp; its
    equilibrium sits below the target); adapt='bang': full beta while t_run < t_target, floor * beta at/above it; else beta."""
    if adapt not in ("tstat", "bang") or t_target <= 0:
        return float(beta)
    if adapt == "bang":
        return float(beta) if t_run < t_target else float(beta) * beta_floor
    m = min(1.0, max(0.0, (t_target - t_run) / t_target))
    return float(beta) * (beta_floor + (1.0 - beta_floor) * m)


def _beta_for_kl_budget(logp_g: torch.Tensor, d: torch.Tensor, beta_max: float, kl_budget: float, iters: int = 12) -> float:
    """Largest beta in [0, beta_max] such that KL(p_beta || p_G) <= kl_budget, where log p_beta ∝ logp_g + beta * d.

    KL is monotone non-decreasing in beta for this exponential tilt, so bisection is exact up to tolerance.
    """
    def kl_at(b: float) -> float:
        w = logp_g + b * d
        w = w - w.max()
        p = torch.exp(w)
        p = p / p.sum()
        return float((p * (torch.log(p + 1e-30) - logp_g)).sum().item())
    if kl_at(beta_max) <= kl_budget:
        return float(beta_max)
    lo, hi = 0.0, float(beta_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kl_at(mid) <= kl_budget:
            lo = mid
        else:
            hi = mid
    return lo


def _beta_for_lp_budget(logp_g: torch.Tensor, d: torch.Tensor, beta_max: float, lp_budget: float, grid: int = 16) -> float:
    """Largest beta on a grid in [0, beta_max] such that E_{p_beta}[log p_G] >= E_{p_G}[log p_G] - lp_budget.

    Directly caps the per-token expected loss of base-model likelihood (the destructiveness proxy), which a KL cap does
    not: the key tilt moves mass toward tokens the base model finds less likely.
    """
    p_g = torch.exp(logp_g)
    ref = float((p_g * logp_g).sum().item())
    best = 0.0
    for i in range(1, grid + 1):
        b = beta_max * i / grid
        w = logp_g + b * d
        w = w - w.max()
        p = torch.exp(w)
        p = p / p.sum()
        if float((p * logp_g).sum().item()) >= ref - lp_budget:
            best = b
        else:
            break
    return best


class QwenRuntime:
    """Backend implementation over the pinned Qwen3.5-0.8B checkpoint."""

    def __init__(self, cfg: RuntimeConfig):
        if cfg.model_id != ALLOWED_MODEL_ID:
            raise ValueError(f"only {ALLOWED_MODEL_ID} is allowed for model-backed roles (got {cfg.model_id})")
        if not cfg.revision or cfg.revision == "RESOLVE_AND_FREEZE":
            raise ValueError("model revision must be an immutable commit sha (resolve it before model-backed runs)")
        self.cfg = cfg
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.deltanet_impl = apply_kernel_policy(cfg.kernels, self.device.type)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        if cfg.dtype == "bfloat16" and self.device.type != "cuda":
            dtype = torch.float32
        else:
            dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[cfg.dtype]
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
        kwargs = dict(revision=cfg.revision, dtype=dtype, output_loading_info=True)
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.use_kernels:
            kwargs["use_kernels"] = True
        model, info = Qwen3_5ForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        self.loading_info = {k: sorted(map(str, v)) if isinstance(v, (list, set, tuple)) else v for k, v in dict(info).items()}
        missing = [k for k in self.loading_info.get("missing_keys", [])]
        if missing:
            raise RuntimeError(f"checkpoint loading left missing weights: {missing[:10]} (total {len(missing)})")
        unexpected = [k for k in self.loading_info.get("unexpected_keys", [])
                      if not (k.startswith("mtp") or ".visual" in k or k.startswith("model.visual"))]
        if unexpected:
            raise RuntimeError(f"unexpected non-vision/non-MTP weights: {unexpected[:10]}")
        mismatched = self.loading_info.get("mismatched_keys", [])
        if mismatched:
            raise RuntimeError(f"mismatched weights: {mismatched[:10]}")
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.config = model.config

        # Special ids: the tokenizer's EOS is <|im_end|>; the text config's is <|endoftext|>.
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        endoftext = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        cfg_eos = getattr(self.config, "eos_token_id", None)
        eos = [im_end, endoftext]
        if isinstance(cfg_eos, int) and cfg_eos not in eos:
            eos.append(cfg_eos)
        self.eos_ids = [int(e) for e in eos if e is not None]
        self.pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else endoftext)
        self.max_input_tokens = cfg.max_input_tokens

        # Template: render once, verify the fixed non-thinking generation prefix, hash the template source.
        template_src = self.tokenizer.chat_template if isinstance(self.tokenizer.chat_template, str) else str(self.tokenizer.chat_template)
        self.template_hash = hashlib.sha256(template_src.encode("utf-8")).hexdigest()
        rendered = self.render_chat("SYSTEM_PROBE", "USER_PROBE")
        if not rendered.endswith(GENERATION_PREFIX_TEXT):
            raise RuntimeError("rendered chat template does not end with the expected non-thinking generation prefix: "
                               f"{rendered[-80:]!r}")
        self.generation_prefix_ids = self.encode_text(GENERATION_PREFIX_TEXT)
        self.rendered_probe = rendered

        # float32 LM head weight (tied to the embedding); ~1 GB on device.
        self._head_w32 = self.model.lm_head.weight.detach().to(torch.float32)

        self._cost = {"scoring_tokens": 0, "scoring_forward_calls": 0, "scored_sequences": 0,
                      "proposal_tokens": 0, "proposal_forward_calls": 0, "fork_hits": 0, "fork_misses": 0}
        self._prefix_cache: tuple[tuple[int, ...], object] | None = None
        # activation steering (labelled extension): per-layer residual-stream offsets applied ONLY while proposals
        # are being sampled (never during scoring).  Hooks are installed once and are inert unless enabled.
        self._steer: dict | None = None
        self._steer_enabled = False
        self._steer_hooks = [layer.register_forward_hook(self._make_steer_hook(i)) for i, layer in enumerate(self.model.model.layers)]

        self.info = BackendInfo(
            name="qwen3_5_runtime", model_id=cfg.model_id, revision=cfg.revision,
            tokenizer_id=cfg.model_id, dtype=str(dtype).replace("torch.", ""), device=str(self.device),
            attn_implementation=getattr(self.config, "_attn_implementation", None) or cfg.attn_implementation,
            kernels=_kernel_availability(), template_hash=self.template_hash,
            generation_prefix_text=GENERATION_PREFIX_TEXT,
            extra={
                "transformers": transformers.__version__, "torch": torch.__version__,
                "cuda": torch.version.cuda, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
                "eos_ids": self.eos_ids, "pad_id": self.pad_id, "vocab_size": int(self.config.vocab_size),
                "layer_types": list(self.config.layer_types),
                "unexpected_keys_ignored": len(self.loading_info.get("unexpected_keys", [])),
                "fork_prefix_cache": cfg.fork_prefix_cache,
                "deltanet_implementation": self.deltanet_impl,
                "kernels_policy": cfg.kernels, "allow_tf32": cfg.allow_tf32,
            },
        )

    # ------------------------------------------------------------------ #
    # tokenization / template
    # ------------------------------------------------------------------ #
    def encode_text(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids: list[int]) -> str:
        ids = [int(i) for i in ids if int(i) not in self.eos_ids]
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def render_chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def chat_prefix_ids(self, system: str, user: str) -> list[int]:
        return self.encode_text(self.render_chat(system, user))

    # ------------------------------------------------------------------ #
    # teacher-forced likelihood
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _target_logprobs(self, hidden: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """hidden: (T, H) states at positions predicting target_ids (T,); returns (T,) float32 log-probs."""
        out = torch.empty(target_ids.shape[0], dtype=torch.float32, device=hidden.device)
        step = self.cfg.head_chunk_positions
        for s in range(0, hidden.shape[0], step):
            h = hidden[s:s + step].to(torch.float32)
            logits = h @ self._head_w32.T                       # (t, V) float32
            lp = torch.log_softmax(logits, dim=-1)
            out[s:s + step] = lp.gather(1, target_ids[s:s + step, None]).squeeze(1)
        return out

    @torch.no_grad()
    def logprob_sums(self, ctx_ids: list[int], targets: list[list[int]]) -> list[float]:
        """Sum of log p(target_i | ctx, target_<i) for each target sequence (float32 accumulations)."""
        if not ctx_ids:
            raise ValueError("context must be non-empty")
        results: list[float] = [0.0] * len(targets)
        order = sorted(range(len(targets)), key=lambda i: len(targets[i]))
        ctx_len = len(ctx_ids)
        i = 0
        while i < len(order):
            # Greedy batch under a token budget (rows are right-padded to the longest in the batch).
            batch = [order[i]]
            longest = ctx_len + len(targets[order[i]])
            i += 1
            while i < len(order):
                cand_len = ctx_len + len(targets[order[i]])
                if max(longest, cand_len) * (len(batch) + 1) > self.cfg.scoring_token_budget:
                    break
                batch.append(order[i])
                longest = max(longest, cand_len)
                i += 1
            self._score_batch(ctx_ids, targets, batch, results)
        return results

    @torch.no_grad()
    def _score_batch(self, ctx_ids, targets, batch, results) -> None:
        B = len(batch)
        lens = [len(ctx_ids) + len(targets[b]) for b in batch]
        L = max(lens)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for r, b in enumerate(batch):
            seq = ctx_ids + targets[b]
            input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[r, :len(seq)] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        out = self.model.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        hidden = out.last_hidden_state                          # (B, L, H)
        c = len(ctx_ids)
        for r, b in enumerate(batch):
            n = len(targets[b])
            if n == 0:
                results[b] = 0.0
                continue
            h = hidden[r, c - 1:c - 1 + n]                      # positions predicting each target token
            tgt = torch.tensor(targets[b], dtype=torch.long, device=self.device)
            lp = self._target_logprobs(h, tgt)
            results[b] = float(lp.double().sum().item())
        self._cost["scoring_tokens"] += int(attn.sum().item())
        self._cost["scoring_forward_calls"] += 1
        self._cost["scored_sequences"] += B

    @torch.no_grad()
    def logprob_tokens(self, ctx_ids: list[int], targets: list[list[int]]) -> list[list[float]]:
        """Per-token log-probs for each target (right-padded batches, float32 head), for token-level detectors."""
        out: list[list[float]] = [[] for _ in targets]
        order = sorted(range(len(targets)), key=lambda i: len(targets[i]))
        c = len(ctx_ids)
        i = 0
        while i < len(order):
            batch = [order[i]]; longest = c + len(targets[order[i]]); i += 1
            while i < len(order) and max(longest, c + len(targets[order[i]])) * (len(batch) + 1) <= self.cfg.scoring_token_budget:
                batch.append(order[i]); longest = max(longest, c + len(targets[order[i]])); i += 1
            B = len(batch); L = max(c + len(targets[b]) for b in batch)
            input_ids = torch.full((B, L), self.pad_id, dtype=torch.long); attn = torch.zeros((B, L), dtype=torch.long)
            for r, b in enumerate(batch):
                seq = ctx_ids + targets[b]; input_ids[r, :len(seq)] = torch.tensor(seq); attn[r, :len(seq)] = 1
            hidden = self.model.model(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device), use_cache=False).last_hidden_state
            for r, b in enumerate(batch):
                n = len(targets[b])
                if n:
                    tgt = torch.tensor(targets[b], dtype=torch.long, device=self.device)
                    out[b] = [float(x) for x in self._target_logprobs(hidden[r, c - 1:c - 1 + n], tgt).tolist()]
            self._cost["scoring_tokens"] += int(attn.sum().item()); self._cost["scoring_forward_calls"] += 1; self._cost["scored_sequences"] += B
        return out

    @torch.no_grad()
    def per_token_logprobs(self, ctx_ids: list[int], target: list[int]) -> list[float]:
        """Single-sequence per-token log-probs (used by the brute-force equality tests)."""
        seq = torch.tensor([ctx_ids + target], dtype=torch.long, device=self.device)
        out = self.model.model(input_ids=seq, use_cache=False)
        c = len(ctx_ids)
        h = out.last_hidden_state[0, c - 1:c - 1 + len(target)]
        tgt = torch.tensor(target, dtype=torch.long, device=self.device)
        return [float(x) for x in self._target_logprobs(h, tgt).tolist()]

    # ------------------------------------------------------------------ #
    # candidate proposals
    # ------------------------------------------------------------------ #
    def _new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.config)

    @torch.no_grad()
    def _prefill(self, ids: list[int], n_rows: int):
        """Prefill ``ids`` for ``n_rows`` identical rows; returns (cache, last-position logits float32 (n_rows, V))."""
        cache = self._new_cache()
        x = torch.tensor([ids], dtype=torch.long, device=self.device).expand(n_rows, -1).contiguous()
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        self._cost["proposal_forward_calls"] += 1
        return out.past_key_values, out.logits[:, -1, :].to(torch.float32)

    @torch.no_grad()
    def _prefill_forked(self, ids: list[int], n_rows: int):
        """Prefill using a deep-copied, batch-expanded cache of a previously remembered prefix when possible.

        The stored state is immutable: every fork deep-copies it, so rejected
        candidates can never contaminate the accepted prefix.
        """
        key = tuple(ids)
        pc = self._prefix_cache
        if pc is not None and len(pc[0]) <= len(key) and key[:len(pc[0])] == pc[0]:
            base_ids, base_cache, base_logits = pc
            cache = copy.deepcopy(base_cache)
            cache.reorder_cache(torch.zeros(n_rows, dtype=torch.long, device=self.device))
            rest = list(key[len(base_ids):])
            self._cost["fork_hits"] += 1
            if not rest:
                return cache, base_logits.expand(n_rows, -1).clone()
            x = torch.tensor([rest], dtype=torch.long, device=self.device).expand(n_rows, -1).contiguous()
            out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            self._cost["proposal_forward_calls"] += 1
            return out.past_key_values, out.logits[:, -1, :].to(torch.float32)
        self._cost["fork_misses"] += 1
        return self._prefill(ids, n_rows)

    @torch.no_grad()
    def remember_prefix(self, ids: list[int]) -> None:
        """Store an immutable batch-1 state for ``ids`` (fork path only).

        If ``ids`` extends the currently stored prefix, only the new tokens are
        processed on a deep copy of the stored cache; otherwise a fresh prefill
        is done.  The stored last-position logits let a fork with no extra
        tokens start sampling immediately.
        """
        if not self.cfg.fork_prefix_cache:
            return
        key = tuple(ids)
        pc = self._prefix_cache
        if pc is not None and key == pc[0]:
            return
        if pc is not None and len(pc[0]) < len(key) and key[:len(pc[0])] == pc[0]:
            cache = copy.deepcopy(pc[1])
            rest = list(key[len(pc[0]):])
            x = torch.tensor([rest], dtype=torch.long, device=self.device)
        else:
            cache = self._new_cache()
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        self._cost["proposal_forward_calls"] += 1
        self._prefix_cache = (key, out.past_key_values, out.logits[:, -1, :].to(torch.float32))

    def forget_prefix(self) -> None:
        self._prefix_cache = None

    @torch.no_grad()
    def sample_units(self, ctx_ids: list[int], n: int, max_new: int, sampling: SamplingParams,
                     seed: int, eos_allowed_from: int) -> list[Candidate]:
        if len(ctx_ids) + max_new > self.max_input_tokens:
            raise ValueError("proposal context exceeds the declared model input limit")
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) % (2 ** 63 - 1))
        self._steer_enabled = self._steer is not None      # steering acts on proposals only; scoring never enables it
        try:
            return self._sample_units_inner(ctx_ids, n, max_new, sampling, gen, eos_allowed_from)
        finally:
            self._steer_enabled = False

    @torch.no_grad()
    def _sample_units_inner(self, ctx_ids, n, max_new, sampling, gen, eos_allowed_from) -> list[Candidate]:
        if self.cfg.fork_prefix_cache:
            cache, logits = self._prefill_forked(ctx_ids, n)
        else:
            cache, logits = self._prefill(ctx_ids, n)
        eos = torch.tensor(self.eos_ids, device=self.device)
        ids = [[] for _ in range(n)]
        ended = [False] * n
        n_gen = [0] * n
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        for step in range(max_new):
            probs = self._sample_probs(logits, sampling, suppress_eos=(step < eos_allowed_from))
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)      # (n,)
            nxt_list = nxt.tolist()
            alive_before = int(alive.sum().item())
            for r in range(n):
                if not alive[r]:
                    continue
                n_gen[r] += 1
                if nxt_list[r] in self.eos_ids:
                    ended[r] = True
                    alive[r] = False
                else:
                    ids[r].append(int(nxt_list[r]))
            self._cost["proposal_tokens"] += alive_before       # one sampled token per row alive at this step
            if not bool(alive.any()) or step == max_new - 1:
                break
            out = self.model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1)
            self._cost["proposal_forward_calls"] += 1
            logits = out.logits[:, -1, :].to(torch.float32)
        del cache
        return [Candidate(token_ids=ids[r], ended=ended[r], n_generated=n_gen[r]) for r in range(n)]

    def _sample_probs(self, logits: torch.Tensor, sp: SamplingParams, suppress_eos: bool) -> torch.Tensor:
        z = logits.clone()
        if suppress_eos:
            z[:, self.eos_ids] = -float("inf")
        if sp.temperature <= 0:
            p = torch.zeros_like(z)
            p.scatter_(1, z.argmax(dim=1, keepdim=True), 1.0)
            return p
        z = z / sp.temperature
        if sp.top_k and 0 < sp.top_k < z.shape[1]:
            kth = torch.topk(z, sp.top_k, dim=1).values[:, -1:]
            z = torch.where(z >= kth, z, torch.full_like(z, -float("inf")))
        p = torch.softmax(z, dim=1)
        if 0 < sp.top_p < 1.0:
            sp_sorted, idx = torch.sort(p, dim=1, descending=True)
            cum = torch.cumsum(sp_sorted, dim=1)
            keep = (cum - sp_sorted) < sp.top_p
            keep[:, 0] = True
            mask = torch.zeros_like(p, dtype=torch.bool).scatter_(1, idx, keep)
            p = torch.where(mask, p, torch.zeros_like(p))
            p = p / p.sum(dim=1, keepdim=True)
        return p

    # ------------------------------------------------------------------ #
    # token-level product-of-experts steering (labelled extension)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_poe(self, ctx_ids: list[int], key_ctx_ids: list[int], neutral_ctx_ids: list[int], max_new: int,
                   sampling: SamplingParams, seed: int, eos_allowed_from: int, beta: float, clip: float,
                   gate: str, entropy_ref: float, entropy_min: float, kl_budget: float = 0.1, steer_eos: bool = True,
                   block_ids: list[int] | None = None, adapt: str = "none", t_target: float = 5.0, beta_floor: float = 0.0,
                   t_window: int = 0, warmup: int = 0, d_max: float = 0.0,
                   repeat_n: int = 0, decoy_ctx_ids: list[list[int]] | None = None, span_tokens: int = 0) -> dict:
        """Sample one response token by token from p_G(t) * exp(beta_t * clip(log p_key(t) - log p_neutral(t))).

        Three contexts advance in lockstep with their own caches: the task-conditioned generator, the keyed world
        and the neutral world (the two worlds see exactly the detector's scoring contexts followed by the response
        tokens).  beta_t = beta * gate(H_t) where H_t is the entropy of the generator's (temperature / top-p filtered)
        distribution: gate = 0 below ``entropy_min``, ramps linearly to 1 at ``entropy_ref`` ("entropy"), or 1 ("none").
        ``block_ids`` (leak/topic guard) get zero contrast.  ``adapt="tstat"`` scales beta by how far the running
        within-text t-statistic of the chosen tokens' raw contrasts is below ``t_target`` (full strength below 0,
        ``beta_floor`` * beta at/above the target): each text is pushed only as hard as its detection needs.
        Returns token ids plus per-token diagnostics (entropy, beta_t, chosen-token contrast, raw contrast, running t).
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) % (2 ** 63 - 1))
        cache_g, lg = self._prefill(ctx_ids, 1)
        cache_k, lk = self._prefill(key_ctx_ids, 1)
        cache_0, l0 = self._prefill(neutral_ctx_ids, 1)
        self._cost["scoring_tokens"] += len(key_ctx_ids) + len(neutral_ctx_ids)
        self._cost["scoring_forward_calls"] += 2
        # decoy-relative contrast (amendment): the push is log p_key - mean_d log p_decoy, which removes the component shared
        # by "any background phrase"; the neutral context is kept for the controller/detector-aligned running t
        dec: list = []
        for c in (decoy_ctx_ids or []):
            cd, ldd = self._prefill(c, 1)
            dec.append([cd, ldd])
            self._cost["scoring_tokens"] += len(c)
            self._cost["scoring_forward_calls"] += 1
        ids: list[int] = []
        ent, betas, contrasts, lp_g, kls, lploss, raw, truns, steer_raw = [], [], [], [], [], [], [], [], []
        ended = False
        block = torch.tensor(sorted(set(int(b) for b in (block_ids or []))), device=self.device, dtype=torch.long)
        s1 = s2 = 0.0                       # running sums of the chosen tokens' raw contrasts (adapt = tstat)
        win: deque[float] = deque()         # ...over the last t_window tokens when a window is set
        span_starts: list[int] = [0]        # controller resets here; the reference contexts never restart
        ngram_next: dict[tuple, set] = {}   # repetition guard: (n-1)-gram -> tokens that followed it earlier in this text
        n_rep_blocked = 0
        for step in range(max_new):
            p_g = self._sample_probs(lg, sampling, suppress_eos=(step < eos_allowed_from))[0]          # (V,), zeros outside nucleus
            support = p_g > 0
            logp_g = torch.log(p_g[support])
            h = float(-(p_g[support] * logp_g).sum().item())
            lpk = torch.log_softmax(lk[0], dim=-1)
            dneutral = lpk - torch.log_softmax(l0[0], dim=-1)
            if dec:
                dfull = lpk - torch.stack([torch.log_softmax(x[1][0], dim=-1) for x in dec]).mean(dim=0)
            else:
                dfull = dneutral
            if not steer_eos:
                dfull[self.eos_ids] = 0.0        # recorded contrast 0 at EOS; its probability is pinned to p_G below
            if block.numel():
                dfull[block] = 0.0               # leak/topic guard: the key's own words are never pushed
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                rep = ngram_next.get(tuple(ids[len(ids) - (repeat_n - 1):]))
                if rep:                          # repetition guard: completing an already-seen n-gram is never pushed
                    dfull[list(rep)] = 0.0
                    n_rep_blocked += len(rep)
            if d_max > 0:                                 # semantic guard (see sample_poe_cross)
                dfull[dfull > d_max] = 0.0
            d = torch.clamp(dfull[support], -clip, clip)
            if span_tokens and step and step % span_tokens == 0:
                s1 = s2 = 0.0; win.clear()         # new span: the controller pushes again (the reference state carries)
                span_starts.append(step)
            t_run = running_t(s1, s2, len(win) if t_window else step - (span_starts[-1] if span_starts else 0))
            beta_eff = adaptive_beta(beta, adapt, t_run, t_target, beta_floor)
            if gate == "entropy":
                g = 0.0 if h <= entropy_min else min(1.0, (h - entropy_min) / max(1e-6, entropy_ref - entropy_min))
                beta_t = beta_eff * g
            elif gate == "kl":
                # KL-constrained: largest beta_t in [0, beta] with KL(p' || p_G) <= kl_budget, by bisection
                beta_t = _beta_for_kl_budget(logp_g, d, beta_eff, kl_budget)
            elif gate == "lp":
                # likelihood-budgeted: largest beta_t whose expected drop in base log-likelihood E_p'[log p_G] - E_p[log p_G]
                # stays within lp_budget nats (= kl_budget field reused as the budget); grid search, robust to non-monotonicity
                beta_t = _beta_for_lp_budget(logp_g, d, beta_eff, kl_budget)
            else:
                beta_t = beta_eff
            if warmup and step < warmup:             # protect the opening: it fixes what the whole answer will say
                beta_t *= step / float(warmup)
            w = logp_g + beta_t * d
            w = w - w.max()
            p = torch.exp(w)
            p = p / p.sum()
            if not steer_eos:
                # keep the stop decision exactly as the generator makes it: p'(EOS) = p_G(EOS), the rest re-weighted
                sup_idx = torch.nonzero(support, as_tuple=False)[:, 0]
                eos_mask = torch.isin(sup_idx, torch.tensor(self.eos_ids, device=self.device))
                if bool(eos_mask.any()):
                    p_eos = float(p_g[support][eos_mask].sum().item())
                    p = torch.where(eos_mask, p_g[support], p * (1.0 - p_eos) / max(1e-30, float(p[~eos_mask].sum().item())))
                    p = p / p.sum()
            kl = float((p * (torch.log(p + 1e-30) - logp_g)).sum().item())
            lp_loss = float(((p_g[support] - p) * logp_g).sum().item())      # E_pG[log pG] - E_p'[log pG]
            j = int(torch.multinomial(p, 1, generator=gen).item())
            tok = int(torch.nonzero(support, as_tuple=False)[j].item())
            r = float(dneutral[tok].item()) if tok not in self.eos_ids else 0.0
            ent.append(h); betas.append(beta_t); contrasts.append(float(d[j].item())); lp_g.append(float(logp_g[j].item())); kls.append(kl); lploss.append(lp_loss)
            raw.append(r); truns.append(t_run); steer_raw.append(float(dfull[tok].item()))
            self._cost["proposal_tokens"] += 1
            if tok in self.eos_ids:
                ended = True
                break
            s1 += r; s2 += r * r
            if t_window:                          # regulate the LOCAL marking rate: drop tokens leaving the window
                win.append(r)
                if len(win) > t_window:
                    o = win.popleft(); s1 -= o; s2 -= o * o
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                ngram_next.setdefault(tuple(ids[len(ids) - (repeat_n - 1):]), set()).add(tok)
            ids.append(tok)
            x = torch.tensor([[tok]], device=self.device)
            og = self.model(input_ids=x, past_key_values=cache_g, use_cache=True, logits_to_keep=1)
            ok = self.model(input_ids=x, past_key_values=cache_k, use_cache=True, logits_to_keep=1)
            o0 = self.model(input_ids=x, past_key_values=cache_0, use_cache=True, logits_to_keep=1)
            lg, lk, l0 = og.logits[:, -1, :].float(), ok.logits[:, -1, :].float(), o0.logits[:, -1, :].float()
            for x_d in dec:
                od = self.model(input_ids=x, past_key_values=x_d[0], use_cache=True, logits_to_keep=1)
                x_d[1] = od.logits[:, -1, :].float()
            self._cost["proposal_forward_calls"] += 1
            self._cost["scoring_forward_calls"] += 2 + len(dec)
            self._cost["scoring_tokens"] += 2 + len(dec)
        del cache_g, cache_k, cache_0, dec
        return {"token_ids": ids, "ended": ended, "entropy": ent, "beta_t": betas, "contrast": contrasts, "logp_g": lp_g, "kl": kls,
                "lp_loss": lploss, "contrast_raw": raw, "t_run": truns, "n_repeat_blocked": n_rep_blocked, "contrast_steer_raw": steer_raw,
                "span_starts": span_starts}

    # ------------------------------------------------------------------ #
    # cross-model steering (labelled extension): the world is a DIFFERENT checkpoint
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_poe_cross(self, ctx_ids: list[int], ref_key, ref_neutral, gen_to_ref, max_new: int,
                         sampling: SamplingParams, seed: int, eos_allowed_from: int, beta: float, clip: float,
                         gate: str, entropy_ref: float, entropy_min: float, steer_eos: bool = True,
                         block_ids: list[int] | None = None, adapt: str = "none", t_target: float = 5.0,
                         beta_floor: float = 0.0, repeat_n: int = 0, ref_decoys: list | None = None, span_tokens: int = 0,
                         t_window: int = 0, warmup: int = 0, d_max: float = 0.0,
                         ref_key_factories: list | None = None) -> dict:
        """Same push as ``sample_poe``, but the keyed and neutral distributions come from a second model.

        ``ref_key`` / ``ref_neutral`` are ``ReferenceRuntime.Stepper`` objects already positioned on their contexts;
        they are advanced with the decoded text, so the reference model uses ITS OWN tokenization throughout.
        ``gen_to_ref`` maps a generator token id to the first reference token id of its surface string (-1 = no push):
        an approximation of `log p_ref(next generator token)` that costs signal, never correctness, because the
        detector recomputes everything from the finished text under the reference model.
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) % (2 ** 63 - 1))
        cache_g, lg = self._prefill(ctx_ids, 1)
        g2r = torch.as_tensor(gen_to_ref, dtype=torch.long, device=self.device)
        # key rotation: span i is steered toward key i mod m.  Only the ACTIVE keyed context is stepped token by token;
        # at a switch the next key's context is built fresh and catches up on the text so far in a single pass.
        rot = list(ref_key_factories or [])
        key_order: list[int] = [0] if rot else []
        ids: list[int] = []
        ent, betas, contrasts, lp_g, kls, lploss, raw, truns = [], [], [], [], [], [], [], []
        ended = False
        block = torch.tensor(sorted(set(int(b) for b in (block_ids or []))), device=self.device, dtype=torch.long)
        ngram_next: dict[tuple, set] = {}
        n_rep_blocked = 0
        n_unmapped = 0
        s1 = s2 = 0.0
        win: deque[float] = deque()
        span_starts: list[int] = [0]
        text = ""
        for step in range(max_new):
            p_g = self._sample_probs(lg, sampling, suppress_eos=(step < eos_allowed_from))[0]
            support = p_g > 0
            logp_g = torch.log(p_g[support])
            h = float(-(p_g[support] * logp_g).sum().item())
            # contrast over the generator vocabulary, via the reference model's next-token distributions
            lp_key = ref_key.logprobs_after(text).to(self.device)
            lp_neu = ref_neutral.logprobs_after(text).to(self.device)
            dneutral = lp_key - lp_neu
            if ref_decoys:
                # decoy-relative push in the REFERENCE model: removes the component shared by any background phrase
                # (and with it much of the generic "this model is not the generator" disagreement)
                dref = lp_key - torch.stack([d.logprobs_after(text).to(self.device) for d in ref_decoys]).mean(dim=0)
            else:
                dref = dneutral
            dfull = torch.zeros(p_g.shape[0], dtype=torch.float32, device=self.device)
            dneu_gen = torch.zeros(p_g.shape[0], dtype=torch.float32, device=self.device)
            ref_idx = g2r[:dfull.shape[0]] if g2r.shape[0] >= dfull.shape[0] else torch.cat(
                [g2r, torch.full((dfull.shape[0] - g2r.shape[0],), -1, dtype=torch.long, device=self.device)])
            ok = (ref_idx >= 0) & (ref_idx < dref.shape[0])
            dfull[ok] = dref[ref_idx[ok]]
            dneu_gen[ok] = dneutral[ref_idx[ok]]          # the detector's own contrast, for the controller and the record
            if d_max > 0:                                 # semantic guard: never push what the key makes MUCH more likely
                dfull[dfull > d_max] = 0.0                # (key-flavoured content words); the mark lives in subtle preferences
            if not steer_eos:
                dfull[self.eos_ids] = 0.0
            if block.numel():
                dfull[block] = 0.0
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                rep = ngram_next.get(tuple(ids[len(ids) - (repeat_n - 1):]))
                if rep:
                    dfull[list(rep)] = 0.0
                    n_rep_blocked += len(rep)
            n_unmapped += int((~ok[support]).sum().item())
            d = torch.clamp(dfull[support], -clip, clip)
            if span_tokens and step and step % span_tokens == 0:
                s1 = s2 = 0.0; win.clear()         # new span: the controller pushes again (the reference state carries)
                span_starts.append(step)
                if rot:
                    idx = (len(span_starts) - 1) % len(rot)     # index of the span just started
                    ref_key = rot[idx]()             # fresh context for this span's key; catches up on the text so far
                    key_order.append(idx)
            t_run = running_t(s1, s2, len(win) if t_window else step - (span_starts[-1] if span_starts else 0))
            beta_eff = adaptive_beta(beta, adapt, t_run, t_target, beta_floor)
            if gate == "entropy":
                g = 0.0 if h <= entropy_min else min(1.0, (h - entropy_min) / max(1e-6, entropy_ref - entropy_min))
                beta_t = beta_eff * g
            else:
                beta_t = beta_eff
            if warmup and step < warmup:             # protect the opening: it fixes what the whole answer will say
                beta_t *= step / float(warmup)
            w = logp_g + beta_t * d
            w = w - w.max()
            p = torch.exp(w)
            p = p / p.sum()
            if not steer_eos:
                sup_idx = torch.nonzero(support, as_tuple=False)[:, 0]
                eos_mask = torch.isin(sup_idx, torch.tensor(self.eos_ids, device=self.device))
                if bool(eos_mask.any()):
                    p_eos = float(p_g[support][eos_mask].sum().item())
                    p = torch.where(eos_mask, p_g[support], p * (1.0 - p_eos) / max(1e-30, float(p[~eos_mask].sum().item())))
                    p = p / p.sum()
            kl = float((p * (torch.log(p + 1e-30) - logp_g)).sum().item())
            lp_loss = float(((p_g[support] - p) * logp_g).sum().item())
            j = int(torch.multinomial(p, 1, generator=gen).item())
            tok = int(torch.nonzero(support, as_tuple=False)[j].item())
            r = float(dneu_gen[tok].item()) if tok not in self.eos_ids else 0.0
            ent.append(h); betas.append(beta_t); contrasts.append(float(d[j].item())); lp_g.append(float(logp_g[j].item()))
            kls.append(kl); lploss.append(lp_loss); raw.append(r); truns.append(t_run)
            self._cost["proposal_tokens"] += 1
            if tok in self.eos_ids:
                ended = True
                break
            s1 += r; s2 += r * r
            if t_window:                          # regulate the LOCAL marking rate: drop tokens leaving the window
                win.append(r)
                if len(win) > t_window:
                    o = win.popleft(); s1 -= o; s2 -= o * o
            if repeat_n >= 2 and len(ids) >= repeat_n - 1:
                ngram_next.setdefault(tuple(ids[len(ids) - (repeat_n - 1):]), set()).add(tok)
            ids.append(tok)
            text = self.decode(ids)
            x = torch.tensor([[tok]], device=self.device)
            og = self.model(input_ids=x, past_key_values=cache_g, use_cache=True, logits_to_keep=1)
            lg = og.logits[:, -1, :].float()
            self._cost["proposal_forward_calls"] += 1
            self._cost["scoring_forward_calls"] += 2
        del cache_g
        return {"token_ids": ids, "ended": ended, "entropy": ent, "beta_t": betas, "contrast": contrasts, "logp_g": lp_g,
                "kl": kls, "lp_loss": lploss, "contrast_raw": raw, "t_run": truns, "n_repeat_blocked": n_rep_blocked,
                "n_unmapped_candidates": n_unmapped,
                "n_ref_decoys": len(ref_decoys or []),
                "ref_forward_calls": ref_key.forward_calls + ref_neutral.forward_calls + sum(d.forward_calls for d in (ref_decoys or [])),
                "ref_recomputed_tokens": ref_key.recomputed_tokens + ref_neutral.recomputed_tokens, "span_starts": span_starts,
                "key_order": key_order, "n_rotation_keys": len(rot)}

    # ------------------------------------------------------------------ #
    # activation steering (labelled extension; proposals only)
    # ------------------------------------------------------------------ #
    def _make_steer_hook(self, layer_idx: int):
        def hook(module, args, output):
            st = self._steer
            if st is None or not self._steer_enabled or layer_idx not in st["layers"]:
                return None
            v = st["vectors"][layer_idx]
            if isinstance(output, tuple):
                return (output[0] + st["alpha"] * v.to(output[0].dtype),) + tuple(output[1:])
            return output + st["alpha"] * v.to(output.dtype)
        return hook

    @torch.no_grad()
    def steering_vectors(self, keyed_system: str, neutral_system: str, user: str) -> dict:
        """Per-layer difference of the residual stream at the last context position: keyed minus neutral context.

        The contexts are the same fixed scoring contexts (thinking disabled); the vector for layer i is the output of
        decoder layer i.  Returned in the model dtype, on device, with per-layer norms for the record.
        """
        outs = []
        for system in (keyed_system, neutral_system):
            ids = self.chat_prefix_ids(system, user)
            o = self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False, output_hidden_states=True)
            outs.append([h[0, -1, :].detach().clone() for h in o.hidden_states[1:]])   # skip the embedding output
        vectors = [k - n for k, n in zip(outs[0], outs[1])]
        return {"vectors": vectors, "norms": [float(v.norm()) for v in vectors],
                "residual_norms": [float(k.norm()) for k in outs[0]], "n_layers": len(vectors)}

    def set_steering(self, vectors: list | None, alpha: float = 0.0, layers: list[int] | None = None) -> None:
        if vectors is None or alpha == 0.0:
            self._steer = None
            return
        layer_ids = set(range(len(vectors))) if layers is None else set(int(i) for i in layers)
        self._steer = {"vectors": vectors, "alpha": float(alpha), "layers": layer_ids}

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def cost_snapshot(self) -> dict:
        d = dict(self._cost)
        if self.device.type == "cuda":
            d["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
        return d

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @torch.no_grad()
    def greedy_continue(self, ctx_ids: list[int], max_new: int) -> list[int]:
        """Greedy decoding helper used by equality tests and the paraphraser's deterministic mode."""
        cache, logits = self._prefill(ctx_ids, 1)
        out_ids: list[int] = []
        for _ in range(max_new):
            nxt = int(torch.argmax(logits[0]).item())
            if nxt in self.eos_ids:
                break
            out_ids.append(nxt)
            out = self.model(input_ids=torch.tensor([[nxt]], device=self.device), past_key_values=cache,
                             use_cache=True, logits_to_keep=1)
            logits = out.logits[:, -1, :].to(torch.float32)
        return out_ids
