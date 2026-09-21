from dataclasses import asdict, dataclass

import torch

from .keys import GENERATOR_SYSTEM, REFERENCE_USER, leak_guard_ids, neutral_system_text


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    min_new_tokens: int = 64
    max_new_tokens: int = 512


@dataclass
class WatermarkConfig:
    beta: float = 4.0
    clip: float = 0.5
    entropy_min: float = 0.2
    entropy_ref: float = 2.0
    steer_eos: bool = False
    block_key_tokens: bool = True
    block_repeat_n: int = 4
    warmup_tokens: int = 0
    contrast_max: float = 0.0
    span_tokens: int = 0

    def to_dict(self):
        return asdict(self)


def normalize_text(text):
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _entropy_gate(h, h_min, h_ref):
    if h <= h_min:
        return 0.0
    return min(1.0, (h - h_min) / max(1e-6, h_ref - h_min))


@torch.no_grad()
def generate_clean(generator, prompt, sampling, seed):
    rng = torch.Generator(device=generator.device)
    rng.manual_seed(int(seed) % (2 ** 63 - 1))
    cache, logits = generator.prefill(generator.chat_ids(GENERATOR_SYSTEM, prompt))
    ids, ended = [], False
    for step in range(sampling.max_new_tokens):
        p = generator.next_token_probs(logits, sampling, suppress_eos=step < sampling.min_new_tokens)
        tok = int(torch.multinomial(p, 1, generator=rng).item())
        if tok in generator.eos_ids:
            ended = True
            break
        ids.append(tok)
        logits = generator.step(tok, cache)
    return {"text": normalize_text(generator.decode(ids)), "token_ids": ids, "ended": ended}


@torch.no_grad()
def generate_watermarked(generator, reference, token_map, prompt, keys, cfg, sampling, seed):
    keys = list(keys) if isinstance(keys, (list, tuple)) else [keys]
    device = generator.device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed) % (2 ** 63 - 1))
    cache, logits = generator.prefill(generator.chat_ids(GENERATOR_SYSTEM, prompt))
    g2r = torch.as_tensor(token_map, dtype=torch.long, device=device)
    eos = torch.tensor(generator.eos_ids, device=device)

    def keyed(k):
        return reference.stepper(reference.chat_ids(k.system_text(), REFERENCE_USER))

    ref_key = keyed(keys[0])
    ref_neutral = reference.stepper(reference.chat_ids(neutral_system_text(), REFERENCE_USER))
    block = leak_guard_ids(generator, keys[0]) if cfg.block_key_tokens else []
    block = torch.tensor(block, dtype=torch.long, device=device)

    ids, text, ended = [], "", False
    ngram_next = {}
    span_starts, key_order = [0], [0]
    trace = {"entropy": [], "beta_t": [], "contrast": [], "kl": []}
    n = cfg.block_repeat_n
    for step in range(sampling.max_new_tokens):
        p_g = generator.next_token_probs(logits, sampling, suppress_eos=step < sampling.min_new_tokens)
        support = p_g > 0
        sup_idx = torch.nonzero(support, as_tuple=False)[:, 0]
        logp_g = torch.log(p_g[support])
        h = float(-(p_g[support] * logp_g).sum().item())

        d_ref = ref_key.logprobs_after(text).to(device) - ref_neutral.logprobs_after(text).to(device)
        d_full = torch.zeros(p_g.shape[0], dtype=torch.float32, device=device)
        ref_idx = g2r[:d_full.shape[0]]
        if ref_idx.shape[0] < d_full.shape[0]:
            pad = torch.full((d_full.shape[0] - ref_idx.shape[0],), -1, dtype=torch.long, device=device)
            ref_idx = torch.cat([ref_idx, pad])
        ok = (ref_idx >= 0) & (ref_idx < d_ref.shape[0])
        d_full[ok] = d_ref[ref_idx[ok]]
        if cfg.contrast_max > 0:
            d_full[d_full > cfg.contrast_max] = 0.0
        if not cfg.steer_eos:
            d_full[eos] = 0.0
        if block.numel():
            d_full[block] = 0.0
        if n >= 2 and len(ids) >= n - 1:
            rep = ngram_next.get(tuple(ids[len(ids) - (n - 1):]))
            if rep:
                d_full[list(rep)] = 0.0
        d = torch.clamp(d_full[support], -cfg.clip, cfg.clip)

        if cfg.span_tokens and step and step % cfg.span_tokens == 0:
            span_starts.append(step)
            if len(keys) > 1:
                idx = (len(span_starts) - 1) % len(keys)
                ref_key = keyed(keys[idx])
                key_order.append(idx)

        beta_t = cfg.beta * _entropy_gate(h, cfg.entropy_min, cfg.entropy_ref)
        if cfg.warmup_tokens and step < cfg.warmup_tokens:
            beta_t *= step / float(cfg.warmup_tokens)

        w = logp_g + beta_t * d
        p = torch.exp(w - w.max())
        p = p / p.sum()
        if not cfg.steer_eos:
            eos_mask = torch.isin(sup_idx, eos)
            if bool(eos_mask.any()):
                p_eos = float(p_g[support][eos_mask].sum().item())
                rest = max(1e-30, float(p[~eos_mask].sum().item()))
                p = torch.where(eos_mask, p_g[support], p * (1.0 - p_eos) / rest)
                p = p / p.sum()

        j = int(torch.multinomial(p, 1, generator=rng).item())
        tok = int(sup_idx[j].item())
        trace["entropy"].append(h)
        trace["beta_t"].append(beta_t)
        trace["contrast"].append(float(d[j].item()))
        trace["kl"].append(float((p * (torch.log(p + 1e-30) - logp_g)).sum().item()))
        if tok in generator.eos_ids:
            ended = True
            break
        if n >= 2 and len(ids) >= n - 1:
            ngram_next.setdefault(tuple(ids[len(ids) - (n - 1):]), set()).add(tok)
        ids.append(tok)
        text = generator.decode(ids)
        logits = generator.step(tok, cache)

    return {"text": normalize_text(generator.decode(ids)), "token_ids": ids, "ended": ended, "trace": trace,
            "span_starts": span_starts, "key_order": key_order, "key_ids": [k.key_id for k in keys]}
