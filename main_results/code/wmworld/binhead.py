"""Bin head (amendment 12): a small classifier on the frozen 3B that predicts where the sentence being written will land.

Input at a position t of a sentence: the 3B's final (normed) hidden state h_t and the input embedding of a candidate
next token v.  Output: logits over REGIONS of the finished sentence's SemStamp embedding under our key:
  0..7  bin b AND at least ``margin`` from every hyperplane (the region the chain's stop rule accepts)
  8     within ``margin`` of some hyperplane ("near a plane")
Steering (Bayes / future discriminator): q(v) ∝ p(v) * P(region in mask | prefix, v); with a calibrated head this is
the model's own sampler conditioned on the finished sentence landing in the mask's clear region.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

N_REGIONS = 9
NEAR = 8


def region_label(proj, margin: float) -> int:
    """Region of a sentence from its 3 hyperplane projections (plane 0 = most significant bit, as Hyperplanes.bin_of)."""
    p = np.asarray(proj, dtype=np.float64)
    if np.min(np.abs(p)) < margin:
        return NEAR
    b = 0
    for x in p:
        b = (b << 1) | int(x > 0.0)
    return b


class BinHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 1024):
        super().__init__()
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_e = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_REGIONS))
        self.d_model, self.hidden = d_model, hidden

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """h: [..., d] hidden states; e: [..., d] next-token embeddings (broadcastable) -> [..., 9] region logits."""
        h, e = torch.broadcast_tensors(self.norm_h(h.float()), self.norm_e(e.float()))
        return self.net(torch.cat([h, e], dim=-1))


def mask_logprob(region_logits: torch.Tensor, mask) -> torch.Tensor:
    """log P(finished sentence lands in the clear region of one of the mask's bins)."""
    lp = torch.log_softmax(region_logits.float(), dim=-1)
    return torch.logsumexp(lp[..., [int(b) for b in mask]], dim=-1)


def save(path: str | Path, head: BinHead, meta: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"d_model": head.d_model, "hidden": head.hidden, "meta": meta, "state": head.state_dict()}, path)


def load(path: str | Path, device="cpu") -> tuple[BinHead, dict]:
    blob = torch.load(path, map_location="cpu")
    head = BinHead(blob["d_model"], blob["hidden"])
    head.load_state_dict(blob["state"])
    return head.to(device).eval(), blob["meta"]


# --------------------------------------------------------------------------- #
# v2: predict the 3 projections themselves (Gaussian per plane), derive region probabilities
# --------------------------------------------------------------------------- #

class ProjHead(nn.Module):
    """Predicts, from (h_t, next-token embedding), a Gaussian over each of the finished sentence's 3 hyperplane
    projections: mean and log-sd per plane, in units of the training projections' sd (``scale``)."""

    def __init__(self, d_model: int, hidden: int = 1024, n_planes: int = 3):
        super().__init__()
        self.norm_h, self.norm_e = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * n_planes))
        self.register_buffer("scale", torch.ones(n_planes))
        self.d_model, self.hidden, self.n_planes = d_model, hidden, n_planes

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (mean, sd) of the projections in raw cosine units, each [..., n_planes]."""
        h, e = torch.broadcast_tensors(self.norm_h(h.float()), self.norm_e(e.float()))
        out = self.net(torch.cat([h, e], dim=-1))
        mu, log_sd = out[..., :self.n_planes], out[..., self.n_planes:].clamp(-6, 3)
        return mu * self.scale, log_sd.exp() * self.scale


def gaussian_nll(mu: torch.Tensor, sd: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-plane Gaussian NLL in standardised units (so planes weigh equally), averaged over planes."""
    z = (y - mu) / sd
    return (0.5 * z ** 2 + torch.log(sd / scale)).mean(dim=-1)


def region_logprobs_from_gaussian(mu: torch.Tensor, sd: torch.Tensor, margin: float) -> torch.Tensor:
    """[..., 9] log P(region): regions 0..7 = bin b clear of ``margin`` on all planes (planes independent given the
    prefix), 8 = near some plane (the rest).  Plane 0 is the most significant bit, as Hyperplanes.bin_of."""
    n = torch.distributions.Normal(mu, sd)
    p_pos = (1 - n.cdf(torch.full_like(mu, margin))).clamp_min(1e-12)      # P(p_j >= m)
    p_neg = n.cdf(torch.full_like(mu, -margin)).clamp_min(1e-12)           # P(p_j <= -m)
    lp = []
    for b in range(8):
        bits = [(b >> (2 - j)) & 1 for j in range(3)]
        lp.append(sum(torch.log(p_pos[..., j] if bits[j] else p_neg[..., j]) for j in range(3)))
    lp = torch.stack(lp, dim=-1)
    clear = torch.logsumexp(lp, dim=-1)
    near = torch.log((1 - clear.exp()).clamp_min(1e-12))
    return torch.cat([lp, near[..., None]], dim=-1)


def save_proj(path: str | Path, head: ProjHead, meta: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "proj", "d_model": head.d_model, "hidden": head.hidden, "meta": meta, "state": head.state_dict()}, path)


def load_proj(path: str | Path, device="cpu") -> tuple[ProjHead, dict]:
    blob = torch.load(path, map_location="cpu")
    head = ProjHead(blob["d_model"], blob["hidden"])
    head.load_state_dict(blob["state"])
    return head.to(device).eval(), blob["meta"]
