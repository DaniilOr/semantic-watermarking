"""Push head (amendment 15): a small network that reproduces the true-shift push without its 20 encoder calls per token.

The push (semwm.NucleusSteerProcessor, mode "marginal") scores the 20 most likely lexical tokens v of the 3B's top-p
nucleus by the depth, in the mask, of "unit so far + v" (SemStamp encoder; capped at 0.05), standardises those gains
with the model's probabilities as weights and clips them to +-1 sd: z_v.  The head predicts z_v from
  h      the 3B's final hidden state at the step
  e_v    the candidate token's input embedding
  mask   the mask id (previous unit's bin; 8 masks) as a learned embedding
  state  8 numbers about where the unit so far sits (ONE encoder call per step): the partial unit's 3 projections
         signed by each of the mask's 2 bins (6), its current score min(0.05, depth in mask) / 0.05, and its length
Everything after the gains (weighting, clip, entropy gate, repetition guard, mass-preserving tilt) stays the push's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

N_STATE = 8


def state_features(cur_proj: np.ndarray | None, mask, planes, n_tok: int, target_depth: float = 0.05) -> np.ndarray:
    """8 numbers describing the partial unit relative to the mask (zeros for an empty unit)."""
    f = np.zeros(N_STATE, dtype=np.float32)
    if cur_proj is not None:
        for i, b in enumerate(list(mask)[:2]):
            f[3 * i:3 * i + 3] = planes.bin_signs(int(b)) * np.asarray(cur_proj) / 0.05
        depth = max(float(np.min(planes.bin_signs(int(b)) * cur_proj)) for b in mask)
        f[6] = min(target_depth, depth) / 0.05
    f[7] = min(n_tok, 64) / 64.0
    return f


class PushHead(nn.Module):
    def __init__(self, d: int, n_masks: int = 8, hidden: int = 512):
        super().__init__()
        self.norm_h, self.norm_e = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mask_emb = nn.Embedding(n_masks, 64)
        self.net = nn.Sequential(nn.Linear(2 * d + 64 + N_STATE, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.d, self.hidden = d, hidden

    def forward(self, h: torch.Tensor, e: torch.Tensor, mask_id: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """h [N, d], e [N, K, d], mask_id [N], state [N, 8] -> predicted z [N, K]."""
        N, K, _ = e.shape
        ctx = torch.cat([self.norm_h(h.float()), self.mask_emb(mask_id), state.float()], dim=-1)[:, None, :].expand(N, K, -1)
        return self.net(torch.cat([ctx, self.norm_e(e.float())], dim=-1)).squeeze(-1)


def standardise(values: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """The push's step: weighted standardisation over the valid entries of each row, clipped to +-clip (0 where the
    weighted sd vanishes)."""
    w = weights * valid
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    mu = (w * values).sum(-1, keepdim=True)
    sd = torch.sqrt((w * (values - mu) ** 2).sum(-1, keepdim=True))
    z = torch.where(sd > 1e-9, (values - mu) / sd.clamp_min(1e-12), torch.zeros_like(values))
    return z.clamp(-clip, clip) * valid


def save(path: str | Path, head: PushHead, meta: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"d": head.d, "hidden": head.hidden, "meta": meta, "state": head.state_dict()}, path)


def load(path: str | Path, device="cpu") -> tuple[PushHead, dict]:
    blob = torch.load(path, map_location="cpu")
    head = PushHead(blob["d"], hidden=blob["hidden"])
    head.load_state_dict(blob["state"])
    return head.to(device).eval(), blob["meta"]
