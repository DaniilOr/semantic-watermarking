"""Several LoRA adapters on one frozen model, selected per batch row (bin adapters, amendment 11).

Every wrapped ``nn.Linear`` computes ``base(x) + scale * B[i] A[i] x`` where ``i`` is the adapter index of the row
(``LoRAController.idx``: LongTensor [batch], -1 = off for that row; ``None`` = off for the whole batch).  ``B`` starts
at zero, so a freshly attached model equals the base model exactly.  Only the A/B tensors are trainable and saved.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRAController:
    """Row pattern decided once per ``set_rows`` on the host, so the wrapped layers never synchronise with the GPU:
    ``single`` = the one adapter every row uses (fast path, no gather), ``any_on`` / ``all_on`` for the general path;
    the index tensor is moved to a device once and cached there."""

    def __init__(self):
        self.idx: torch.Tensor | None = None
        self.single: int | None = None
        self.any_on = self.all_on = False
        self._dev: dict = {}

    def set_rows(self, idx) -> None:
        """Adapter index per batch row (list/tensor, -1 = base for that row) or None (base for every row)."""
        self._dev = {}
        if idx is None:
            self.idx, self.single, self.any_on, self.all_on = None, None, False, False
            return
        self.idx = torch.as_tensor(idx, dtype=torch.long).cpu()
        vals = self.idx.tolist()
        self.any_on, self.all_on = any(v >= 0 for v in vals), all(v >= 0 for v in vals)
        self.single = vals[0] if (vals and self.all_on and all(v == vals[0] for v in vals)) else None

    def idx_on(self, device) -> torch.Tensor:
        t = self._dev.get(device)
        if t is None:
            t = self._dev[device] = self.idx.to(device)
        return t


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, ctrl: LoRAController, n: int, r: int, alpha: float):
        super().__init__()
        self.base, self.ctrl, self.n, self.r, self.scale = base, ctrl, int(n), int(r), float(alpha) / float(r)
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(n, r, base.in_features, dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(n, base.out_features, r, dtype=torch.float32, device=dev))
        for i in range(n):                                                     # peft's default init for A
            nn.init.kaiming_uniform_(self.A.data[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        c = self.ctrl
        if c.idx is None or not c.any_on:
            return y
        if c.idx.shape[0] != x.shape[0]:
            raise ValueError(f"adapter rows {c.idx.shape[0]} != batch {x.shape[0]}")
        if c.single is not None:                                              # every row uses the same adapter
            d = (x.float() @ self.A[c.single].t()) @ self.B[c.single].t() * self.scale
            return y + d.to(y.dtype)
        idx = c.idx_on(x.device)
        j = idx.clamp(min=0)
        h = torch.einsum("bti,bri->btr", x.float(), self.A[j])
        d = torch.einsum("btr,bor->bto", h, self.B[j]) * self.scale
        if not c.all_on:
            d = d * (idx >= 0).to(d.dtype)[:, None, None]
        return y + d.to(y.dtype)


def attach(model: nn.Module, n: int, r: int = 16, alpha: float = 32.0, targets=TARGETS) -> LoRAController:
    """Freeze ``model`` and wrap every target linear with ``n`` adapters; returns the shared controller."""
    for p in model.parameters():
        p.requires_grad_(False)
    ctrl = LoRAController()
    todo = [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets]
    for name, mod in todo:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, MultiLoRALinear(mod, ctrl, n, r, alpha))
    model._multilora = {"n": n, "r": r, "alpha": alpha, "targets": list(targets), "n_modules": len(todo)}
    return ctrl


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def trainable(model: nn.Module) -> list[nn.Parameter]:
    return [p for n_, p in model.named_parameters() if n_.endswith(".A") or n_.endswith(".B")]


def save(path: str | Path, model: nn.Module, meta: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model._multilora, "meta": meta, "state": {k: v.detach().cpu() for k, v in lora_state(model).items()}}, path)


def load(path: str | Path, model: nn.Module) -> tuple[LoRAController, dict]:
    """Attach adapters with the saved configuration and load their weights; returns (controller, meta)."""
    blob = torch.load(path, map_location="cpu")
    c = blob["config"]
    ctrl = attach(model, c["n"], c["r"], c["alpha"], tuple(c["targets"]))
    missing = model.load_state_dict(blob["state"], strict=False)
    bad = [k for k in missing.missing_keys if k.endswith(".A") or k.endswith(".B")]
    if bad or missing.unexpected_keys:
        raise RuntimeError(f"adapter load mismatch: missing {bad[:3]} unexpected {missing.unexpected_keys[:3]}")
    return ctrl, blob["meta"]
