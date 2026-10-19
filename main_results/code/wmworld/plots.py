"""Matplotlib figures (spec Section 13).  Missing data is labelled, never filled."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"base": "#6c757d", "budget_control": "#adb5bd", "passive_prefix": "#f4a261", "llm_global": "#2a9d8f",
          "llm_local": "#e76f51", "reference": "#264653"}


def _save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(str(p))
    plt.close(fig)
    return paths


def _missing(ax, msg: str):
    ax.text(0.5, 0.5, f"MISSING: {msg}", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="crimson")
    ax.set_xticks([]); ax.set_yticks([])


def score_distributions(scores_by_arm: dict[str, list[float]], out: Path, title: str, name: str = "score_distributions") -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for arm, vals in scores_by_arm.items():
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        any_data = True
        ax.hist(vals, bins=30, alpha=0.5, label=f"{arm} (n={len(vals)})", color=COLORS.get(arm))
    if not any_data:
        _missing(ax, "no scores")
    ax.set_xlabel("F_k(y) under the assigned key (nats / token)"); ax.set_ylabel("count"); ax.set_title(title); ax.legend(fontsize=8)
    return _save(fig, out, name)


def score_vs_length(points_by_arm: dict[str, list[tuple[int, float]]], out: Path, name: str = "score_vs_length") -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for arm, pts in points_by_arm.items():
        pts = [(n, s) for n, s in pts if s is not None and np.isfinite(s)]
        if not pts:
            continue
        any_data = True
        n, s = zip(*pts)
        ax.scatter(n, s, s=10, alpha=0.6, label=arm, color=COLORS.get(arm))
    if not any_data:
        _missing(ax, "no scores")
    ax.set_xlabel("visible tokens"); ax.set_ylabel("F_k(y)"); ax.set_title("Score versus realized length"); ax.legend(fontsize=8)
    return _save(fig, out, name)


def detection_vs_length(curves: dict[str, list[tuple[int, float, int]]], out: Path, name: str = "detection_vs_length",
                        title: str = "Detection rate versus prefix length (toy calibration)") -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for arm, pts in curves.items():
        if not pts:
            continue
        any_data = True
        L, r, n = zip(*pts)
        ax.plot(L, r, marker="o", label=arm, color=COLORS.get(arm))
        for x, y, k in pts:
            ax.annotate(f"n={k}", (x, y), fontsize=6, textcoords="offset points", xytext=(0, 4))
    if not any_data:
        _missing(ax, "no calibrated detections")
    ax.set_xlabel("prefix length (tokens)"); ax.set_ylabel("flag rate at alpha"); ax.set_ylim(-0.02, 1.02); ax.set_title(title); ax.legend(fontsize=8)
    return _save(fig, out, name)


def per_domain_bars(values: dict[str, dict[str, float]], out: Path, ylabel: str, title: str, name: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4))
    arms = list(values)
    domains = sorted({d for a in arms for d in values[a]})
    if not arms or not domains:
        _missing(ax, "no per-domain data")
    else:
        w = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            xs = np.arange(len(domains)) + i * w
            ax.bar(xs, [values[arm].get(d, np.nan) for d in domains], width=w, label=arm, color=COLORS.get(arm))
        ax.set_xticks(np.arange(len(domains)) + 0.4 - w / 2); ax.set_xticklabels(domains)
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    return _save(fig, out, name)


def confusion_heatmap(matrix: np.ndarray, labels: list[str], out: Path, title: str, name: str = "key_confusion") -> list[str]:
    fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(labels) + 2), max(4, 0.35 * len(labels) + 1.5)))
    if matrix.size == 0:
        _missing(ax, "no marked outputs")
    else:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("best-scoring key"); ax.set_ylabel("key used for marking")
        fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title)
    return _save(fig, out, name)


def tradeoff(points: list[dict], out: Path, name: str = "detection_quality_cost") -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not points:
        _missing(axes[0], "no arms"); _missing(axes[1], "no arms")
    else:
        for p in points:
            axes[0].scatter(p["cost_per_token"], p["detection"], color=COLORS.get(p["arm"]), s=40)
            axes[0].annotate(p["arm"], (p["cost_per_token"], p["detection"]), fontsize=7)
            q = p.get("quality")
            if q is None:
                axes[1].scatter(p["detection"], 0, marker="x", color=COLORS.get(p["arm"]))
                axes[1].annotate(f"{p['arm']} (quality MISSING)", (p["detection"], 0), fontsize=7, rotation=45)
            else:
                axes[1].scatter(p["detection"], q, color=COLORS.get(p["arm"]), s=40)
                axes[1].annotate(p["arm"], (p["detection"], q), fontsize=7)
        axes[0].set_xlabel("scored+proposed tokens per accepted token"); axes[0].set_ylabel("detection (or paired shift)")
        axes[1].set_xlabel("detection"); axes[1].set_ylabel("quality (pending human review)")
    fig.suptitle("Detection / quality / cost")
    return _save(fig, out, name)


def dilution_curves(curves: dict[str, list[tuple[float, float, int]]], out: Path, name: str = "dilution") -> list[str]:
    fig, ax = plt.subplots(figsize=(6, 4))
    if not curves:
        _missing(ax, "dilution attack stage not run")
    else:
        for arm, pts in curves.items():
            x, y, n = zip(*pts)
            ax.plot(x, y, marker="o", label=arm, color=COLORS.get(arm))
        ax.legend(fontsize=8)
    ax.set_xlabel("added unwatermarked length (fraction)"); ax.set_ylabel("detection rate"); ax.set_title("Dilution")
    return _save(fig, out, name)


def paired_shift(shifts_by_arm: dict[str, list[float]], out: Path, name: str = "paired_shift") -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [(a, [v for v in vals if v is not None and np.isfinite(v)]) for a, vals in shifts_by_arm.items()]
    data = [(a, v) for a, v in data if v]
    if not data:
        _missing(ax, "no paired outputs")
    else:
        ax.boxplot([v for _, v in data], showmeans=True)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([f"{a}\n(n={len(v)})" for a, v in data])
        ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("F_k(marked) - F_k(base), same prompt/key"); ax.set_title("Paired score shift versus base")
    return _save(fig, out, name)
