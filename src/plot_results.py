"""
Plot results from already-saved artifacts (NO inference, NO retraining).

This script is aligned with the current VQC-ZTI experiment layout.

QNN runs are expected under:
  artifacts/experiments/<run_name>/<split>/<ablation>/seed_<seed>/

Baseline runs are expected under:
  artifacts/experiments/<run_name>/<split>/baseline/<model>/seed_<seed>/

Primary evaluation files:
  QNN:
    <run_dir>/qnn_eval_metrics.json
    <run_dir>/qnn_eval_outputs.npz

  Baseline:
    <run_dir>/baseline_eval_metrics.json
    <run_dir>/baseline_eval_outputs.npz

Optional fallback for QNN only:
  tagged root-level files written by eval_qnn.py, e.g.
    artifacts/qnn_eval_metrics__random_seed42__abl_no_head.json
    artifacts/qnn_eval_outputs__random_seed42__abl_no_head.npz

Outputs are written under:
  artifacts/experiments/<run_name>/plots/

Examples
--------
Aggregate mode:
  python -m src.plot_results
  python -m src.plot_results --mode aggregate
  python -m src.plot_results --mode aggregate --baseline-model extratrees

Single-run mode:
  python -m src.plot_results --mode single --split group_holdout --ablation full_hybrid --seed 42
  python -m src.plot_results --mode single --split random --ablation no_head --seed 43 --baseline-model rf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# MATLAB-like palette and figure styling
MATLAB = {
    "blue":   "#0072BD",
    "orange": "#D95319",
    "yellow": "#EDB120",
    "purple": "#7E2F8E",
    "green":  "#77AC30",
    "cyan":   "#4DBEEE",
    "red":    "#A2142F",
    "gray":   "#7A7A7A",
    "light_gray": "#F2F2F2",
    "grid":   "#BDBDBD",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _apply_reference_style(ax, add_grid: bool = True) -> None:
    """Apply MATLAB-like visual styling to axes."""
    ax.set_facecolor(MATLAB["light_gray"])
    if add_grid:
        ax.grid(True, color=MATLAB["grid"], linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(MATLAB["gray"])


def _series_color(i: int) -> str:
    """Cycle through a MATLAB-like color palette."""
    palette = [
        MATLAB["blue"],
        MATLAB["orange"],
        MATLAB["yellow"],
        MATLAB["purple"],
        MATLAB["green"],
        MATLAB["cyan"],
        MATLAB["red"],
    ]
    return palette[i % len(palette)]


def _caption_below(
    ax,
    label: str,
    text: str,
    y: float = -0.26,
    fontsize: int = 18,
) -> None:
    """
    Add a large serif caption below the subplot, matching the naming style
    from your reference figures.
    """
    ax.text(
        0.5,
        y,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontfamily="serif",
        color="black",
    )


def _best_text_color(val: float, vmax: float) -> str:
    """Choose white/black text depending on heatmap intensity."""
    return "white" if val > 0.45 * vmax else "black"


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _load_json(path: str) -> Dict[str, Any]:
    """Load JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_load_json(path: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON loader."""
    if not path or not os.path.exists(path):
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _safe_np_load(path: str) -> Optional[Dict[str, Any]]:
    """Best-effort NPZ loader."""
    if not path or not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


def _read_yaml_like(path: str) -> Optional[Dict[str, Any]]:
    """Load YAML config if possible."""
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _ensure_dir(path: str) -> None:
    """Create a directory if needed."""
    os.makedirs(path, exist_ok=True)


def _style_axes_border(ax) -> None:
    """Apply a thin black border around axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


def _add_cell_grid(ax, n: int = 2) -> None:
    """Draw inner cell boundaries for confusion matrices."""
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)


def _normalize_split_name(name: str) -> str:
    """Normalize split aliases to canonical names."""
    s = str(name).strip().lower()
    aliases = {
        "random": "random_stratified",
        "random_stratified": "random_stratified",
        "group": "group_holdout",
        "group_holdout": "group_holdout",
        "time": "time_holdout",
        "time_holdout": "time_holdout",
    }
    return aliases.get(s, s)


def _display_split_name(name: str) -> str:
    """Short display name used in some filenames / tags / titles."""
    s = _normalize_split_name(name)
    if s == "random_stratified":
        return "random"
    if s == "group_holdout":
        return "group"
    if s == "time_holdout":
        return "time"
    return s


def _run_tag_candidates(split_name: str, seed: int, ablation: Optional[str]) -> List[str]:
    """
    Build candidate run tags for tagged root-level QNN eval artifacts.

    eval_qnn.py uses display-style split names in tags, e.g.
      random_seed42
      group_seed42__abl_no_head
    """
    split_norm = _normalize_split_name(split_name)
    split_display = _display_split_name(split_norm)

    tags: List[str] = []
    base_names = [split_display, split_norm]

    seen = set()
    for sp in base_names:
        base = f"{sp}_seed{seed}"
        if ablation:
            cand = f"{base}__abl_{ablation}"
            if cand not in seen:
                tags.append(cand)
                seen.add(cand)
        if base not in seen:
            tags.append(base)
            seen.add(base)

    return tags


def _cm_from_metrics(m: Dict[str, Any]) -> np.ndarray:
    """Build a 2x2 confusion matrix from saved metrics JSON."""
    c = m.get("Confusion", {})
    return np.array(
        [
            [c.get("TN", 0), c.get("FP", 0)],
            [c.get("FN", 0), c.get("TP", 0)],
        ],
        dtype=int,
    )


def _norm_cm(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix."""
    denom = np.maximum(cm.sum(axis=1, keepdims=True).astype(float), 1.0)
    return cm.astype(float) / denom


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    """Compute mean and sample std for a list of floats."""
    arr = np.asarray([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

def _clear_all_titles(fig) -> None:
    """
    Remove all top titles from a figure before saving.
    This clears:
    - subplot titles set with ax.set_title(...)
    - figure-level titles set with fig.suptitle(...)
    """
    try:
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
    except Exception:
        pass

    for ax in getattr(fig, "axes", []):
        try:
            ax.set_title("")
        except Exception:
            pass


def _savefig_clean(fig, out_path: str, dpi: int = 300) -> None:
    """
    Save a figure after stripping subplot titles / suptitle.
    Keeps colorbars, axis labels, legends, and captions intact.
    """
    _clear_all_titles(fig)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------
# Run descriptors
# ---------------------------------------------------------------------
@dataclass
class QNNRun:
    split: str
    ablation: str
    seed: int
    run_dir: str


@dataclass
class BaselineRun:
    split: str
    model: str
    seed: int
    run_dir: str


# ---------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------
def _discover_qnn_runs(run_root: str) -> List[QNNRun]:
    """
    Discover QNN runs under:
      <run_root>/<split>/<ablation>/seed_<seed>/
    """
    runs: List[QNNRun] = []
    if not os.path.isdir(run_root):
        return runs

    for split in sorted(os.listdir(run_root)):
        split_dir = os.path.join(run_root, split)
        if not os.path.isdir(split_dir):
            continue

        for ablation in sorted(os.listdir(split_dir)):
            ab_dir = os.path.join(split_dir, ablation)
            if not os.path.isdir(ab_dir):
                continue
            if ablation.lower() == "baseline":
                continue

            for seed_folder in sorted(os.listdir(ab_dir)):
                if not seed_folder.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_folder.split("_", 1)[1])
                except Exception:
                    continue

                run_dir = os.path.join(ab_dir, seed_folder)
                if os.path.isdir(run_dir):
                    runs.append(
                        QNNRun(
                            split=split,
                            ablation=ablation,
                            seed=seed,
                            run_dir=run_dir,
                        )
                    )

    return runs


def _discover_baseline_runs(run_root: str) -> List[BaselineRun]:
    """
    Discover baseline runs under:
      <run_root>/<split>/baseline/<model>/seed_<seed>/
    """
    runs: List[BaselineRun] = []
    if not os.path.isdir(run_root):
        return runs

    for split in sorted(os.listdir(run_root)):
        split_dir = os.path.join(run_root, split)
        if not os.path.isdir(split_dir):
            continue

        base_dir = os.path.join(split_dir, "baseline")
        if not os.path.isdir(base_dir):
            continue

        for model in sorted(os.listdir(base_dir)):
            model_dir = os.path.join(base_dir, model)
            if not os.path.isdir(model_dir):
                continue

            for seed_folder in sorted(os.listdir(model_dir)):
                if not seed_folder.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_folder.split("_", 1)[1])
                except Exception:
                    continue

                run_dir = os.path.join(model_dir, seed_folder)
                if os.path.isdir(run_dir):
                    runs.append(
                        BaselineRun(
                            split=split,
                            model=model,
                            seed=seed,
                            run_dir=run_dir,
                        )
                    )

    return runs


# ---------------------------------------------------------------------
# Artifact lookup
# ---------------------------------------------------------------------
def _qnn_history_path(run: QNNRun) -> str:
    """Training history path for a QNN run."""
    return os.path.join(run.run_dir, "qnn_train_history.json")


def _qnn_eval_paths(run: QNNRun) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve QNN evaluation artifact paths.

    Priority:
      1) per-run files in the run directory
      2) tagged root-level fallback files written by eval_qnn.py

    We intentionally do NOT fall back to generic:
      artifacts/qnn_eval_metrics.json
      artifacts/qnn_eval_outputs.npz

    because those files can be overwritten by unrelated runs.
    """
    local_metrics = os.path.join(run.run_dir, "qnn_eval_metrics.json")
    local_outputs = os.path.join(run.run_dir, "qnn_eval_outputs.npz")
    if os.path.exists(local_metrics) and os.path.exists(local_outputs):
        return local_metrics, local_outputs

    for tag in _run_tag_candidates(run.split, run.seed, run.ablation):
        m = os.path.join("artifacts", f"qnn_eval_metrics__{tag}.json")
        o = os.path.join("artifacts", f"qnn_eval_outputs__{tag}.npz")
        if os.path.exists(m) and os.path.exists(o):
            return m, o

    return None, None


def _baseline_eval_paths(run: BaselineRun) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve baseline evaluation artifact paths.

    Baseline plotting is strict: it uses only the per-run files.
    """
    m = os.path.join(run.run_dir, "baseline_eval_metrics.json")
    o = os.path.join(run.run_dir, "baseline_eval_outputs.npz")
    if os.path.exists(m) and os.path.exists(o):
        return m, o
    return None, None


# ---------------------------------------------------------------------
# Baseline selection helpers
# ---------------------------------------------------------------------
def _select_baseline_models_for_split(
    baselines: List[BaselineRun],
    split: str,
    requested_model: Optional[str],
) -> List[str]:
    """
    Return the baseline model(s) available for a split.

    Behavior:
    - If requested_model is provided, validate it and return [requested_model].
    - Otherwise, return all available models for that split.
    - Returned order prefers stronger tree baselines first so representative
      plots default naturally to ExtraTrees when multiple models exist.
    """
    split_norm = _normalize_split_name(split)
    models = sorted(
        {
            b.model
            for b in baselines
            if _normalize_split_name(b.split) == split_norm
        }
    )

    if not models:
        return []

    if requested_model is not None:
        if requested_model not in models:
            raise FileNotFoundError(
                f"[plot_results] Requested baseline model '{requested_model}' "
                f"not found for split='{split_norm}'. Available: {models}"
            )
        return [requested_model]

    preferred_order = ["extratrees", "rf", "logreg"]
    ordered = [m for m in preferred_order if m in models]
    ordered += [m for m in models if m not in ordered]
    return ordered


def _find_baseline_for(
    baselines: List[BaselineRun],
    split: str,
    seed: int,
    model: Optional[str],
) -> Optional[BaselineRun]:
    """Find an exact baseline run for split/seed/model."""
    split_norm = _normalize_split_name(split)

    for b in baselines:
        if (
            _normalize_split_name(b.split) == split_norm
            and b.seed == seed
            and (model is None or b.model == model)
        ):
            return b
    return None


# ---------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------
def _plot_training_curves(hist: List[Dict[str, Any]], out_dir: str) -> List[str]:
    """
    Generate training-curve plots using a MATLAB-like visual style.

    Outputs:
      - training_loss.png
      - training_loss_zoom.png
      - training_acc.png
      - training_acc_zoom.png
      - train_eval_gap.png
    """
    saved: List[str] = []

    epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(hist)]
    tr_loss = [float(h.get("train_loss", np.nan)) for h in hist]
    ev_loss = [float(h.get("eval_loss", np.nan)) for h in hist]
    tr_acc = [float(h.get("train_acc", np.nan)) for h in hist]
    ev_acc = [float(h.get("eval_acc", np.nan)) for h in hist]

    def _save(fig, ax, path: str) -> str:
        _apply_reference_style(ax)
        _savefig_clean(fig, path, dpi=240)
        plt.close(fig)
        return path

    # Full loss
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        epochs,
        tr_loss,
        color=MATLAB["blue"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        markerfacecolor="none",
        label="Train Loss",
    )
    ax.plot(
        epochs,
        ev_loss,
        color=MATLAB["orange"],
        marker="s",
        markersize=3.8,
        linewidth=1.8,
        markerfacecolor="none",
        label="Eval Loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves (Loss)")
    ax.legend()
    saved.append(_save(fig, ax, os.path.join(out_dir, "training_loss.png")))

    # Zoomed loss
    k0 = max(int(0.3 * len(epochs)), 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        epochs[k0:],
        tr_loss[k0:],
        color=MATLAB["blue"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        markerfacecolor="none",
        label="Train Loss",
    )
    ax.plot(
        epochs[k0:],
        ev_loss[k0:],
        color=MATLAB["orange"],
        marker="s",
        markersize=3.8,
        linewidth=1.8,
        markerfacecolor="none",
        label="Eval Loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Zoom)")
    ax.legend()
    saved.append(_save(fig, ax, os.path.join(out_dir, "training_loss_zoom.png")))

    # Full accuracy
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        epochs,
        tr_acc,
        color=MATLAB["blue"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        markerfacecolor="none",
        label="Train Accuracy",
    )
    ax.plot(
        epochs,
        ev_acc,
        color=MATLAB["orange"],
        marker="s",
        markersize=3.8,
        linewidth=1.8,
        markerfacecolor="none",
        label="Eval Accuracy",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Training Curves (Accuracy)")
    ax.legend()
    saved.append(_save(fig, ax, os.path.join(out_dir, "training_acc.png")))

    # Zoomed accuracy
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        epochs[k0:],
        tr_acc[k0:],
        color=MATLAB["blue"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        markerfacecolor="none",
        label="Train Accuracy",
    )
    ax.plot(
        epochs[k0:],
        ev_acc[k0:],
        color=MATLAB["orange"],
        marker="s",
        markersize=3.8,
        linewidth=1.8,
        markerfacecolor="none",
        label="Eval Accuracy",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Training Accuracy (Zoom)")
    ax.legend()
    saved.append(_save(fig, ax, os.path.join(out_dir, "training_acc_zoom.png")))

    # Train-eval gap
    gap = np.asarray(tr_acc) - np.asarray(ev_acc)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        epochs,
        gap,
        color=MATLAB["green"],
        marker="o",
        markersize=3.8,
        linewidth=1.8,
        markerfacecolor="none",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("TrainAcc - EvalAcc")
    ax.set_title("Train–Eval Accuracy Gap")
    saved.append(_save(fig, ax, os.path.join(out_dir, "train_eval_gap.png")))

    return saved


def _plot_confusion(cm: np.ndarray, title: str, out_path: str, normalize: bool) -> str:
    """Plot a confusion matrix using a MATLAB-like heatmap style."""
    cm_show = _norm_cm(cm) if normalize else cm
    vmax = float(np.max(cm_show)) if np.size(cm_show) else 1.0

    fig, ax = plt.subplots(figsize=(5.3, 4.3))
    im = ax.imshow(cm_show, cmap="viridis", aspect="equal")

    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for (i, j), val in np.ndenumerate(cm_show):
        txt = f"{val:.2f}" if normalize else str(int(val))
        ax.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
            color=_best_text_color(float(val), vmax),
            fontsize=10,
        )

    _apply_reference_style(ax, add_grid=False)
    _add_cell_grid(ax, n=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.8)

    _savefig_clean(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _plot_roc_pr(qnn_m: Dict[str, Any], base_m: Dict[str, Any], out_dir: str) -> List[str]:
    """Plot ROC and PR comparisons in the reference visual style."""
    saved: List[str] = []

    qroc = qnn_m.get("ROC", {})
    broc = base_m.get("ROC", {})

    q_fpr = np.asarray(qroc.get("fpr", []), dtype=float)
    q_tpr = np.asarray(qroc.get("tpr", []), dtype=float)
    b_fpr = np.asarray(broc.get("fpr", []), dtype=float)
    b_tpr = np.asarray(broc.get("tpr", []), dtype=float)

    # ROC
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if len(q_fpr) and len(q_tpr):
        ax.plot(
            q_fpr,
            q_tpr,
            color=MATLAB["blue"],
            linewidth=2.0,
            label=f"QNN (AUC={qnn_m.get('AUC', np.nan):.4f})",
        )
    if len(b_fpr) and len(b_tpr):
        ax.plot(
            b_fpr,
            b_tpr,
            color=MATLAB["orange"],
            linewidth=2.0,
            label=f"Baseline (AUC={base_m.get('AUC', np.nan):.4f})",
        )
    ax.plot([0, 1], [0, 1], "--", color=MATLAB["gray"], linewidth=1.2, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Binary)")
    ax.legend(loc="lower right")
    _apply_reference_style(ax)
    p = os.path.join(out_dir, "roc_comparison.png")
    _savefig_clean(fig, p, dpi=240)
    plt.close(fig)
    saved.append(p)

    # ROC log-FPR
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if len(q_fpr) and len(q_tpr):
        ax.plot(q_fpr, q_tpr, color=MATLAB["blue"], linewidth=2.0, label="QNN")
    if len(b_fpr) and len(b_tpr):
        ax.plot(b_fpr, b_tpr, color=MATLAB["orange"], linewidth=2.0, label="Baseline")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_xlabel("False Positive Rate (log)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (Log-FPR View)")
    ax.legend(loc="lower right")
    _apply_reference_style(ax)
    p = os.path.join(out_dir, "roc_comparison_logfpr.png")
    _savefig_clean(fig, p, dpi=240)
    plt.close(fig)
    saved.append(p)

    # PR
    qpr = qnn_m.get("PR", {})
    bpr = base_m.get("PR", {})

    q_rec = np.asarray(qpr.get("recall", []), dtype=float)
    q_prec = np.asarray(qpr.get("precision", []), dtype=float)
    b_rec = np.asarray(bpr.get("recall", []), dtype=float)
    b_prec = np.asarray(bpr.get("precision", []), dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if len(q_rec) and len(q_prec):
        ax.plot(
            q_rec,
            q_prec,
            color=MATLAB["blue"],
            linewidth=2.0,
            label=f"QNN (AP={qnn_m.get('AP', np.nan):.4f})",
        )
    if len(b_rec) and len(b_prec):
        ax.plot(
            b_rec,
            b_prec,
            color=MATLAB["orange"],
            linewidth=2.0,
            label=f"Baseline (AP={base_m.get('AP', np.nan):.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (Binary)")
    ax.legend(loc="lower left")
    _apply_reference_style(ax)
    p = os.path.join(out_dir, "pr_comparison.png")
    _savefig_clean(fig, p, dpi=240)
    plt.close(fig)
    saved.append(p)

    return saved


def _plot_threshold_sweep(
    outputs_npz: Optional[Dict[str, Any]],
    title: str,
    out_path: str,
) -> Optional[str]:
    """Plot threshold-vs-TPR/FPR in the reference visual style."""
    if outputs_npz is None:
        return None

    y = outputs_npz.get("y_true", None)
    p = outputs_npz.get("p_anom", None)
    if y is None or p is None:
        return None

    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()

    thr = np.linspace(0.0, 1.0, 201)
    tpr_vals = []
    fpr_vals = []
    eps = 1e-12

    for t in thr:
        yp = (p >= t).astype(int)
        tn = int(((y == 0) & (yp == 0)).sum())
        fp = int(((y == 0) & (yp == 1)).sum())
        fn = int(((y == 1) & (yp == 0)).sum())
        tp = int(((y == 1) & (yp == 1)).sum())

        tpr_vals.append(tp / (tp + fn + eps))
        fpr_vals.append(fp / (fp + tn + eps))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        thr,
        tpr_vals,
        color=MATLAB["blue"],
        linewidth=2.0,
        marker="o",
        markevery=20,
        markersize=4,
        markerfacecolor="none",
        label="TPR",
    )
    ax.plot(
        thr,
        fpr_vals,
        color=MATLAB["orange"],
        linewidth=2.0,
        marker="s",
        markevery=20,
        markersize=4,
        markerfacecolor="none",
        label="FPR",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    _apply_reference_style(ax)
    _savefig_clean(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _plot_metric_bars(
    mean_std: Dict[str, Dict[str, Tuple[float, float]]],
    out_path: str,
    title: str,
) -> str:
    """Plot mean ± std bars with a MATLAB-like color combo."""
    metrics = ["AUC", "AP", "Accuracy", "TPR", "FPR"]
    methods = list(mean_std.keys())

    means = np.array(
        [[mean_std[m].get(k, (np.nan, np.nan))[0] for k in metrics] for m in methods],
        dtype=float,
    )
    stds = np.array(
        [[mean_std[m].get(k, (np.nan, np.nan))[1] for k in metrics] for m in methods],
        dtype=float,
    )

    x = np.arange(len(metrics))
    width = 0.78 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for i, method in enumerate(methods):
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            means[i],
            yerr=stds[i],
            width=width,
            label=method,
            color=_series_color(i),
            edgecolor="black",
            linewidth=0.6,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    _apply_reference_style(ax)
    _savefig_clean(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _plot_seed_stability(
    points: List[Tuple[int, float]],
    out_path: str,
    title: str,
    ylabel: str,
) -> str:
    """Plot metric value versus seed using the reference line style."""
    points = sorted(points, key=lambda x: x[0])
    seeds = [p[0] for p in points]
    vals = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(
        seeds,
        vals,
        color=MATLAB["blue"],
        linewidth=2.0,
        marker="o",
        markersize=5,
        markerfacecolor="none",
    )
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    _apply_reference_style(ax)
    _savefig_clean(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _plot_combined_onecol(
    hist: Optional[List[Dict[str, Any]]],
    qnn_m: Dict[str, Any],
    base_m: Dict[str, Any],
    out_path: str,
) -> str:
    """
    Create a 2x4 composite figure using only already-existing plot types:

      Row 1: (a) Training Loss, (b) Training Accuracy,
             (c) Train-Eval Gap, (d) Metrics Comparison

      Row 2: (e) ROC Comparison, (f) PR Comparison,
             (g) QNN Confusion, (h) Baseline Confusion

    Notes:
    - No top titles on any panel
    - Large serif captions only below each subplot
    - Confusion matrices keep their colorbars
    """
    qroc = qnn_m.get("ROC", {})
    broc = base_m.get("ROC", {})

    qpr = qnn_m.get("PR", {})
    bpr = base_m.get("PR", {})

    q_fpr = np.asarray(qroc.get("fpr", []), dtype=float)
    q_tpr = np.asarray(qroc.get("tpr", []), dtype=float)
    b_fpr = np.asarray(broc.get("fpr", []), dtype=float)
    b_tpr = np.asarray(broc.get("tpr", []), dtype=float)

    q_prec = np.asarray(qpr.get("precision", []), dtype=float)
    q_rec = np.asarray(qpr.get("recall", []), dtype=float)
    b_prec = np.asarray(bpr.get("precision", []), dtype=float)
    b_rec = np.asarray(bpr.get("recall", []), dtype=float)

    cm_q = _cm_from_metrics(qnn_m).astype(float)
    cm_b = _cm_from_metrics(base_m).astype(float)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 9.5))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    # ------------------------------------------------------------
    # (a) Training loss
    # ------------------------------------------------------------
    ax = axes[0]
    if hist:
        ep = [int(h.get("epoch", i + 1)) for i, h in enumerate(hist)]
        train_loss = [float(h.get("train_loss", np.nan)) for h in hist]
        eval_loss = [float(h.get("eval_loss", np.nan)) for h in hist]

        ax.plot(
            ep, train_loss,
            color=MATLAB["blue"], linewidth=1.9,
            marker="o", markersize=4.0, markerfacecolor="none",
            label="Train",
        )
        ax.plot(
            ep, eval_loss,
            color=MATLAB["orange"], linewidth=1.9,
            marker="s", markersize=3.6, markerfacecolor="none",
            label="Eval",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        _apply_reference_style(ax)
    else:
        ax.axis("off")
    _caption_below(ax, "a", "Training Loss Convergence", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (b) Training accuracy
    # ------------------------------------------------------------
    ax = axes[1]
    if hist:
        ep = [int(h.get("epoch", i + 1)) for i, h in enumerate(hist)]
        train_acc = [float(h.get("train_acc", np.nan)) for h in hist]
        eval_acc = [float(h.get("eval_acc", np.nan)) for h in hist]

        ax.plot(
            ep, train_acc,
            color=MATLAB["blue"], linewidth=1.9,
            marker="o", markersize=4.0, markerfacecolor="none",
            label="Train",
        )
        ax.plot(
            ep, eval_acc,
            color=MATLAB["orange"], linewidth=1.9,
            marker="s", markersize=3.6, markerfacecolor="none",
            label="Eval",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=9)
        _apply_reference_style(ax)
    else:
        ax.axis("off")
    _caption_below(ax, "b", "Training Accuracy Progression", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (c) Train-eval gap
    # ------------------------------------------------------------
    ax = axes[2]
    if hist:
        ep = [int(h.get("epoch", i + 1)) for i, h in enumerate(hist)]
        train_acc = np.asarray([float(h.get("train_acc", np.nan)) for h in hist], dtype=float)
        eval_acc = np.asarray([float(h.get("eval_acc", np.nan)) for h in hist], dtype=float)
        gap = train_acc - eval_acc

        ax.plot(
            ep, gap,
            color=MATLAB["green"], linewidth=1.9,
            marker="d", markersize=3.8, markerfacecolor="none",
        )
        ax.axhline(0.0, color=MATLAB["gray"], linestyle="--", linewidth=1.1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gap")
        _apply_reference_style(ax)
    else:
        ax.axis("off")
    _caption_below(ax, "c", "Train-Eval Gap", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (d) Performance metrics comparison
    # ------------------------------------------------------------
    ax = axes[3]
    labels = ["AUC", "AP", "Acc", "TPR", "FPR"]
    q_vals = [
        qnn_m.get("AUC", np.nan),
        qnn_m.get("AP", np.nan),
        qnn_m.get("Accuracy", np.nan),
        qnn_m.get("TPR", np.nan),
        qnn_m.get("FPR", np.nan),
    ]
    b_vals = [
        base_m.get("AUC", np.nan),
        base_m.get("AP", np.nan),
        base_m.get("Accuracy", np.nan),
        base_m.get("TPR", np.nan),
        base_m.get("FPR", np.nan),
    ]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(
        x - w / 2, q_vals, width=w,
        label="QNN", color=MATLAB["blue"],
        edgecolor="black", linewidth=0.6
    )
    ax.bar(
        x + w / 2, b_vals, width=w,
        label="Baseline", color=MATLAB["orange"],
        edgecolor="black", linewidth=0.6
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=9)
    _apply_reference_style(ax)
    _caption_below(ax, "d", "Performance Metrics Comparison", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (e) ROC comparison
    # ------------------------------------------------------------
    ax = axes[4]
    if len(q_fpr) and len(q_tpr):
        ax.plot(
            q_fpr, q_tpr,
            color=MATLAB["blue"], linewidth=2.0,
            label=f"QNN (AUC={qnn_m.get('AUC', np.nan):.3f})",
        )
    if len(b_fpr) and len(b_tpr):
        ax.plot(
            b_fpr, b_tpr,
            color=MATLAB["orange"], linewidth=2.0,
            label=f"Baseline (AUC={base_m.get('AUC', np.nan):.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color=MATLAB["gray"], linewidth=1.1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    _apply_reference_style(ax)
    _caption_below(ax, "e", "ROC Comparison Across Methods", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (f) PR comparison
    # ------------------------------------------------------------
    ax = axes[5]
    if len(q_rec) and len(q_prec):
        ax.plot(
            q_rec, q_prec,
            color=MATLAB["blue"], linewidth=2.0,
            label=f"QNN (AP={qnn_m.get('AP', np.nan):.3f})",
        )
    if len(b_rec) and len(b_prec):
        ax.plot(
            b_rec, b_prec,
            color=MATLAB["orange"], linewidth=2.0,
            label=f"Baseline (AP={base_m.get('AP', np.nan):.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    _apply_reference_style(ax)
    _caption_below(ax, "f", "Precision-Recall Comparison", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (g) QNN confusion matrix (with colorbar)
    # ------------------------------------------------------------
    ax = axes[6]
    im_q = ax.imshow(cm_q, cmap="viridis")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    for (i, j), v in np.ndenumerate(cm_q):
        ax.text(
            j, i, str(int(v)),
            ha="center", va="center",
            color=_best_text_color(float(v), float(cm_q.max())),
            fontsize=11,
        )
    _apply_reference_style(ax, add_grid=False)
    _add_cell_grid(ax, n=2)
    fig.colorbar(im_q, ax=ax, fraction=0.046, pad=0.04)
    _caption_below(ax, "g", "QNN Confusion Matrix", y=-0.22, fontsize=18)

    # ------------------------------------------------------------
    # (h) Baseline confusion matrix (with colorbar)
    # ------------------------------------------------------------
    ax = axes[7]
    im_b = ax.imshow(cm_b, cmap="viridis")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    for (i, j), v in np.ndenumerate(cm_b):
        ax.text(
            j, i, str(int(v)),
            ha="center", va="center",
            color=_best_text_color(float(v), float(cm_b.max())),
            fontsize=11,
        )
    _apply_reference_style(ax, add_grid=False)
    _add_cell_grid(ax, n=2)
    fig.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
    _caption_below(ax, "h", "Baseline Confusion Matrix", y=-0.22, fontsize=18)

    fig.tight_layout(h_pad=4.0, w_pad=2.8)
    _savefig_clean(fig, out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------
def _collect_run_metrics_qnn(run: QNNRun) -> Optional[Dict[str, Any]]:
    """Load QNN metrics for one run."""
    m_path, _ = _qnn_eval_paths(run)
    if not m_path:
        return None
    return _try_load_json(m_path)


def _collect_run_metrics_baseline(run: BaselineRun) -> Optional[Dict[str, Any]]:
    """Load baseline metrics for one run."""
    m_path, _ = _baseline_eval_paths(run)
    if not m_path:
        return None
    return _try_load_json(m_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cesnet_vqczti.yaml")
    parser.add_argument("--mode", choices=["aggregate", "single"], default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Baseline model name to use when multiple baseline models exist.",
    )
    args = parser.parse_args()

    cfg = _read_yaml_like(args.config) or {}
    exp = cfg.get("experiment", {}) or {}

    run_name = str(exp.get("run_name", "run"))
    exp_out_dir = str(exp.get("out_dir", os.path.join("artifacts", "experiments")))
    run_root = os.path.join(exp_out_dir, run_name)

    if args.mode is None:
        args.mode = "aggregate" if bool(exp.get("enabled", False)) else "single"

    qnn_runs = _discover_qnn_runs(run_root)
    baseline_runs = _discover_baseline_runs(run_root)

    if not qnn_runs:
        raise FileNotFoundError(
            f"[plot_results] No QNN runs found under: {run_root}\n"
            f"Did you run: python -m src.train_qnn ?"
        )

    plots_root = os.path.join(run_root, "plots")
    _ensure_dir(plots_root)
    saved_all: List[str] = []

    # -----------------------------------------------------------------
    # Single-run mode
    # -----------------------------------------------------------------
    if args.mode == "single":
        if args.split is None or args.ablation is None or args.seed is None:
            raise ValueError(
                "[plot_results] single mode requires --split, --ablation, and --seed"
            )

        split = _normalize_split_name(args.split)
        ablation = str(args.ablation)
        seed = int(args.seed)

        match = None
        for r in qnn_runs:
            if (
                _normalize_split_name(r.split) == split
                and r.ablation == ablation
                and r.seed == seed
            ):
                match = r
                break

        if match is None:
            raise FileNotFoundError(
                f"[plot_results] Could not find run split={split} ablation={ablation} "
                f"seed={seed} under {run_root}"
            )

        baseline_models = _select_baseline_models_for_split(
            baseline_runs,
            split=split,
            requested_model=args.baseline_model,
        )
        if not baseline_models:
            raise FileNotFoundError(
                f"[plot_results] No baseline run found for split='{split}'. "
                f"Run: python -m src.baseline_train_eval"
            )

        # In single mode, if multiple baseline models exist and the user did not
        # explicitly request one, use the first preferred one.
        baseline_model = baseline_models[0]

        b_run = _find_baseline_for(
            baseline_runs,
            split=split,
            seed=seed,
            model=baseline_model,
        )
        if b_run is None:
            raise FileNotFoundError(
                f"[plot_results] Missing baseline run for split='{split}', seed={seed}, "
                f"model='{baseline_model}'."
            )

        out_dir = os.path.join(plots_root, split, ablation, f"seed_{seed}")
        _ensure_dir(out_dir)

        # Training history
        hist_obj = _try_load_json(_qnn_history_path(match))
        hist_list = hist_obj if isinstance(hist_obj, list) else []
        if hist_list:
            print("[plot_results] Step 1/6: Training curves...")
            saved_all += _plot_training_curves(hist_list, out_dir)
        else:
            print("[plot_results] Training history missing; skipping training plots.")

        # QNN eval
        qnn_m_path, qnn_o_path = _qnn_eval_paths(match)
        if not qnn_m_path:
            raise FileNotFoundError(
                "[plot_results] Missing QNN eval metrics for this run. "
                "Run: python -m src.eval_qnn ..."
            )

        qnn_m = _load_json(qnn_m_path)
        qnn_o = _safe_np_load(qnn_o_path)

        # Baseline eval
        b_m_path, b_o_path = _baseline_eval_paths(b_run)
        if not b_m_path:
            raise FileNotFoundError(
                f"[plot_results] Missing baseline eval files for split='{split}', "
                f"seed={seed}, model='{baseline_model}'."
            )

        base_m = _load_json(b_m_path)
        base_o = _safe_np_load(b_o_path)

        print("[plot_results] Step 2/6: ROC + PR...")
        saved_all += _plot_roc_pr(qnn_m, base_m, out_dir)

        print("[plot_results] Step 3/6: Threshold sweeps...")
        p = _plot_threshold_sweep(
            qnn_o,
            "Threshold Sweep (QNN)",
            os.path.join(out_dir, "threshold_sweep_qnn.png"),
        )
        if p:
            saved_all.append(p)

        p = _plot_threshold_sweep(
            base_o,
            "Threshold Sweep (Baseline)",
            os.path.join(out_dir, "threshold_sweep_baseline.png"),
        )
        if p:
            saved_all.append(p)

        print("[plot_results] Step 4/6: Confusion matrices...")
        cm_q = _cm_from_metrics(qnn_m)
        cm_b = _cm_from_metrics(base_m)
        saved_all.append(
            _plot_confusion(
                cm_q,
                "QNN Confusion Matrix (Counts)",
                os.path.join(out_dir, "confusion_qnn.png"),
                normalize=False,
            )
        )
        saved_all.append(
            _plot_confusion(
                cm_q,
                "QNN Confusion Matrix (Normalized)",
                os.path.join(out_dir, "confusion_qnn_norm.png"),
                normalize=True,
            )
        )
        saved_all.append(
            _plot_confusion(
                cm_b,
                "Baseline Confusion Matrix (Counts)",
                os.path.join(out_dir, "confusion_baseline.png"),
                normalize=False,
            )
        )
        saved_all.append(
            _plot_confusion(
                cm_b,
                "Baseline Confusion Matrix (Normalized)",
                os.path.join(out_dir, "confusion_baseline_norm.png"),
                normalize=True,
            )
        )

        print("[plot_results] Step 5/6: Combined one-column report figure...")
        representative_name = f"{split}_seed{seed}_{ablation}_{baseline_model}"
        representative_name = representative_name.replace("\\", "_").replace("/", "_").replace(" ", "_")
        saved_all.append(
            _plot_combined_onecol(
                hist_list,
                qnn_m,
                base_m,
                os.path.join(
                    rep_dir,
                    f"combine_fig_col_{representative_name}.png",
                ),
            )
        )

        print("[plot_results] Step 6/6: Done.")
        print(f"[plot_results] Saved plots to: {out_dir}")

    # -----------------------------------------------------------------
    # Aggregate mode
    # -----------------------------------------------------------------
    else:
        splits = sorted({_normalize_split_name(r.split) for r in qnn_runs})
        ablations = sorted({r.ablation for r in qnn_runs})

        print("[plot_results] Aggregate mode enabled.")
        print(f"[plot_results] Found splits: {splits}")
        print(f"[plot_results] Found ablations: {ablations}")

        summary_rows: List[Dict[str, Any]] = []

        for split in splits:
            split_out_dir = os.path.join(plots_root, split)
            _ensure_dir(split_out_dir)

            mean_std: Dict[str, Dict[str, Tuple[float, float]]] = {}

            # -------------------------
            # QNN aggregate metrics
            # -------------------------
            for ablation in ablations:
                ms = []
                for r in qnn_runs:
                    if _normalize_split_name(r.split) == split and r.ablation == ablation:
                        m = _collect_run_metrics_qnn(r)
                        if m is not None:
                            ms.append(m)

                if not ms:
                    continue

                aucs = [float(m.get("AUC", np.nan)) for m in ms]
                aps = [float(m.get("AP", np.nan)) for m in ms]
                accs = [float(m.get("Accuracy", np.nan)) for m in ms]
                tprs = [float(m.get("TPR", np.nan)) for m in ms]
                fprs = [float(m.get("FPR", np.nan)) for m in ms]

                mean_std[ablation] = {
                    "AUC": _mean_std(aucs),
                    "AP": _mean_std(aps),
                    "Accuracy": _mean_std(accs),
                    "TPR": _mean_std(tprs),
                    "FPR": _mean_std(fprs),
                }

                summary_rows.append(
                    {
                        "split": split,
                        "method": f"QNN::{ablation}",
                        "AUC_mean": mean_std[ablation]["AUC"][0],
                        "AUC_std": mean_std[ablation]["AUC"][1],
                        "AP_mean": mean_std[ablation]["AP"][0],
                        "AP_std": mean_std[ablation]["AP"][1],
                        "Acc_mean": mean_std[ablation]["Accuracy"][0],
                        "Acc_std": mean_std[ablation]["Accuracy"][1],
                        "TPR_mean": mean_std[ablation]["TPR"][0],
                        "TPR_std": mean_std[ablation]["TPR"][1],
                        "FPR_mean": mean_std[ablation]["FPR"][0],
                        "FPR_std": mean_std[ablation]["FPR"][1],
                        "n_runs": len(ms),
                    }
                )

            # -------------------------
            # Baseline aggregate metrics
            # -------------------------
            baseline_models = _select_baseline_models_for_split(
                baseline_runs,
                split=split,
                requested_model=args.baseline_model,
            )

            for baseline_model in baseline_models:
                base_ms = []
                for b in baseline_runs:
                    if _normalize_split_name(b.split) == split and b.model == baseline_model:
                        m = _collect_run_metrics_baseline(b)
                        if m is not None:
                            base_ms.append(m)

                if not base_ms:
                    continue

                aucs = [float(m.get("AUC", np.nan)) for m in base_ms]
                aps = [float(m.get("AP", np.nan)) for m in base_ms]
                accs = [float(m.get("Accuracy", np.nan)) for m in base_ms]
                tprs = [float(m.get("TPR", np.nan)) for m in base_ms]
                fprs = [float(m.get("FPR", np.nan)) for m in base_ms]

                key = f"BASE::{baseline_model}"
                mean_std[key] = {
                    "AUC": _mean_std(aucs),
                    "AP": _mean_std(aps),
                    "Accuracy": _mean_std(accs),
                    "TPR": _mean_std(tprs),
                    "FPR": _mean_std(fprs),
                }

                summary_rows.append(
                    {
                        "split": split,
                        "method": key,
                        "AUC_mean": mean_std[key]["AUC"][0],
                        "AUC_std": mean_std[key]["AUC"][1],
                        "AP_mean": mean_std[key]["AP"][0],
                        "AP_std": mean_std[key]["AP"][1],
                        "Acc_mean": mean_std[key]["Accuracy"][0],
                        "Acc_std": mean_std[key]["Accuracy"][1],
                        "TPR_mean": mean_std[key]["TPR"][0],
                        "TPR_std": mean_std[key]["TPR"][1],
                        "FPR_mean": mean_std[key]["FPR"][0],
                        "FPR_std": mean_std[key]["FPR"][1],
                        "n_runs": len(base_ms),
                    }
                )

            # Mean/std metric bar plot
            if mean_std:
                print(f"[plot_results] Writing aggregate metric bars for split='{split}' ...")
                saved_all.append(
                    _plot_metric_bars(
                        mean_std,
                        out_path=os.path.join(split_out_dir, "metric_bar_mean_std.png"),
                        title=f"Mean ± Std Metrics (split={split})",
                    )
                )

            # Seed stability for representative QNN ablation
            rep_ab = "full_hybrid" if "full_hybrid" in ablations else (ablations[0] if ablations else None)
            if rep_ab:
                pts_auc: List[Tuple[int, float]] = []
                pts_acc: List[Tuple[int, float]] = []
                pts_tpr: List[Tuple[int, float]] = []

                for r in qnn_runs:
                    if _normalize_split_name(r.split) == split and r.ablation == rep_ab:
                        m = _collect_run_metrics_qnn(r)
                        if m is not None:
                            pts_auc.append((r.seed, float(m.get("AUC", np.nan))))
                            pts_acc.append((r.seed, float(m.get("Accuracy", np.nan))))
                            pts_tpr.append((r.seed, float(m.get("TPR", np.nan))))

                seed_note = " (same split across seeds)" if split == "time_holdout" else ""

                if len(pts_auc) >= 2:
                    saved_all.append(
                        _plot_seed_stability(
                            pts_auc,
                            os.path.join(split_out_dir, "seed_stability_auc.png"),
                            f"Seed Stability (AUC) — {rep_ab}{seed_note}",
                            "AUC",
                        )
                    )
                if len(pts_acc) >= 2:
                    saved_all.append(
                        _plot_seed_stability(
                            pts_acc,
                            os.path.join(split_out_dir, "seed_stability_acc.png"),
                            f"Seed Stability (Accuracy) — {rep_ab}{seed_note}",
                            "Accuracy",
                        )
                    )
                if len(pts_tpr) >= 2:
                    saved_all.append(
                        _plot_seed_stability(
                            pts_tpr,
                            os.path.join(split_out_dir, "seed_stability_tpr.png"),
                            f"Seed Stability (TPR) — {rep_ab}{seed_note}",
                            "TPR",
                        )
                    )

            # Representative run plots
            rep_run = None
            if rep_ab:
                candidates = [
                    r
                    for r in qnn_runs
                    if _normalize_split_name(r.split) == split and r.ablation == rep_ab
                ]
                scored: List[Tuple[float, QNNRun]] = []
                for r in candidates:
                    m = _collect_run_metrics_qnn(r)
                    if m is not None and "AUC" in m:
                        val = float(m.get("AUC", np.nan))
                        if not np.isnan(val):
                            scored.append((val, r))

                if scored:
                    scored.sort(key=lambda x: x[0])
                    rep_run = scored[len(scored) // 2][1]
                elif candidates:
                    rep_run = candidates[0]

            if rep_run is not None and baseline_models:
                for baseline_model in baseline_models:
                    b_run = _find_baseline_for(
                        baseline_runs,
                        split=split,
                        seed=rep_run.seed,
                        model=baseline_model,
                    )

                    if b_run is None:
                        continue

                    qnn_m_path, qnn_o_path = _qnn_eval_paths(rep_run)
                    b_m_path, b_o_path = _baseline_eval_paths(b_run)

                    if qnn_m_path and b_m_path:
                        qnn_m = _load_json(qnn_m_path)
                        base_m = _load_json(b_m_path)
                        qnn_o = _safe_np_load(qnn_o_path)
                        base_o = _safe_np_load(b_o_path)

                        rep_dir = os.path.join(split_out_dir, "representative", baseline_model)
                        _ensure_dir(rep_dir)

                        print(
                            f"[plot_results] Writing representative plots for split='{split}' "
                            f"seed={rep_run.seed} ablation='{rep_ab}' baseline='{baseline_model}' ..."
                        )

                        saved_all += _plot_roc_pr(qnn_m, base_m, rep_dir)

                        p = _plot_threshold_sweep(
                            qnn_o,
                            "Threshold Sweep (QNN)",
                            os.path.join(rep_dir, "threshold_sweep_qnn.png"),
                        )
                        if p:
                            saved_all.append(p)

                        p = _plot_threshold_sweep(
                            base_o,
                            f"Threshold Sweep ({baseline_model})",
                            os.path.join(rep_dir, "threshold_sweep_baseline.png"),
                        )
                        if p:
                            saved_all.append(p)

                        cm_q = _cm_from_metrics(qnn_m)
                        cm_b = _cm_from_metrics(base_m)

                        saved_all.append(
                            _plot_confusion(
                                cm_q,
                                "QNN Confusion (Counts)",
                                os.path.join(rep_dir, "confusion_qnn.png"),
                                normalize=False,
                            )
                        )
                        saved_all.append(
                            _plot_confusion(
                                cm_q,
                                "QNN Confusion (Normalized)",
                                os.path.join(rep_dir, "confusion_qnn_norm.png"),
                                normalize=True,
                            )
                        )
                        saved_all.append(
                            _plot_confusion(
                                cm_b,
                                f"{baseline_model} Confusion (Counts)",
                                os.path.join(rep_dir, "confusion_baseline.png"),
                                normalize=False,
                            )
                        )
                        saved_all.append(
                            _plot_confusion(
                                cm_b,
                                f"{baseline_model} Confusion (Normalized)",
                                os.path.join(rep_dir, "confusion_baseline_norm.png"),
                                normalize=True,
                            )
                        )

                        hist_obj = _try_load_json(_qnn_history_path(rep_run))
                        hist_list = hist_obj if isinstance(hist_obj, list) else []
                        if hist_list:
                            saved_all += _plot_training_curves(hist_list, rep_dir)

                        representative_name = f"{split}_seed{rep_run.seed}_{rep_ab}_{baseline_model}"
                        representative_name = representative_name.replace("\\", "_").replace("/", "_").replace(" ", "_")

                        saved_all.append(
                            _plot_combined_onecol(
                                hist_list,
                                qnn_m,
                                base_m,
                                os.path.join(
                                    rep_dir,
                                    f"combine_fig_col_{representative_name}.png",
                                ),
                            )
                        )

        # Save aggregate summary
        table_path = os.path.join(plots_root, "summary_mean_std.json")
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        csv_path = os.path.join(plots_root, "summary_mean_std.csv")
        if summary_rows:
            cols = list(summary_rows[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)

        print(f"[plot_results] Saved summary table to:\n  - {table_path}\n  - {csv_path}")

    # Final file list
    print(f"\n[plot_results] Saved {len(saved_all)} plot files under: {plots_root}")
    for path in saved_all:
        print(f" - {path}")


if __name__ == "__main__":
    main()