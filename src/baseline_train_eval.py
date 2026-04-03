"""
Train + evaluate classical baselines on CESNET split NPZs created by make_splits.py.

Supported baseline models:
- logreg
- rf
- extratrees

This script can:
- run one configured model (backward compatible)
- or run multiple models sequentially in one command via:
    baseline:
      models: [logreg, rf, extratrees]

It evaluates on the same split regime as the QNN:
- random_stratified
- group_holdout
- time_holdout

Metrics (binary anomaly detection):
- ROC AUC + ROC arrays
- AP (PR-AUC) + PR arrays
- Threshold selected via Youden's J
- Accuracy, TPR, FPR, Precision, Recall, F1, BalancedAccuracy, MCC
- Confusion matrix

Outputs (per run directory, no overwrite):
  artifacts/experiments/<run_name>/<split>/baseline/<model>/seed_<seed>/
    baseline_eval_metrics.json
    baseline_eval_outputs.npz
    run_config.json

Examples
--------
Run full experiment sweep (all configured splits / seeds / baseline models):
  python -m src.baseline_train_eval

Run one split/seed only, but still for all configured baseline models:
  python -m src.baseline_train_eval --single --split group --seed 42

Run one split/seed for one specific baseline model:
  python -m src.baseline_train_eval --single --split group --seed 42 --model extratrees
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    """Write JSON to disk with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_split_name(name: str) -> str:
    """
    Normalize user-friendly split aliases to canonical split names.

    Accepted aliases:
    - random -> random_stratified
    - group  -> group_holdout
    - time   -> time_holdout
    """
    s = str(name).strip().lower()
    aliases = {
        "random": "random_stratified",
        "random_stratified": "random_stratified",
        "group": "group_holdout",
        "group_holdout": "group_holdout",
        "time": "time_holdout",
        "time_holdout": "time_holdout",
    }
    if s not in aliases:
        raise ValueError(
            f"Unknown split '{name}'. Use one of: "
            "random, group, time (or *_holdout / *_stratified)."
        )
    return aliases[s]


def resolve_split_npz_path(cfg: Dict[str, Any], split_name: str, seed: int) -> str:
    """
    Determine which split NPZ file to use.

    Expected files (from make_splits.py):
      - <out_dir>/random_stratified_seed{seed}.npz
      - <out_dir>/group_holdout_seed{seed}.npz
      - <out_dir>/time_holdout.npz
    """
    split_name = normalize_split_name(split_name)

    # Support both config styles:
    # 1) make_splits.out_dir
    # 2) experiment.splits.output_dir
    make_splits_cfg = cfg.get("make_splits", {}) or {}
    exp_cfg = cfg.get("experiment", {}) or {}
    exp_splits_cfg = exp_cfg.get("splits", {}) or {}

    out_dir = str(
        make_splits_cfg.get(
            "out_dir",
            exp_splits_cfg.get("output_dir", os.path.join("data", "processed", "splits")),
        )
    )

    if split_name == "time_holdout":
        path = os.path.join(out_dir, "time_holdout.npz")
    else:
        path = os.path.join(out_dir, f"{split_name}_seed{seed}.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split NPZ not found: {path}\n"
            f"Did you run: python -m src.make_splits ?"
        )
    return path


def load_split_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load X_train / y_train / X_eval / y_eval from a split NPZ.

    Expected keys:
      X_train, y_train, X_eval, y_eval
    """
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"].astype(np.float32),
        d["y_train"].astype(np.int64),
        d["X_eval"].astype(np.float32),
        d["y_eval"].astype(np.int64),
    )


# ---------------------------------------------------------------------
# Experiment / baseline config helpers
# ---------------------------------------------------------------------
def get_experiment_splits(cfg: Dict[str, Any]) -> List[str]:
    """
    Resolve which split strategies to run.

    Supports either:
      experiment.split_modes: [random_stratified, group, time]
    or
      experiment.splits.strategies: [...]
    """
    exp = cfg.get("experiment", {}) or {}

    if "split_modes" in exp:
        raw = exp.get("split_modes", [])
    else:
        raw = (exp.get("splits", {}) or {}).get(
            "strategies",
            ["random_stratified", "group_holdout", "time_holdout"],
        )

    return [normalize_split_name(x) for x in raw]


def get_baseline_models(cfg: Dict[str, Any]) -> List[str]:
    """
    Resolve which baseline models to run.

    Preferred new style:
      baseline:
        models: [logreg, rf, extratrees]

    Backward-compatible old style:
      baseline:
        model: extratrees
    """
    bcfg = cfg.get("baseline", {}) or {}

    if "models" in bcfg and bcfg["models"] is not None:
        models = [str(m).lower().strip() for m in bcfg["models"]]
    else:
        models = [str(bcfg.get("model", "logreg")).lower().strip()]

    valid = {"logreg", "rf", "extratrees"}
    bad = [m for m in models if m not in valid]
    if bad:
        raise ValueError(
            f"Unknown baseline model(s): {bad}. "
            f"Use only: {sorted(valid)}"
        )
    return models


def get_model_cfg(cfg: Dict[str, Any], model_name: str, seed: int) -> Dict[str, Any]:
    """
    Build an effective per-model config.

    Supports:
    1) New style:
         baseline:
           models: [logreg, rf, extratrees]
           random_state: 42
           n_jobs: -1
           logreg: {...}
           rf: {...}
           extratrees: {...}

    2) Old style flat config:
         baseline:
           model: extratrees
           n_estimators: 600
           ...
    """
    bcfg = cfg.get("baseline", {}) or {}

    # Common keys shared across models
    merged: Dict[str, Any] = {
        "random_state": int(bcfg.get("random_state", seed)),
        "n_jobs": int(bcfg.get("n_jobs", -1)),
    }

    # New nested style
    nested_cfg = bcfg.get(model_name, None)
    if isinstance(nested_cfg, dict):
        merged.update(nested_cfg)
        return merged

    # Old flat style fallback
    # We keep backward compatibility by pulling model-specific keys
    # directly from baseline: if nested config is not present.
    if model_name == "logreg":
        merged.update(
            {
                "C": float(bcfg.get("C", 1.0)),
                "max_iter": int(bcfg.get("max_iter", 4000)),
                "class_weight": bcfg.get("class_weight", "balanced"),
            }
        )
    elif model_name == "rf":
        merged.update(
            {
                "n_estimators": int(bcfg.get("n_estimators", 400)),
                "class_weight": bcfg.get("class_weight", "balanced_subsample"),
            }
        )
    elif model_name == "extratrees":
        merged.update(
            {
                "n_estimators": int(bcfg.get("n_estimators", 600)),
                "max_depth": bcfg.get("max_depth", None),
                "min_samples_split": int(bcfg.get("min_samples_split", 2)),
                "min_samples_leaf": int(bcfg.get("min_samples_leaf", 1)),
                "max_features": bcfg.get("max_features", "sqrt"),
                "class_weight": bcfg.get("class_weight", "balanced"),
            }
        )

    return merged


# ---------------------------------------------------------------------
# Baseline core (single run)
# ---------------------------------------------------------------------
def run_baseline_one(
    *,
    cfg: Dict[str, Any],
    split_name: str,
    seed: int,
    model_name: str,
    out_dir: str,
) -> None:
    """
    Train + evaluate one baseline model for one split / seed pair.
    """
    ensure_dir(out_dir)

    split_name = normalize_split_name(split_name)
    model_name = str(model_name).lower().strip()
    model_cfg = get_model_cfg(cfg, model_name, seed)

    split_npz_path = resolve_split_npz_path(cfg, split_name, seed)
    X_train, y_train_raw, X_eval, y_eval_raw = load_split_npz(split_npz_path)

    # Collapse to binary anomaly detection:
    # normal=0, anomaly={1,2}
    y_train = (y_train_raw != 0).astype(int)
    y_eval = (y_eval_raw != 0).astype(int)

    print("\n[baseline] *=================================================================================================================================*")
    print(f"[baseline] split={split_name}  seed={seed}  model={model_name}")
    print(f"[baseline] split_npz={split_npz_path}")
    print(f"[baseline] X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[baseline] X_eval ={X_eval.shape},  y_eval ={y_eval.shape}")

    # -----------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 400)),
            random_state=int(model_cfg.get("random_state", seed)),
            class_weight=str(model_cfg.get("class_weight", "balanced_subsample")),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
        )
    elif model_name == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 600)),
            random_state=int(model_cfg.get("random_state", seed)),
            class_weight=str(model_cfg.get("class_weight", "balanced")),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=int(model_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            max_features=model_cfg.get("max_features", "sqrt"),
        )
    elif model_name == "logreg":
        model = LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 4000)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
            class_weight=str(model_cfg.get("class_weight", "balanced")),
            random_state=int(model_cfg.get("random_state", seed)),
        )
    else:
        raise ValueError(
            f"Unknown baseline model '{model_name}'. "
            f"Use one of: logreg, rf, extratrees"
        )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    model.fit(X_train, y_train)
    print(f"[baseline] Training complete. model='{model_name}'")

    # Probability of anomaly class
    p_anom = model.predict_proba(X_eval)[:, 1].astype(float)

    # -----------------------------------------------------------------
    # Curves + threshold selection
    # -----------------------------------------------------------------
    auc = float(roc_auc_score(y_eval, p_anom))
    ap = float(average_precision_score(y_eval, p_anom))

    fpr_curve, tpr_curve, thr_curve = roc_curve(y_eval, p_anom)
    j_stat = tpr_curve - fpr_curve
    best_idx = int(np.argmax(j_stat))
    best_thr = float(thr_curve[best_idx])

    y_pred = (p_anom >= best_thr).astype(int)

    # -----------------------------------------------------------------
    # Metrics at selected threshold
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    acc = float(accuracy_score(y_eval, y_pred))
    tpr_at = float(tp / (tp + fn + 1e-12))
    fpr_at = float(fp / (fp + tn + 1e-12))
    prec_at = float(precision_score(y_eval, y_pred, zero_division=0))
    rec_at = float(recall_score(y_eval, y_pred, zero_division=0))
    f1_at = float(f1_score(y_eval, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_eval, y_pred))
    mcc = float(matthews_corrcoef(y_eval, y_pred)) if np.unique(y_eval).size == 2 else 0.0

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_eval, p_anom)

    metrics = {
        "Model": model_name,
        "Split": split_name,
        "Seed": int(seed),
        "AUC": auc,
        "AP": ap,
        "TPR": tpr_at,
        "FPR": fpr_at,
        "Precision": prec_at,
        "Recall": rec_at,
        "F1": f1_at,
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc,
        "MCC": mcc,
        "Threshold": best_thr,
        "Confusion": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
        "ROC": {
            "fpr": fpr_curve.tolist(),
            "tpr": tpr_curve.tolist(),
            "thresholds": thr_curve.tolist(),
        },
        "PR": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "SplitNPZ": split_npz_path,
    }

    print("[baseline] Metrics:")
    print(f"  AUC       : {auc:.4f}")
    print(f"  AP        : {ap:.4f}")
    print(f"  TPR (rec) : {tpr_at:.4f}")
    print(f"  FPR       : {fpr_at:.4f}")
    print(f"  Precision : {prec_at:.4f}")
    print(f"  F1        : {f1_at:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Threshold : {best_thr:.6f}")
    print(f"  Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # -----------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------
    out_metrics = os.path.join(out_dir, "baseline_eval_metrics.json")
    out_outputs = os.path.join(out_dir, "baseline_eval_outputs.npz")
    out_cfg = os.path.join(out_dir, "run_config.json")

    save_json(out_metrics, metrics)

    np.savez(
        out_outputs,
        p_anom=p_anom,
        y_true=y_eval.astype(int),
        y_pred=y_pred.astype(int),
        threshold=best_thr,
        split=split_name,
        seed=int(seed),
        model=model_name,
    )

    run_cfg = {
        "split": split_name,
        "seed": int(seed),
        "model": model_name,
        "baseline_cfg_effective": model_cfg,
    }
    save_json(out_cfg, run_cfg)

    print(f"[baseline] Saved metrics to: {out_metrics}")
    print(f"[baseline] Saved outputs to: {out_outputs}")
    print(f"[baseline] Saved run config to: {out_cfg}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/cesnet_vqczti.yaml")
    ap.add_argument(
        "--single",
        action="store_true",
        help="Disable experiment-wide split/seed sweep. "
        "Uses the provided --split / --seed, but still supports multiple configured models.",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="Split: random/group/time or random_stratified/group_holdout/time_holdout",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--model",
        default=None,
        help="Optional specific baseline model to run: logreg / rf / extratrees",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.get("experiment", {}) or {}

    sweep_enabled = bool(exp.get("enabled", False)) and (not args.single)

    # Output root follows the same run_name convention as train_qnn.py
    out_root = str(exp.get("out_dir", os.path.join("artifacts", "experiments")))
    run_name = str(exp.get("run_name", "run"))
    out_root = os.path.join(out_root, run_name)
    ensure_dir(out_root)

    # -------------------------------------------------------------
    # Resolve seeds / splits
    # -------------------------------------------------------------
    if sweep_enabled:
        seeds = [int(s) for s in exp.get("seeds", [42])]
        strategies = get_experiment_splits(cfg)
    else:
        default_seed = int((cfg.get("baseline", {}) or {}).get("random_state", 42))
        seeds = [int(args.seed if args.seed is not None else default_seed)]
        strategies = [normalize_split_name(args.split or "random_stratified")]

    # -------------------------------------------------------------
    # Resolve baseline models
    # -------------------------------------------------------------
    if args.model is not None:
        models = [str(args.model).lower().strip()]
    else:
        models = get_baseline_models(cfg)

    # -------------------------------------------------------------
    # Run jobs
    # -------------------------------------------------------------
    for split_name in strategies:
        split_norm = normalize_split_name(split_name)
        for model_name in models:
            for seed in seeds:
                out_dir = os.path.join(
                    out_root,
                    split_norm,
                    "baseline",
                    model_name,
                    f"seed_{seed}",
                )
                run_baseline_one(
                    cfg=cfg,
                    split_name=split_norm,
                    seed=int(seed),
                    model_name=model_name,
                    out_dir=out_dir,
                )

    print(f"\n[baseline] DONE. Outputs are under: {out_root}")


if __name__ == "__main__":
    main()