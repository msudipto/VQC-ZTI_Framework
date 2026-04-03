"""
Evaluate trained VQC-ZTI QNN runs.

Single-run mode:
  python -m src.eval_qnn
  python -m src.eval_qnn --split group --seed 43
  python -m src.eval_qnn --split random --seed 42 --ablation no_head
  python -m src.eval_qnn --split time --weights artifacts/qnn_model.pt

All-runs mode:
  python -m src.eval_qnn --all-runs
  python -m src.eval_qnn --all-runs --only-missing

Notes
-----
- Single-run mode prints legacy-style console output.
- All-runs mode iterates over every trained run_config.json found under:
    artifacts/experiments/run/<split>/<ablation>/seed_<seed>/run_config.json
- The evaluator saves:
    1) per-run outputs inside the corresponding run directory
    2) backward-compatible root-level copies under artifacts/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml
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
from torch.utils.data import DataLoader, TensorDataset

from .qnn_model import VQCClassifier


# ---------------------------------------------------------------------
# Basic I/O helpers
# ---------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_torch_load(path: str, device: torch.device):
    """
    Load a torch checkpoint in a backward-compatible way.

    Some torch versions support weights_only=True; older versions do not.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file if it exists and contains a dict."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _try_paths(paths: Iterable[str]) -> Optional[str]:
    """Return the first existing path from a list."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------
def normalize_split_name(name: str) -> str:
    """
    Normalize split aliases to the canonical names used by the pipeline.
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
            f"Unknown split '{name}'. Use one of: random, group, time "
            f"(or the canonical names random_stratified, group_holdout, time_holdout)."
        )
    return aliases[s]


def _display_split_name(split_name: str) -> str:
    """
    Friendly/legacy display name for console output.
    """
    s = str(split_name).strip().lower()
    if s == "random_stratified":
        return "random"
    if s == "group_holdout":
        return "group"
    if s == "time_holdout":
        return "time"
    return s


def resolve_split_npz_path(cfg: Dict[str, Any], split_name: str, seed: int) -> str:
    """
    Resolve the exact split NPZ path written by src.make_splits.

    Expected files:
      data/processed/splits/random_stratified_seed42.npz
      data/processed/splits/group_holdout_seed42.npz
      data/processed/splits/time_holdout.npz
    """
    exp = cfg.get("experiment", {}) or {}
    splits_cfg = exp.get("splits", {}) or {}
    out_dir = str(
        splits_cfg.get("output_dir", os.path.join("data", "processed", "splits"))
    )

    split_name = normalize_split_name(split_name)

    if split_name == "time_holdout":
        split_path = os.path.join(out_dir, "time_holdout.npz")
    else:
        split_path = os.path.join(out_dir, f"{split_name}_seed{seed}.npz")

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split NPZ not found: {split_path}\n"
            f"Did you run: python -m src.make_splits ?"
        )
    return split_path


# ---------------------------------------------------------------------
# Run / artifact helpers
# ---------------------------------------------------------------------
def _build_run_tag(split_name: str, seed: int, ablation: Optional[str]) -> str:
    """
    Build a safe run tag for filenames.
    """
    tag = f"{_display_split_name(split_name)}_seed{seed}"
    if ablation:
        tag += f"_abl_{ablation}"
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in tag)


def _get_run_dir(
    cfg: Dict[str, Any],
    split_name: str,
    seed: int,
    ablation: Optional[str],
) -> str:
    """
    Return the canonical per-run artifact directory.
    """
    exp = cfg.get("experiment", {}) or {}
    out_root = str(exp.get("out_dir", os.path.join("artifacts", "experiments")))
    run_name = str(exp.get("run_name", "run"))
    return os.path.join(
        out_root,
        run_name,
        normalize_split_name(split_name),
        ablation if ablation else "full_hybrid",
        f"seed_{seed}",
    )


def _resolve_weights_path(
    cfg: Dict[str, Any],
    split_name: str,
    seed: int,
    ablation: Optional[str],
    run_tag: str,
    cli_weights: Optional[str],
) -> str:
    """
    Resolve the checkpoint path for the requested run.
    """
    if cli_weights:
        if not os.path.exists(cli_weights):
            raise FileNotFoundError(f"Provided --weights not found: {cli_weights}")
        return cli_weights

    evcfg = cfg.get("eval_qnn", {}) or {}
    if isinstance(evcfg, dict):
        weights_path = evcfg.get("weights_path")
        if weights_path and os.path.exists(weights_path):
            return weights_path

    run_dir = _get_run_dir(cfg, split_name, seed, ablation)
    split_name = normalize_split_name(split_name)

    candidates = [
        os.path.join(run_dir, "qnn_model.pt"),
        os.path.join("artifacts", f"qnn_model_{run_tag}.pt"),
        os.path.join("artifacts", "qnn_model.pt"),
    ]

    found = _try_paths(candidates)
    if found is None:
        raise FileNotFoundError(
            "Could not find QNN weights. Tried:\n  - " + "\n  - ".join(candidates)
        )
    return found


def _resolve_model_info_path(
    cfg: Dict[str, Any],
    split_name: str,
    seed: int,
    ablation: Optional[str],
    run_tag: str,
) -> Optional[str]:
    """
    Resolve the model-info JSON path for the requested run.
    """
    run_dir = _get_run_dir(cfg, split_name, seed, ablation)
    candidates = [
        os.path.join(run_dir, "qnn_model_info.json"),
        os.path.join("artifacts", f"qnn_model_info_{run_tag}.json"),
        os.path.join("artifacts", "qnn_model_info.json"),
    ]
    return _try_paths(candidates)


def _iter_qnn_run_configs(cfg: Dict[str, Any]):
    """
    Iterate over all run_config.json files produced by train_qnn.py.
    """
    exp = cfg.get("experiment", {}) or {}
    out_root = str(exp.get("out_dir", os.path.join("artifacts", "experiments")))
    run_name = str(exp.get("run_name", "run"))
    base = os.path.join(out_root, run_name)

    pattern = os.path.join(base, "*", "*", "seed_*", "run_config.json")
    for path in sorted(glob.glob(pattern)):
        obj = _load_json(path)
        if isinstance(obj, dict):
            yield path, obj


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------
def _bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_boot: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap 95% confidence intervals for key binary metrics
    at a fixed threshold.
    """
    rng = np.random.default_rng(seed)
    n = int(y_true.shape[0])

    stats = {
        "AUC": [],
        "AP": [],
        "Accuracy": [],
        "TPR": [],
        "FPR": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "BalancedAccuracy": [],
        "MCC": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        sb = scores[idx]

        # AUC/AP need both classes present.
        if np.unique(yb).size < 2:
            continue

        try:
            auc_b = roc_auc_score(yb, sb)
            ap_b = average_precision_score(yb, sb)
        except Exception:
            continue

        yhat = (sb >= threshold).astype(int)
        cm = confusion_matrix(yb, yhat, labels=[0, 1])
        tn, fp, fn, tp = map(int, cm.ravel())

        eps = 1e-12
        acc_b = (tp + tn) / max(tp + tn + fp + fn, 1)
        tpr_b = tp / (tp + fn + eps)
        fpr_b = fp / (fp + tn + eps)
        prec_b = tp / (tp + fp + eps)
        rec_b = tpr_b
        f1_b = 0.0 if (prec_b + rec_b) == 0 else (2 * prec_b * rec_b) / (prec_b + rec_b)
        bal_b = 0.5 * (tpr_b + (tn / (tn + fp + eps)))
        mcc_b = matthews_corrcoef(yb, yhat) if np.unique(yb).size == 2 else 0.0

        stats["AUC"].append(float(auc_b))
        stats["AP"].append(float(ap_b))
        stats["Accuracy"].append(float(acc_b))
        stats["TPR"].append(float(tpr_b))
        stats["FPR"].append(float(fpr_b))
        stats["Precision"].append(float(prec_b))
        stats["Recall"].append(float(rec_b))
        stats["F1"].append(float(f1_b))
        stats["BalancedAccuracy"].append(float(bal_b))
        stats["MCC"].append(float(mcc_b))

    def ci(arr: list[float]) -> Optional[Dict[str, float]]:
        if len(arr) < 30:
            return None
        return {
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
        }

    out = {k: ci(v) for k, v in stats.items()}
    out["n_effective"] = int(max(len(stats["AUC"]), len(stats["Accuracy"])))
    out["n_boot"] = int(n_boot)
    return out


def pick_device(prefer_cuda: bool) -> torch.device:
    """Pick CUDA when requested and available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------
def evaluate_one(
    cfg: Dict[str, Any],
    split_name: str,
    seed: int,
    ablation: Optional[str],
    cli_weights: Optional[str] = None,
    legacy_console: bool = False,
) -> None:
    """
    Evaluate one specific trained run.
    """
    tcfg = cfg.get("train_qnn", {}) or {}
    evcfg = cfg.get("eval_qnn", {}) or {}

    split_name = normalize_split_name(split_name)
    display_split = _display_split_name(split_name)
    run_tag = _build_run_tag(split_name, seed, ablation)

    # -----------------------------------------------------------------
    # Load split NPZ
    # -----------------------------------------------------------------
    split_npz_path = resolve_split_npz_path(cfg, split_name, seed)
    d = np.load(split_npz_path, allow_pickle=True)

    X_eval = np.asarray(d["X_eval"], dtype=np.float32)
    y_eval = np.asarray(d["y_eval"], dtype=np.int64)
    id_time_eval = np.asarray(d.get("id_time_eval", np.full(len(y_eval), -1)), dtype=int)
    group_key_eval = np.asarray(
        d.get("group_key_eval", np.full(len(y_eval), "-1", dtype=object)),
        dtype=str,
    )

    if legacy_console:
        print(
            f"[eval_qnn] Using legacy eval split (X_eval/y_eval) | split='{display_split}' seed={seed}",
            flush=True,
        )
    else:
        print(
            f"[eval_qnn] Using split='{split_name}' seed={seed} ablation={ablation} | split_npz={split_npz_path}",
            flush=True,
        )

    print(f"[eval_qnn] Eval split: X_eval={X_eval.shape}, y_eval={y_eval.shape}", flush=True)

    # Binary anomaly target: normal=0, anomaly={1,2}
    y_true = (y_eval != 0).astype(int)

    # -----------------------------------------------------------------
    # Resolve model artifacts
    # -----------------------------------------------------------------
    weights_path = _resolve_weights_path(cfg, split_name, seed, ablation, run_tag, cli_weights)
    info_path = _resolve_model_info_path(cfg, split_name, seed, ablation, run_tag)
    info = _load_json(info_path) if info_path else None
    info = info if isinstance(info, dict) else {}

    n_features = int(X_eval.shape[1])
    task = str(info.get("task", tcfg.get("task", "multiclass"))).lower().strip()
    n_layers = int(info.get("n_layers", tcfg.get("n_layers", 2)))
    n_qubits = int(info.get("n_qubits", tcfg.get("n_qubits", n_features)))
    n_classes = int(info.get("n_classes", (2 if task == "binary" else 3)))

    prefer_cuda = bool(tcfg.get("prefer_cuda", False))
    dev_req = pick_device(prefer_cuda)
    print(f"[eval_qnn] Device (requested): {dev_req}", flush=True)

    model = VQCClassifier(
        n_features=n_features,
        n_layers=n_layers,
        n_qubits=n_qubits,
        task=task,
        n_classes=n_classes,
        diff_method=str(info.get("diff_method", tcfg.get("diff_method", "adjoint"))),
        use_embedder=bool(info.get("use_embedder", tcfg.get("use_embedder", True))),
        embed_hidden=int(info.get("embed_hidden", tcfg.get("embed_hidden", 64))),
        embed_layers=int(info.get("embed_layers", tcfg.get("embed_layers", 2))),
        angle_scale=float(info.get("angle_scale", tcfg.get("angle_scale", float(np.pi)))),
        use_head=bool(info.get("use_head", tcfg.get("use_head", True))),
        head_hidden=int(info.get("head_hidden", tcfg.get("head_hidden", 32))),
    ).to(dev_req)

    # Smoke-test CUDA path. If it fails, gracefully fall back to CPU.
    dev_final = dev_req
    try:
        with torch.no_grad():
            dummy = torch.zeros((2, n_features), dtype=torch.float32, device=dev_req)
            _ = model(dummy)
    except Exception as e:
        if dev_req.type == "cuda":
            print(
                f"[eval_qnn] CUDA path failed; falling back to CPU. "
                f"Reason: {type(e).__name__}: {e}",
                flush=True,
            )
            dev_final = torch.device("cpu")
            model = model.to(dev_final)
        else:
            raise

    print(f"[eval_qnn] Device (final): {dev_final}", flush=True)

    # -----------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------
    print(f"[eval_qnn] Loading weights: {weights_path}", flush=True)
    state = safe_torch_load(weights_path, dev_final)
    model.load_state_dict(state)
    model.eval()
    print("[eval_qnn] Weights loaded. Starting eval...", flush=True)

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------
    X_t = torch.tensor(X_eval, dtype=torch.float32)
    batch_size = int(tcfg.get("eval_batch_size", 32))
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    num_batches = len(loader)
    print(f"[eval_qnn] Running forward pass in {num_batches} batches...", flush=True)

    probs_chunks = []
    seen = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (bx,) in enumerate(loader, start=1):
            bx = bx.to(dev_final)
            log_probs = model(bx)
            probs = torch.exp(log_probs)
            probs_chunks.append(probs.cpu())

            seen += int(bx.shape[0])
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == num_batches:
                elapsed = time.time() - start_time
                print(
                    f"[eval_qnn] {seen}/{len(X_eval)} samples | batch {batch_idx}/{num_batches} | elapsed {elapsed:.1f}s",
                    flush=True,
                )

    probs = torch.cat(probs_chunks, dim=0).numpy()[: len(X_eval)]
    print(f"[eval_qnn] Final probs: {probs.shape}", flush=True)

    # -----------------------------------------------------------------
    # Binary anomaly score
    # -----------------------------------------------------------------
    if probs.shape[1] == 2:
        p_anom = probs[:, 1]
    else:
        # Treat classes 1 and 2 as anomalous.
        p_anom = probs[:, 1] + probs[:, 2]

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    auc = float(roc_auc_score(y_true, p_anom))
    fpr, tpr, thr = roc_curve(y_true, p_anom)

    threshold_policy = str(evcfg.get("threshold_policy", "youden")).lower().strip()
    if threshold_policy.startswith("fpr@"):
        target = float(threshold_policy.split("@", 1)[1])
        ok = np.where(fpr <= target)[0]
        best_idx = int(ok[-1]) if len(ok) > 0 else int(np.argmax(tpr - fpr))
        best_thr = float(thr[best_idx])
    else:
        best_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thr[best_idx])

    y_pred = (p_anom >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    acc = float(accuracy_score(y_true, y_pred))
    tpr_at = float(tp / (tp + fn + 1e-12))
    fpr_at = float(fp / (fp + tn + 1e-12))
    prec_at = float(precision_score(y_true, y_pred, zero_division=0))
    rec_at = float(recall_score(y_true, y_pred, zero_division=0))
    f1_at = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if np.unique(y_true).size == 2 else 0.0

    precision, recall, pr_thr = precision_recall_curve(y_true, p_anom)
    ap_score = float(average_precision_score(y_true, p_anom))

    ci_cfg = evcfg.get("bootstrap_ci", {}) or {}
    do_ci = bool(ci_cfg.get("enabled", False))
    n_boot = int(ci_cfg.get("n_boot", 300))
    ci_seed = int(ci_cfg.get("seed", seed))
    ci = _bootstrap_ci(y_true, p_anom, best_thr, n_boot=n_boot, seed=ci_seed) if do_ci else None

    metrics = {
        "RunTag": run_tag,
        "Split": str(display_split) if legacy_console else str(split_name),
        "Seed": int(seed),
        "Ablation": ablation,
        "AUC": auc,
        "AP": ap_score,
        "TPR": tpr_at,
        "FPR": fpr_at,
        "Precision": prec_at,
        "Recall": rec_at,
        "F1": f1_at,
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc,
        "MCC": mcc,
        "Threshold": best_thr,
        "Confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "ROC": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thr.tolist(),
        },
        "PR": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thr.tolist(),
        },
        "BootstrapCI": ci,
        "EvalMeta": {
            "n_eval": int(len(y_true)),
            "id_time_min": int(id_time_eval.min()) if len(id_time_eval) else None,
            "id_time_max": int(id_time_eval.max()) if len(id_time_eval) else None,
            "id_time_unique": int(len(set(id_time_eval.tolist()))) if len(id_time_eval) else None,
            "group_key_unique": int(len(set(group_key_eval.tolist()))) if len(group_key_eval) else None,
        },
    }

    print("[eval_qnn] Metrics:")
    print(f"  RunTag    : {run_tag}")
    print(f"  Split     : {display_split} (seed={seed})")
    print(f"  AUC       : {auc:.4f}")
    print(f"  AP        : {ap_score:.4f}")
    print(f"  TPR (rec) : {tpr_at:.4f}")
    print(f"  FPR       : {fpr_at:.4f}")
    print(f"  Precision : {prec_at:.4f}")
    print(f"  F1        : {f1_at:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Threshold : {best_thr:.6f}")
    print(f"  Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    run_dir = _get_run_dir(cfg, split_name, seed, ablation)
    os.makedirs(run_dir, exist_ok=True)

    # Primary per-run outputs
    metrics_path = os.path.join(run_dir, "qnn_eval_metrics.json")
    outputs_path = os.path.join(run_dir, "qnn_eval_outputs.npz")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        outputs_path,
        probs=probs,
        p_anom=p_anom,
        y_true=y_true,
        y_pred=y_pred,
        threshold=best_thr,
        split=str(split_name),
        seed=int(seed),
        run_tag=run_tag,
        id_time_eval=id_time_eval,
        group_key_eval=group_key_eval,
    )

    # Backward-compatible root-level copies
    artifact_root = "artifacts"
    os.makedirs(artifact_root, exist_ok=True)

    out_json_std = os.path.join(artifact_root, "qnn_eval_metrics.json")
    out_npz_std = os.path.join(artifact_root, "qnn_eval_outputs.npz")
    out_json_tag = os.path.join(artifact_root, f"qnn_eval_metrics_{run_tag}.json")
    out_npz_tag = os.path.join(artifact_root, f"qnn_eval_outputs_{run_tag}.npz")

    for out_json in (out_json_std, out_json_tag):
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    for out_npz in (out_npz_std, out_npz_tag):
        np.savez(
            out_npz,
            probs=probs,
            p_anom=p_anom,
            y_true=y_true,
            y_pred=y_pred,
            threshold=best_thr,
            split=str(split_name),
            seed=int(seed),
            run_tag=run_tag,
            id_time_eval=id_time_eval,
            group_key_eval=group_key_eval,
        )

    if legacy_console:
        print(f"[eval_qnn] Saved metrics to:\n  - {out_json_std}\n  - {out_json_tag}")
        print(f"[eval_qnn] Saved eval outputs to:\n  - {out_npz_std}\n  - {out_npz_tag}")
    else:
        print(
            f"[eval_qnn] Saved metrics to:\n"
            f"  - {metrics_path}\n"
            f"  - {out_json_std}\n"
            f"  - {out_json_tag}"
        )
        print(
            f"[eval_qnn] Saved eval outputs to:\n"
            f"  - {outputs_path}\n"
            f"  - {out_npz_std}\n"
            f"  - {out_npz_tag}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    """
    Entry point.

    Single-run mode:
      evaluate one requested run.

    All-runs mode:
      iterate over every trained run under artifacts/experiments/run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cesnet_vqczti.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Evaluate every trained QNN run under artifacts/experiments/run",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="With --all-runs, skip runs that already have qnn_eval_metrics.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evcfg = cfg.get("eval_qnn", {}) or {}

    if args.all_runs:
        jobs = list(_iter_qnn_run_configs(cfg))
        if not jobs:
            raise FileNotFoundError(
                "No run_config.json files found under artifacts/experiments/run"
            )

        print(f"[eval_qnn] Found {len(jobs)} trained runs.", flush=True)

        for _, rcfg in jobs:
            split_name = normalize_split_name(rcfg["split"])
            seed = int(rcfg["seed"])
            ablation = rcfg.get("ablation", None)

            run_dir = _get_run_dir(cfg, split_name, seed, ablation)
            metrics_path = os.path.join(run_dir, "qnn_eval_metrics.json")

            if args.only_missing and os.path.exists(metrics_path):
                print(
                    f"[eval_qnn] Skipping existing: split={split_name} seed={seed} ablation={ablation}",
                    flush=True,
                )
                continue
            
            print(f"\n[eval_qnn] *=================================================================================================================================*")
            print(
                f"[eval_qnn] Evaluating: split={split_name} seed={seed} ablation={ablation}",
                flush=True,
            )
            evaluate_one(
                cfg,
                split_name,
                seed,
                ablation,
                cli_weights=None,
                legacy_console=False,
            )

        print("[eval_qnn] DONE. All requested runs evaluated.", flush=True)
        return

    # Single-run mode
    split_name = normalize_split_name(
        args.split or evcfg.get("split", evcfg.get("split_name", "random_stratified"))
    )
    seed = int(args.seed if args.seed is not None else evcfg.get("seed", 42))
    ablation = args.ablation if args.ablation is not None else evcfg.get("ablation", None)

    evaluate_one(
        cfg,
        split_name,
        seed,
        ablation,
        cli_weights=args.weights,
        legacy_console=True,
    )
    return


if __name__ == "__main__":
    main()