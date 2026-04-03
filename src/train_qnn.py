"""
Train the Hybrid QNN on CESNET split NPZs (random/group/time) created by make_splits.py.

This file supports:
- Multi-seed training (for mean ± std reporting)
- Ablation study (publishability)
- Split regimes:
    * random_stratified_seed{seed}.npz
    * group_holdout_seed{seed}.npz
    * time_holdout.npz

Recommended workflow:
1) python -m src.preprocess_cesnet
2) python -m src.make_splits
3) python -m src.train_qnn          (runs experiment sweep if experiment.enabled: true)
4) python -m src.eval_qnn           (we'll update to evaluate per-run outputs)
5) python -m src.baseline_train_eval
6) python -m src.plot_results

Run:
  python -m src.train_qnn

Optional single-run overrides:
  python -m src.train_qnn --single --split group_holdout --seed 42 --ablation full_hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .qnn_model import VQCClassifier

# Optional tqdm progress bars
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# --------------------------
# Utilities
# --------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Deterministic training behavior (to the extent possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_device(prefer_cuda: bool) -> torch.device:
    """Prefer CUDA if requested and available."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_split_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads X_train/y_train/X_eval/y_eval from a split NPZ produced by make_splits.py."""
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"].astype(np.float32),
        d["y_train"].astype(np.int64),
        d["X_eval"].astype(np.float32),
        d["y_eval"].astype(np.int64),
    )


def iter_batches(loader, desc: str, use_tqdm: bool = True):
    """tqdm if installed; otherwise a plain iterator."""
    if use_tqdm and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False)
    return loader


def normalize_split_name(name: str) -> str:
    """
    Accepts friendly aliases and normalizes to the make_splits naming convention.
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
        raise ValueError(f"Unknown split '{name}'. Use one of: random, group, time (or *_holdout/*_stratified).")
    return aliases[s]


def resolve_split_npz_path(cfg: Dict[str, Any], split_name: str, seed: int) -> str:
    """
    Determines which split NPZ file to use.

    Expected files (from make_splits.py):
      - <output_dir>/random_stratified_seed{seed}.npz
      - <output_dir>/group_holdout_seed{seed}.npz
      - <output_dir>/time_holdout.npz
    """
    exp = cfg.get("experiment", {}) or {}
    splits_cfg = (exp.get("splits", {}) or {})
    out_dir = str(splits_cfg.get("output_dir", os.path.join("data", "processed", "splits")))

    split_name = normalize_split_name(split_name)

    if split_name == "time_holdout":
        p = os.path.join(out_dir, "time_holdout.npz")
    else:
        p = os.path.join(out_dir, f"{split_name}_seed{seed}.npz")

    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Split NPZ not found: {p}\n"
            f"Did you run: python -m src.make_splits ?"
        )
    return p


def extract_ablation_list(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reads ablations from YAML.

    Supports either schema:
      experiment.ablations[].train_qnn_overrides
    or:
      experiment.ablations[].overrides.train_qnn

    Returns list of dicts: {"name": str, "overrides": dict}
    """
    exp = cfg.get("experiment", {}) or {}
    abl = exp.get("ablations", None)
    if not isinstance(abl, list) or len(abl) == 0:
        # default minimal ablations
        return [
            {"name": "full_hybrid", "overrides": {}},
            {"name": "no_head", "overrides": {"use_head": False}},
            {"name": "no_embedder", "overrides": {"use_embedder": False}},
        ]

    out = []
    for item in abl:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "ablation")).strip()

        overrides = {}
        if "train_qnn_overrides" in item and isinstance(item["train_qnn_overrides"], dict):
            overrides = dict(item["train_qnn_overrides"])
        else:
            ov = item.get("overrides", {})
            if isinstance(ov, dict):
                t = ov.get("train_qnn", {})
                if isinstance(t, dict):
                    overrides = dict(t)

        out.append({"name": name, "overrides": overrides})

    return out


def apply_train_overrides(tcfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ablation overrides to train_qnn settings."""
    out = dict(tcfg)
    for k, v in overrides.items():
        out[k] = v
    return out


# --------------------------
# Training (single run)
# --------------------------
def train_one_run(
    *,
    cfg: Dict[str, Any],
    split_name: str,
    seed: int,
    ablation_name: str,
    ablation_overrides: Dict[str, Any],
    out_dir: str,
) -> None:
    """
    Train a single model run and save artifacts into out_dir.
    """
    ensure_dir(out_dir)

    # Effective training config = base + ablation overrides
    tcfg_base = cfg["train_qnn"]
    tcfg = apply_train_overrides(tcfg_base, ablation_overrides)

    # Seed
    set_seed(seed)

    # Load split NPZ (created by make_splits.py)
    split_npz_path = resolve_split_npz_path(cfg, split_name, seed)
    X_train, y_train_raw, X_eval, y_eval_raw = load_split_npz(split_npz_path)

    # Task collapse
    task = str(tcfg.get("task", "multiclass")).lower().strip()
    if task == "binary":
        y_train = (y_train_raw != 0).astype(np.int64)
        y_eval = (y_eval_raw != 0).astype(np.int64)
        n_classes = 2
    else:
        y_train = y_train_raw.astype(np.int64)
        y_eval = y_eval_raw.astype(np.int64)
        n_classes = 3

    # Device selection (safe fallback later)
    prefer_cuda = bool(tcfg.get("prefer_cuda", False))
    device_req = pick_device(prefer_cuda)
    
    print(f"\n[train_qnn] *=================================================================================================================================*")
    print(f"[train_qnn] split={split_name}  seed={seed}  ablation={ablation_name}")
    print(f"[train_qnn] split_npz={split_npz_path}")
    print(f"[train_qnn] X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[train_qnn] X_eval ={X_eval.shape},  y_eval ={y_eval.shape}")
    print(f"[train_qnn] device(requested)={device_req}")

    # Build model
    n_features = int(X_train.shape[1])
    model = VQCClassifier(
        n_features=n_features,
        n_layers=int(tcfg.get("n_layers", 2)),
        n_qubits=int(tcfg.get("n_qubits", n_features)),
        task=task,
        n_classes=n_classes,
        diff_method=str(tcfg.get("diff_method", "adjoint")),
        use_embedder=bool(tcfg.get("use_embedder", True)),
        embed_hidden=int(tcfg.get("embed_hidden", 64)),
        embed_layers=int(tcfg.get("embed_layers", 2)),
        angle_scale=float(tcfg.get("angle_scale", float(np.pi))),
        use_head=bool(tcfg.get("use_head", True)),
        head_hidden=int(tcfg.get("head_hidden", 32)),
    ).to(device_req)

    # Smoke test for CUDA path (PennyLane backend may not support GPU)
    device_final = device_req
    try:
        with torch.no_grad():
            dummy = torch.zeros((2, n_features), dtype=torch.float32, device=device_req)
            _ = model(dummy)
    except Exception as e:
        if device_req.type == "cuda":
            print(f"[train_qnn] CUDA path failed; falling back to CPU. Reason: {type(e).__name__}: {e}")
            device_final = torch.device("cpu")
            model = model.to(device_final)
        else:
            raise

    print(f"[train_qnn] device(final)={device_final}")
    print(f"[train_qnn] PennyLane device={getattr(model, 'pl_device_name', 'unknown')}")
    print(f"[train_qnn] diff_method={getattr(model, 'diff_method', 'unknown')}")

    # Data loaders
    batch_size = int(tcfg.get("batch_size", 32))
    eval_batch_size = int(tcfg.get("eval_batch_size", batch_size))

    # Generator for deterministic shuffle
    gen = torch.Generator()
    gen.manual_seed(seed)

    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xev_t = torch.tensor(X_eval, dtype=torch.float32)
    yev_t = torch.tensor(y_eval, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, generator=gen)
    eval_loader = DataLoader(TensorDataset(Xev_t, yev_t), batch_size=eval_batch_size, shuffle=False)

    # Class weights (helps recall/TPR under imbalance)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=device_final)
    criterion = nn.NLLLoss(weight=class_weights)

    # Optimizer
    lr = float(tcfg.get("lr", 0.002))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training controls
    epochs = int(tcfg.get("epochs", 20))
    grad_clip = float(tcfg.get("grad_clip_norm", 0.0))
    use_tqdm = True

    best_epoch = 1
    best_score = -1.0
    best_state = None

    history: List[Dict[str, Any]] = []

    # Training loop
    print("[train_qnn] Starting training...")
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()

        # ---- train ----
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for bx, by in iter_batches(train_loader, desc=f"ep{ep:02d} train", use_tqdm=use_tqdm):
            bx = bx.to(device_final)
            by = by.to(device_final)

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(bx)  # (B, n_classes)
            loss = criterion(log_probs, by)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = int(bx.size(0))
            tr_loss_sum += float(loss.item()) * bs
            pred = torch.argmax(log_probs, dim=1)
            tr_correct += int((pred == by).sum().item())
            tr_total += bs

        tr_loss = tr_loss_sum / max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # ---- eval ----
        model.eval()
        ev_loss_sum = 0.0
        ev_correct = 0
        ev_total = 0

        # For binary AUC/AP checkpointing
        all_scores: List[float] = []
        all_y: List[int] = []

        with torch.no_grad():
            for bx, by in iter_batches(eval_loader, desc=f"ep{ep:02d} eval", use_tqdm=use_tqdm):
                bx = bx.to(device_final)
                by = by.to(device_final)

                log_probs = model(bx)
                loss = criterion(log_probs, by)

                bs = int(bx.size(0))
                ev_loss_sum += float(loss.item()) * bs
                pred = torch.argmax(log_probs, dim=1)
                ev_correct += int((pred == by).sum().item())
                ev_total += bs

                probs = torch.exp(log_probs).cpu().numpy()
                yb = by.cpu().numpy().astype(int)

                if probs.shape[1] == 2:
                    score = probs[:, 1]
                else:
                    score = probs[:, 1] + probs[:, 2]

                all_scores.extend(score.tolist())
                all_y.extend(yb.tolist())

        ev_loss = ev_loss_sum / max(ev_total, 1)
        ev_acc = ev_correct / max(ev_total, 1)

        # Compute AUC/AP only for binary task
        ev_auc = float("nan")
        ev_ap = float("nan")
        if n_classes == 2:
            try:
                ev_auc = float(roc_auc_score(all_y, all_scores))
                ev_ap = float(average_precision_score(all_y, all_scores))
            except Exception:
                pass

        dt = time.perf_counter() - t0
        print(
            f"[train_qnn] Epoch {ep:02d}/{epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"eval_loss={ev_loss:.4f} eval_acc={ev_acc:.4f} | "
            f"eval_auc={ev_auc:.4f} eval_ap={ev_ap:.4f} | "
            f"time={dt:.1f}s"
        )

        history.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "eval_loss": ev_loss,
                "eval_acc": ev_acc,
                "eval_auc": ev_auc,
                "eval_ap": ev_ap,
                "time_sec": dt,
            }
        )

        # Best checkpoint selection:
        # - binary: maximize AUC
        # - multiclass: maximize eval_acc
        score = ev_auc if (n_classes == 2 and not np.isnan(ev_auc)) else ev_acc
        if score > best_score:
            best_score = float(score)
            best_epoch = ep
            best_state = deepcopy(model.state_dict())

    # --------------------------
    # Save artifacts (per-run)
    # --------------------------
    weights_path = os.path.join(out_dir, "qnn_model.pt")
    if best_state is not None:
        torch.save(best_state, weights_path)
    else:
        torch.save(model.state_dict(), weights_path)

    save_json(os.path.join(out_dir, "qnn_train_history.json"), history)

    info = model.model_info()
    info.update(
        {
            "task": task,
            "seed": int(seed),
            "split": normalize_split_name(split_name),
            "ablation": ablation_name,
            "split_npz": split_npz_path,
            "device_final": str(device_final),
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "train_size": int(len(X_train)),
            "eval_size": int(len(X_eval)),
        }
    )
    save_json(os.path.join(out_dir, "qnn_model_info.json"), info)

    # Save exact config used for this run (paper reproducibility)
    run_cfg = {
        "split": normalize_split_name(split_name),
        "seed": int(seed),
        "ablation": ablation_name,
        "ablation_overrides": ablation_overrides,
        "train_qnn_effective": tcfg,
    }
    save_json(os.path.join(out_dir, "run_config.json"), run_cfg)

    print(f"[train_qnn] Saved weights to: {weights_path}")
    print(f"[train_qnn] Saved history to: {os.path.join(out_dir, 'qnn_train_history.json')}")
    print(f"[train_qnn] Saved model info to: {os.path.join(out_dir, 'qnn_model_info.json')}")
    print(f"[train_qnn] Saved run config to: {os.path.join(out_dir, 'run_config.json')}")


# --------------------------
# Main (single run or sweep)
# --------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/cesnet_vqczti.yaml")
    ap.add_argument("--single", action="store_true", help="Run exactly one job (ignore experiment.enabled sweep).")
    ap.add_argument("--split", default=None, help="Split: random/group/time or random_stratified/group_holdout/time_holdout")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ablation", default=None, help="Ablation name (must exist in YAML ablations if sweeping).")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if "train_qnn" not in cfg:
        raise KeyError("Missing train_qnn section in YAML.")

    exp = cfg.get("experiment", {}) or {}
    sweep_enabled = bool(exp.get("enabled", False)) and (not args.single)

    # Output root
    out_root = str(exp.get("out_dir", os.path.join("artifacts", "experiments")))
    run_name = str(exp.get("run_name", "run"))
    out_root = os.path.join(out_root, run_name)
    ensure_dir(out_root)

    # Determine what to run
    if sweep_enabled:
        seeds = exp.get("seeds", [int(cfg["train_qnn"].get("seed", 42))])
        splits_cfg = (exp.get("splits", {}) or {})
        strategies = splits_cfg.get("strategies", ["random_stratified", "group_holdout", "time_holdout"])
        ablations = extract_ablation_list(cfg)
    else:
        seeds = [int(args.seed if args.seed is not None else cfg["train_qnn"].get("seed", 42))]
        strategies = [args.split or "random_stratified"]
        ablations = extract_ablation_list(cfg)

        # If user asked for a specific ablation in single mode, keep only that
        if args.ablation is not None:
            keep = [a for a in ablations if a["name"] == args.ablation]
            if keep:
                ablations = keep
            else:
                # If not found, treat as "full_hybrid"
                ablations = [{"name": args.ablation, "overrides": {}}]

    # Run all jobs
    for split_name in strategies:
        split_norm = normalize_split_name(split_name)
        for ab in ablations:
            ab_name = str(ab["name"])
            ab_over = dict(ab.get("overrides", {}))

            for seed in seeds:
                seed = int(seed)

                out_dir = os.path.join(out_root, split_norm, ab_name, f"seed_{seed}")
                train_one_run(
                    cfg=cfg,
                    split_name=split_norm,
                    seed=seed,
                    ablation_name=ab_name,
                    ablation_overrides=ab_over,
                    out_dir=out_dir,
                )

    print(f"\n[train_qnn] DONE. All outputs are under: {out_root}")


if __name__ == "__main__":
    main()