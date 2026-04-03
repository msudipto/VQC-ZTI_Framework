"""
Preprocess CESNET-TimeSeries24-CD into a compact NPZ for VQC–ZTI.

Key idea
- CESNET-TimeSeries24-CD provides per-entity time-series CSVs of *aggregated* traffic metrics
  (e.g., 10-minute windows). Each row is already an aggregated datapoint.
- There is no (src_ip, dst_ip, src_port, ...) flow tuple in these files.

This script supports combining multiple CESNET “views” at once, e.g.:
- ip_addresses_full/agg_10_minutes/**/*.csv
- institutions/agg_10_minutes/*.csv
- institution_subnets/agg_10_minutes/*.csv

Pipeline
1) Select per-entity CSV files under `data_globs` (string glob or list of globs).
   Optionally cap the number of files per glob (max_files_per_glob) for practicality.
2) Load datapoints (rows) from selected CSVs.
   - Robustly read id_time if present (important for time split)
   - Create group_key for group split: group_key = "{entity_type}:{filename_stem}"
3) Robust-scale features by median/IQR.
4) Flag outliers using an IQR rule (and remove them for training).
5) Create pseudo-labels via quantiles on an anomaly score (unsupervised).
6) (Optional) Stratified subsampling for fast/replicable QNN training.
   - IMPORTANT: subsampling must also subsample metadata consistently.
7) Train/eval split (random stratified, default) but we also save full arrays + metadata so
   you can create:
   - group split NPZ
   - time split NPZ
8) Save NPZ:
   - X_train, y_train, X_eval, y_eval
   - X_all, y_all
   - id_time_all, group_key_all, entity_type_all
   - scaling stats

Run:
  python -m src.preprocess_cesnet
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# -----------------------
# Config
# -----------------------
def load_config(path: str) -> Dict:
    """
    Loads YAML and flattens dataset: {...} into top-level keys for backward compatibility.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Flatten dataset: {...} into top-level keys for backward compatibility
    ds = cfg.get("dataset", {})
    if isinstance(ds, dict):
        for k in ("root_dir", "data_globs", "max_files_per_glob", "random_state"):
            if k not in cfg and k in ds:
                cfg[k] = ds[k]

    # Backward-compat aliases (older config names)
    if "data_globs" not in cfg and "data_glob" in cfg:
        cfg["data_globs"] = cfg["data_glob"]
    if "max_files_per_glob" not in cfg and "max_files_cfg" in cfg:
        cfg["max_files_per_glob"] = cfg["max_files_cfg"]

    # If core_fields is not present, default it to feature_cols
    if "core_fields" not in cfg:
        cfg["core_fields"] = list(cfg.get("feature_cols", []))

    return cfg


def _normalize_list_cfg(value: Union[None, int, List[int]], n: int) -> List[Union[None, int]]:
    """Normalize an optional scalar/list config to a list of length n."""
    if value is None:
        return [None] * n
    if isinstance(value, int):
        return [value] * n
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(f"max_files_per_glob must have length {n}, got {len(value)}")
        return [int(v) if v is not None else None for v in value]
    raise TypeError("max_files_per_glob must be None, int, or list[int]")


# -----------------------
# Dataset helpers
# -----------------------
def _infer_entity_type(path: str) -> str:
    """Infer which CESNET view a CSV belongs to based on its path."""
    p = path.replace("\\", "/").lower()
    if "institution_subnets" in p:
        return "institution_subnets"
    if "/institutions/" in p or p.endswith("/institutions") or "institutions" in p:
        return "institutions"
    if "ip_addresses_full" in p:
        return "ip_addresses_full"
    if "ip_addresses_sample" in p:
        return "ip_addresses_sample"
    return "unknown"


def select_csv_files(cfg: Dict) -> List[str]:
    """
    Select CSV files based on cfg['data_globs'] with optional per-glob caps and shuffling.
    """
    patterns = cfg["data_globs"]
    if isinstance(patterns, str):
        patterns = [patterns]

    max_files_cfg = cfg.get("max_files_per_glob", None)
    max_files_per_glob = _normalize_list_cfg(max_files_cfg, len(patterns))

    shuffle_files = bool(cfg.get("shuffle_files", False))
    seed = int(cfg.get("random_state", 42))
    rng = np.random.default_rng(seed)

    selected: List[str] = []
    selected_counts = []

    for pat, cap in zip(patterns, max_files_per_glob):
        files = glob.glob(pat, recursive=True)
        files = sorted(set(files))

        if shuffle_files and files:
            files = list(rng.permutation(files))

        if cap is not None and cap > 0 and len(files) > cap:
            files = files[:cap]

        selected_counts.append((pat, len(files), cap))
        selected.extend(files)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for fp in selected:
        if fp not in seen:
            uniq.append(fp)
            seen.add(fp)

    if not uniq:
        raise FileNotFoundError(
            f"No CSVs found under data_globs={patterns}. "
            "Did you extract the CESNET tar.gz files into data/raw/cesnet_timeseries24?"
        )

    print("[preprocess] File selection:")
    for pat, n, cap in selected_counts:
        cap_str = "no cap" if cap is None else str(cap)
        print(f"  - {pat} -> selected {n} files (cap={cap_str})")
    print(f"[preprocess] Total unique CSV files selected: {len(uniq)}")

    return uniq


def load_raw_datapoints(cfg: Dict) -> pd.DataFrame:
    """
    Load CESNET per-entity aggregate CSVs into a single DataFrame.

    Key improvements for publishable evaluation:
    - Correctly loads id_time when present (needed for time split).
    - Adds stable group_key for group split: "{entity_type}:{filename_stem}".
    - Handles BOM/whitespace in column names robustly.
    - Skips non-matching CSVs cleanly (e.g., identifiers.csv).
    """
    files = select_csv_files(cfg)

    core = cfg["core_fields"]          # required for "missing core fields" filter
    feature_cols = cfg["feature_cols"] # exactly the 12 feature columns you train on
    required_cols = list(dict.fromkeys(core))  # preserve order

    max_rows = cfg.get("max_rows_per_file", None)
    if max_rows is not None:
        max_rows = int(max_rows)

    frames: List[pd.DataFrame] = []
    skipped = 0

    def _norm(c: object) -> str:
        # normalize header names: strip spaces and strip BOM (\ufeff)
        return str(c).strip().lstrip("\ufeff")

    for fp in files:
        # Read header only
        try:
            cols_actual = list(pd.read_csv(fp, nrows=0).columns)
        except Exception as e:
            print(f"[preprocess] WARN: failed reading header {fp}: {e}")
            skipped += 1
            continue

        # Map normalized -> actual header name
        col_map = {_norm(c): c for c in cols_actual}

        # Require feature/core columns to exist (normalized)
        missing_feats = [c for c in required_cols if c not in col_map]
        if missing_feats:
            skipped += 1
            continue

        # Detect time column, if any
        time_norm = None
        for cand in ("id_time", "time", "timestamp", "idTime", "id_time_bin"):
            if cand in col_map:
                time_norm = cand
                break

        # Read only the needed columns (fast and robust)
        usecols = [col_map[c] for c in required_cols]
        if time_norm is not None:
            usecols = [col_map[time_norm]] + usecols

        try:
            df = pd.read_csv(fp, usecols=usecols, nrows=max_rows)
        except Exception as e:
            print(f"[preprocess] WARN: failed reading {fp}: {e}")
            skipped += 1
            continue

        # Normalize column names in the dataframe
        df.columns = [_norm(c) for c in df.columns]

        # Ensure id_time exists (rename alternative time column -> id_time)
        if "id_time" not in df.columns:
            for alt in ("time", "timestamp", "idTime", "id_time_bin"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "id_time"})
                    break
            else:
                # If file truly has no time column, keep placeholder (-1)
                df["id_time"] = -1

        # Add metadata for group/time splits
        entity_type = _infer_entity_type(fp)
        stem = os.path.splitext(os.path.basename(fp))[0]
        df["entity_type"] = entity_type
        df["group_key"] = f"{entity_type}:{stem}"  # stable grouping id

        # Optional numeric entity_id (only when filename is numeric)
        if stem.isdigit():
            df["entity_id"] = int(stem)

        # Keep: id_time + 12 features + metadata
        keep = ["id_time"] + feature_cols + ["entity_type", "group_key"] + (["entity_id"] if "entity_id" in df.columns else [])

        missing = [c for c in keep if c not in df.columns]
        if missing:
            print(f"[preprocess] Skipping file (missing columns {missing}): {fp}")
            skipped += 1
            continue

        frames.append(df[keep])

    if not frames:
        raise RuntimeError(
            "No usable aggregate CSVs were loaded. "
            "Check that data_globs points to */agg_10_minutes/*.csv and that the archives were extracted."
        )

    raw = pd.concat(frames, ignore_index=True)
    print(f"[preprocess] Loaded datapoints: {raw.shape[0]} rows from {len(frames)} files (skipped={skipped}).")
    return raw


# -----------------------
# Feature scaling + labels
# -----------------------
def robust_scale_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Robust scaling: (x - median) / IQR per feature."""
    X = df[feature_cols].copy()

    # Ensure numeric
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    med = X.median(axis=0)
    q1 = X.quantile(0.25, axis=0)
    q3 = X.quantile(0.75, axis=0)
    iqr = (q3 - q1).replace(0, 1.0)  # avoid divide-by-zero

    X_scaled = (X - med) / iqr
    return X_scaled, med, iqr


def iqr_outlier_mask(X_scaled: pd.DataFrame, k: float = 1.5) -> np.ndarray:
    """
    Flag outliers using an IQR rule in scaled space.

    A row is an outlier if ANY feature lies outside [Q1 - k*IQR, Q3 + k*IQR].
    Returns:
      inlier_mask (np.ndarray[bool])
    """
    q1 = X_scaled.quantile(0.25, axis=0)
    q3 = X_scaled.quantile(0.75, axis=0)
    iqr = (q3 - q1).replace(0, 1.0)
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    outlier = ((X_scaled < lo) | (X_scaled > hi)).any(axis=1).to_numpy()
    return ~outlier


def quantile_labels(X_scaled: pd.DataFrame, q_susp: float, q_mal: float) -> np.ndarray:
    """
    Unsupervised, quantile-based 3-class labeling.

    We compute an anomaly score s(x) = ||x||_2 in robust-scaled feature space.

    Labeling:
      score < Q(q_susp)              -> class 0 (normal)
      Q(q_susp) <= score < Q(q_mal)  -> class 1 (suspicious)
      score >= Q(q_mal)              -> class 2 (malicious)
    """
    scores = np.linalg.norm(X_scaled.to_numpy(dtype=np.float32), axis=1)
    t_susp = np.quantile(scores, q_susp)
    t_mal = np.quantile(scores, q_mal)

    y = np.zeros_like(scores, dtype=np.int64)
    y[(scores >= t_susp) & (scores < t_mal)] = 1
    y[scores >= t_mal] = 2
    return y


# -----------------------
# Subsampling (must return indices to subsample metadata too)
# -----------------------
def stratified_subsample(
    X: np.ndarray, y: np.ndarray, total: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified subsample to a target total size (approximately preserves class proportions).

    Returns:
      X_sub, y_sub, idx_sub  (idx_sub are indices into the original arrays)
    """
    n = len(y)
    if total is None or total <= 0 or total >= n:
        idx = np.arange(n, dtype=int)
        return X, y, idx

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()
    target_counts = np.maximum(1, (props * total).astype(int))

    # Adjust rounding to hit exact total
    while target_counts.sum() < total:
        target_counts[np.argmax(props)] += 1
    while target_counts.sum() > total:
        i = int(np.argmax(target_counts))
        if target_counts[i] > 1:
            target_counts[i] -= 1

    idx_all = []
    for cls, tc in zip(classes, target_counts):
        idx = np.where(y == cls)[0]
        if tc >= len(idx):
            idx_all.append(idx)
        else:
            idx_all.append(rng.choice(idx, size=tc, replace=False))

    idx_all = np.concatenate(idx_all).astype(int)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all], idx_all


# -----------------------
# Main
# -----------------------
def main(config_path: str = "config/cesnet_vqczti.yaml") -> None:
    cfg = load_config(config_path)

    # 1) Load raw datapoints + metadata columns
    raw_df = load_raw_datapoints(cfg)

    # 2) Basic sanity check for missing core fields.
    core_fields: List[str] = cfg["core_fields"]
    missing_mask = raw_df[core_fields].isna().any(axis=1)
    print(f"[preprocess] Rows with missing core fields: {int(missing_mask.sum())}")
    df_clean = raw_df.loc[~missing_mask].reset_index(drop=True)

    # Metadata aligned with df_clean (for splits later)
    meta_cols = ["id_time", "group_key", "entity_type"] + (["entity_id"] if "entity_id" in df_clean.columns else [])
    meta_clean = df_clean[meta_cols].copy()

    # 3) Robust scale features.
    feature_cols: List[str] = cfg["feature_cols"]
    X_scaled, med, iqr = robust_scale_features(df_clean, feature_cols)

    # 4) IQR outlier flagging/removal.
    k = float(cfg.get("outlier", {}).get("k", 1.5))
    inlier_mask = iqr_outlier_mask(X_scaled, k=k)
    outliers = int((~inlier_mask).sum())
    print(f"[preprocess] Outliers flagged by IQR rule: {outliers}")

    X_scaled_in = X_scaled.loc[inlier_mask].reset_index(drop=True)
    meta_in = meta_clean.loc[inlier_mask].reset_index(drop=True)

    # Build full metadata arrays aligned to X_scaled_in
    id_time_all = pd.to_numeric(meta_in["id_time"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    group_key_all = meta_in["group_key"].astype(str).to_numpy()
    entity_type_all = meta_in["entity_type"].astype(str).to_numpy()

    # 5) Quantile labels (unsupervised)
    q_cfg = cfg["labeling"]
    y_all = quantile_labels(X_scaled_in, q_susp=float(q_cfg["q_susp"]), q_mal=float(q_cfg["q_mal"]))

    # Print label distribution
    uniq, cnt = np.unique(y_all, return_counts=True)
    print("[preprocess] Label distribution (0=norm, 1=susp, 2=mal):")
    for u, c in zip(uniq, cnt):
        print(f"  class {u}: {c} samples")

    X_all = X_scaled_in.to_numpy(dtype=np.float32)

    # 6) Optional stratified subsample (must subsample metadata too)
    sub_cfg = cfg.get("subsample", {})
    if sub_cfg.get("enabled", False):
        total = int(sub_cfg.get("total", 0))
        rs = int(sub_cfg.get("random_state", 42))
        X_all, y_all, idx_sub = stratified_subsample(X_all, y_all, total=total, random_state=rs)
        id_time_all = id_time_all[idx_sub]
        group_key_all = group_key_all[idx_sub]
        entity_type_all = entity_type_all[idx_sub]
        print(f"[preprocess] Stratified subsample to total={total}: X_all={X_all.shape}")

    # Quick sanity: time stats
    try:
        print(
            f"[preprocess] id_time stats: min={int(id_time_all.min())}, max={int(id_time_all.max())}, "
            f"unique={len(set(id_time_all.tolist()))}"
        )
    except Exception:
        pass

    # 7) Random stratified train/eval split (baseline reference split)
    split_cfg = cfg["train_eval_split"]
    test_size = split_cfg.get("eval_size", 0.20)
    rs_split = int(split_cfg.get("random_state", 42))

    idx = np.arange(len(y_all), dtype=int)
    train_idx, eval_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y_all,
        random_state=rs_split,
    )

    X_train, X_eval = X_all[train_idx], X_all[eval_idx]
    y_train, y_eval = y_all[train_idx], y_all[eval_idx]

    id_time_train, id_time_eval = id_time_all[train_idx], id_time_all[eval_idx]
    group_key_train, group_key_eval = group_key_all[train_idx], group_key_all[eval_idx]
    entity_type_train, entity_type_eval = entity_type_all[train_idx], entity_type_all[eval_idx]

    print(f"[preprocess] Final split sizes: X_train={X_train.shape}, X_eval={X_eval.shape}")

    # 8) Save NPZ with full arrays + split metadata
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "cesnet_vqczti.npz")

    np.savez(
        out_path,
        # Main split used by your current train/eval scripts (random stratified)
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,

        # Full arrays + metadata (used to create group/time split NPZs)
        X_all=X_all,
        y_all=y_all,
        id_time_all=id_time_all,
        group_key_all=group_key_all,
        entity_type_all=entity_type_all,

        # Split metadata (debugging / reporting)
        id_time_train=id_time_train,
        id_time_eval=id_time_eval,
        group_key_train=group_key_train,
        group_key_eval=group_key_eval,
        entity_type_train=entity_type_train,
        entity_type_eval=entity_type_eval,

        # Scaling stats
        med=med.to_numpy(),
        iqr=iqr.to_numpy(),
        feature_cols=np.array(feature_cols),
    )

    print(f"[preprocess] Saved processed dataset to: {out_path}")
    print("[preprocess] NOTE: NPZ now includes X_all/y_all/id_time_all/group_key_all for group/time splits.")


if __name__ == "__main__":
    main()