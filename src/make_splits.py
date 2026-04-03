"""
Create reproducible train/eval splits from the processed CESNET NPZ.

This script assumes preprocess_cesnet.py saved these arrays into:
  data/processed/cesnet_vqczti.npz

Required keys (paper-grade splits):
  - X_all          : (N, d)
  - y_all          : (N,)   3-class labels {0,1,2}
  - id_time_all    : (N,)   time bin / time id for time-based splits
  - group_key_all  : (N,)   group identifier for group-based splits (e.g., entity_type:entity_id)

It generates splits for:
  1) random_stratified : stratified random split (seeded)
  2) group_holdout     : group holdout split using GroupShuffleSplit (seeded)
  3) time_holdout      : chronological split using id_time_all (deterministic)

Outputs:
  data/processed/splits/
    random_stratified_seed42.npz
    group_holdout_seed42.npz
    time_holdout.npz
    ...
  plus a manifest JSON:
    data/processed/splits/splits_manifest.json

Run:
  python -m src.make_splits
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_keys(d: np.lib.npyio.NpzFile, keys: List[str], npz_path: str) -> None:
    missing = [k for k in keys if k not in d.files]
    if missing:
        raise KeyError(
            f"[make_splits] Missing keys in {npz_path}: {missing}\n"
            f"Your preprocess must save X_all/y_all/id_time_all/group_key_all.\n"
            f"Re-run: python -m src.preprocess_cesnet"
        )


def load_all_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    _require_keys(d, ["X_all", "y_all", "id_time_all", "group_key_all"], npz_path)

    X = d["X_all"].astype(np.float32)
    y = d["y_all"].astype(np.int64).ravel()
    t = d["id_time_all"].astype(np.int64).ravel()

    # group_key can be object dtype; normalize to string for safety
    g_raw = d["group_key_all"]
    if g_raw.dtype == object:
        g = np.array([str(x) for x in g_raw.tolist()], dtype=str)
    else:
        g = g_raw.astype(str)

    if len(X) != len(y) or len(y) != len(t) or len(t) != len(g):
        raise ValueError(
            f"[make_splits] Length mismatch: X={len(X)}, y={len(y)}, id_time={len(t)}, group_key={len(g)}"
        )

    return X, y, t, g


def _label_counts(y: np.ndarray) -> Dict[str, int]:
    u, c = np.unique(y, return_counts=True)
    return {str(int(uu)): int(cc) for uu, cc in zip(u, c)}


def _check_nonempty_classes(y_tr: np.ndarray, y_ev: np.ndarray) -> None:
    tr_u = set(np.unique(y_tr).tolist())
    ev_u = set(np.unique(y_ev).tolist())
    all_u = set(np.unique(np.concatenate([y_tr, y_ev])).tolist())
    if tr_u != all_u or ev_u != all_u:
        raise RuntimeError(
            f"[make_splits] Split lost a class.\n"
            f"  train classes: {sorted(tr_u)}\n"
            f"  eval  classes: {sorted(ev_u)}\n"
            f"  all   classes: {sorted(all_u)}\n"
            f"Try a different seed or adjust eval_size."
        )


def split_random_stratified(y: np.ndarray, eval_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=eval_size, random_state=seed)
    idx = np.arange(len(y))
    tr_idx, ev_idx = next(sss.split(idx, y))
    return tr_idx, ev_idx


def split_group_holdout(y: np.ndarray, groups: np.ndarray, eval_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=eval_size, random_state=seed)
    idx = np.arange(len(y))
    tr_idx, ev_idx = next(gss.split(idx, y, groups=groups))
    return tr_idx, ev_idx


def split_time_holdout(
    y: np.ndarray,
    times: np.ndarray,
    eval_size: float,
    gap_unique_steps: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic chronological split.

    We split by unique time steps to avoid leakage:
      - choose a cutoff in the sorted unique times
      - train <= cutoff
      - (optional) gap: skip next gap_unique_steps times
      - eval >= eval_start_time
    """
    times = times.astype(np.int64).ravel()
    uniq = np.unique(times)
    uniq.sort()

    if len(uniq) < 5:
        # still works, but warn by being strict
        raise RuntimeError(f"[make_splits] Too few unique id_time values: {len(uniq)}")

    # cutoff index in unique-time space
    cut = int(round((1.0 - eval_size) * len(uniq))) - 1
    cut = max(0, min(cut, len(uniq) - 2))  # keep room for eval

    train_max_time = uniq[cut]
    eval_start_idx = min(cut + 1 + max(int(gap_unique_steps), 0), len(uniq) - 1)
    eval_min_time = uniq[eval_start_idx]

    tr_idx = np.where(times <= train_max_time)[0]
    ev_idx = np.where(times >= eval_min_time)[0]

    if len(tr_idx) == 0 or len(ev_idx) == 0:
        raise RuntimeError(
            f"[make_splits] Time split produced empty set: "
            f"train={len(tr_idx)}, eval={len(ev_idx)} "
            f"(train_max_time={train_max_time}, eval_min_time={eval_min_time})"
        )

    return tr_idx, ev_idx


def save_split_npz(
    out_path: str,
    X: np.ndarray,
    y: np.ndarray,
    id_time: np.ndarray,
    group_key: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    meta: Dict,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]

    # sanity: keep all 3 classes present (important for later multiclass if needed)
    _check_nonempty_classes(y_train, y_eval)

    np.savez(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        train_idx=train_idx.astype(np.int64),
        eval_idx=eval_idx.astype(np.int64),
        id_time_train=id_time[train_idx].astype(np.int64),
        id_time_eval=id_time[eval_idx].astype(np.int64),
        group_key_train=group_key[train_idx].astype(str),
        group_key_eval=group_key[eval_idx].astype(str),
        meta=json.dumps(meta),
    )


def main(config_path: str = "config/cesnet_vqczti.yaml") -> None:
    cfg = load_config(config_path)
    exp = cfg.get("experiment", {}) or {}
    splits_cfg = (exp.get("splits", {}) or {})

    npz_path = os.path.join("data", "processed", "cesnet_vqczti.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"[make_splits] Missing processed dataset: {npz_path}. Run preprocess first.")

    X, y, t, g = load_all_npz(npz_path)

    out_dir = str(splits_cfg.get("output_dir", os.path.join("data", "processed", "splits")))
    eval_size = float(splits_cfg.get("eval_size", 0.20))
    strategies = splits_cfg.get("strategies", ["random_stratified", "group_holdout", "time_holdout"])
    seeds = exp.get("seeds", [42])

    gap = int(splits_cfg.get("time_gap_unique_steps", 0))

    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "npz_source": npz_path,
        "N": int(len(y)),
        "label_counts_all": _label_counts(y),
        "id_time_min": int(np.min(t)),
        "id_time_max": int(np.max(t)),
        "id_time_unique": int(len(np.unique(t))),
        "group_key_unique": int(len(np.unique(g))),
        "eval_size": eval_size,
        "time_gap_unique_steps": gap,
        "splits": [],
    }

    print(f"[make_splits] Loaded X_all={X.shape}, y_all={y.shape}")
    print(f"[make_splits] id_time: min={int(t.min())}, max={int(t.max())}, unique={len(np.unique(t))}")
    print(f"[make_splits] group_key unique: {len(np.unique(g))}")
    print(f"[make_splits] Writing splits to: {out_dir}")

    strategies = list(strategies)

    for strat in strategies:
        strat = str(strat).strip().lower()

        if strat in ("random_stratified", "group_holdout"):
            for seed in seeds:
                seed = int(seed)

                if strat == "random_stratified":
                    tr_idx, ev_idx = split_random_stratified(y, eval_size=eval_size, seed=seed)
                else:
                    tr_idx, ev_idx = split_group_holdout(y, groups=g, eval_size=eval_size, seed=seed)

                meta = {
                    "strategy": strat,
                    "seed": seed,
                    "eval_size": eval_size,
                }

                out_path = os.path.join(out_dir, f"{strat}_seed{seed}.npz")
                save_split_npz(out_path, X, y, t, g, tr_idx, ev_idx, meta)

                manifest["splits"].append(
                    {
                        "strategy": strat,
                        "seed": seed,
                        "path": out_path.replace("\\", "/"),
                        "n_train": int(len(tr_idx)),
                        "n_eval": int(len(ev_idx)),
                        "label_counts_train": _label_counts(y[tr_idx]),
                        "label_counts_eval": _label_counts(y[ev_idx]),
                    }
                )

                print(f"[make_splits] Wrote {strat}_seed{seed}: train={len(tr_idx)}, eval={len(ev_idx)}")

        elif strat == "time_holdout":
            tr_idx, ev_idx = split_time_holdout(y, times=t, eval_size=eval_size, gap_unique_steps=gap)

            meta = {
                "strategy": strat,
                "seed": None,
                "eval_size": eval_size,
                "time_gap_unique_steps": gap,
            }

            out_path = os.path.join(out_dir, "time_holdout.npz")
            save_split_npz(out_path, X, y, t, g, tr_idx, ev_idx, meta)

            manifest["splits"].append(
                {
                    "strategy": strat,
                    "seed": None,
                    "path": out_path.replace("\\", "/"),
                    "n_train": int(len(tr_idx)),
                    "n_eval": int(len(ev_idx)),
                    "label_counts_train": _label_counts(y[tr_idx]),
                    "label_counts_eval": _label_counts(y[ev_idx]),
                }
            )

            print(f"[make_splits] Wrote time_holdout: train={len(tr_idx)}, eval={len(ev_idx)}")

        else:
            raise ValueError(f"[make_splits] Unknown split strategy: {strat}")

    manifest_path = os.path.join(out_dir, "splits_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[make_splits] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()