"""Resumable, long-running temporal-search engine for the VK F1 task.

The script evaluates regularized recency-weighted CatBoost models around the
leaderboard-confirmed ensemble. It checkpoints every completed trial and can be
stopped/restarted safely. Results are written to results/experiment_log.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

import build_advanced_ensembles as adv
import run_experiments as exp
import solution as sol

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
SEED = 42


def best_ensemble(y: np.ndarray, p_all: np.ndarray, p_rec: np.ndarray, p_lgb: np.ndarray) -> dict:
    best = None
    thresholds = np.arange(0.44, 0.681, 0.004)
    for w_all in np.arange(0.0, 0.51, 0.1):
        for w_rec in np.arange(0.4, 1.01, 0.1):
            w_lgb = round(1.0 - w_all - w_rec, 8)
            if w_lgb < 0:
                continue
            p = w_all * p_all + w_rec * p_rec + w_lgb * p_lgb
            for threshold in thresholds:
                pred = p >= threshold
                score = f1_score(y, pred, zero_division=0)
                if best is None or score > best["f1"]:
                    best = {
                        "f1": float(score),
                        "threshold": float(threshold),
                        "weight_cat_all": float(w_all),
                        "weight_cat_recent": float(w_rec),
                        "weight_lgb_idte": float(w_lgb),
                        "positive_rate": float(pred.mean()),
                        "positive_predictions": int(pred.sum()),
                    }
    return best


def fit_recency_model(x_train, y_train, x_val, y_val, cat_cols, sample_weight, l2_leaf_reg, seed):
    model = CatBoostClassifier(
        iterations=2400,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=float(l2_leaf_reg),
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=int(seed),
        verbose=False,
        early_stopping_rounds=150,
        thread_count=6,
        allow_writing_files=False,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(x_val, y_val),
        use_best_model=True,
        sample_weight=sample_weight,
    )
    return model.predict_proba(x_val)[:, 1], int(model.get_best_iteration() + 1)


def trial_grid(mode: str):
    half_lives = [75, 90, 105, 120, 135, 150, 165, 180]
    l2_values = [3, 5, 7, 9]
    seeds = [42] if mode == "fast" else [42, 2026, 777, 2027, 1337]
    for half_life in half_lives:
        for l2_leaf_reg in l2_values:
            for seed in seeds:
                yield {"half_life_days": half_life, "l2_leaf_reg": l2_leaf_reg, "seed": seed}


def append_result(result: dict, path: Path):
    row = pd.DataFrame([result])
    if path.exists():
        row.to_csv(path, mode="a", index=False, header=False)
    else:
        row.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "extended"], default="extended")
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means no count limit")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "experiment_log.csv"
    state_path = CHECKPOINT_DIR / "state.json"

    train = pd.read_csv(DATA_DIR / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv")
    mappings = sol.compute_agg_counts(train, test)
    tr_idx, va_idx = list(TimeSeriesSplit(n_splits=5).split(train))[-1]
    raw_train = train.iloc[tr_idx].copy()
    raw_val = train.iloc[va_idx].copy()
    frame_train, frame_val, no_id, cat_features, cat_cols = adv.prepare_features(raw_train, raw_val, mappings)
    y_train = frame_train["is_valid"].to_numpy()
    y_val = frame_val["is_valid"].to_numpy()
    x_cat_train = exp.prepare_cat(frame_train, cat_features, cat_cols)
    x_cat_val = exp.prepare_cat(frame_val, cat_features, cat_cols)
    x_lgb_train, x_lgb_val, lgb_cols = exp.encode_lgb(frame_train, frame_val, no_id, [c for c in sol.CAT_FEATURES if c in no_id])

    # Train stable baseline components once. The variable component is the recency model.
    cat_all, cat_all_iteration = adv.fit_cat(x_cat_train, y_train, x_cat_val, y_val, cat_cols, None, None)
    lgb_model, lgb_iteration = adv.fit_lgb(x_lgb_train, y_train, x_lgb_val, y_val, lgb_cols, None)
    p_all = cat_all.predict_proba(x_cat_val)[:, 1]
    p_lgb = lgb_model.predict_proba(x_lgb_val)[:, 1]

    done = set()
    if log_path.exists():
        old = pd.read_csv(log_path)
        done = set(zip(old["half_life_days"], old["l2_leaf_reg"], old["seed"]))

    elapsed_start = time.time()
    completed = 0
    for parameters in trial_grid(args.mode):
        key = (parameters["half_life_days"], parameters["l2_leaf_reg"], parameters["seed"])
        if key in done:
            continue
        if args.max_trials and completed >= args.max_trials:
            break
        if time.time() - elapsed_start >= args.max_hours * 3600:
            break

        event_times = pd.to_datetime(raw_train["first_event_time"])
        age_days = (event_times.max() - event_times).dt.total_seconds().to_numpy() / 86400.0
        sample_weight = np.exp(np.log(0.5) * age_days / parameters["half_life_days"])
        p_rec, best_iteration = fit_recency_model(
            x_cat_train,
            y_train,
            x_cat_val,
            y_val,
            cat_cols,
            sample_weight,
            parameters["l2_leaf_reg"],
            parameters["seed"],
        )
        score = best_ensemble(y_val, p_all, p_rec, p_lgb)
        result = parameters | score | {
            "best_iteration_recency": best_iteration,
            "baseline_cat_all_iteration": cat_all_iteration,
            "baseline_lgb_iteration": lgb_iteration,
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        }
        append_result(result, log_path)
        state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        completed += 1
        print(json.dumps(result, ensure_ascii=False), flush=True)

    if log_path.exists():
        leaderboard = pd.read_csv(log_path).sort_values("f1", ascending=False)
        leaderboard.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
        print("\nTOP 20")
        print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
