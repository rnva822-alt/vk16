"""Honest expanding-window temporal CV that caches component probabilities.

Design decisions (they fix the methodological problems of the earlier search):

1. Validation blocks have the same size as the hidden test set (7,446 rows) and are
   contiguous in time, so every fold emulates "train on the past, score the next
   ~2 months".
2. Early stopping never sees the validation block. Inside every fold the last
   `BLOCK` rows of the training prefix form an inner validation set used only to pick
   the iteration count; the component is then refit on the full prefix with that
   fixed count.
3. Nothing about the decision rule (threshold, ensemble weights) is fitted here. The
   script only stores per-fold, per-component probabilities so that decision rules can
   be compared offline on identical predictions.

Usage:
    python src/cv_probs.py --folds 4 --seeds 42,2026,777        # CV probabilities
    python src/cv_probs.py --final --seeds 42,2026,777          # test probabilities
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solution as sol  # noqa: E402

BLOCK = 7446  # hidden test size
THREADS = 2
ARTIFACTS = ROOT / "artifacts"

# Component definitions. `cat_recent` is the champion recency model (half-life 105 days,
# l2 = 9); the two LightGBM components differ in capacity/regularization to give the
# ensemble two independent error patterns.
CAT_PARAMS = {
    "cat_all": {"depth": 7, "l2_leaf_reg": 3.0, "half_life": None},
    "cat_recent": {"depth": 7, "l2_leaf_reg": 9.0, "half_life": 105.0},
}
LGB_PARAMS = {
    # champion-exact pair (documented in the experiment catalogue)
    "lgb_neutral": {
        "learning_rate": 0.03, "num_leaves": 48, "min_child_samples": 35,
        "colsample_bytree": 0.85, "subsample": 0.85, "reg_lambda": 2.0,
        "scale_pos_weight": 1.0,
    },
    "lgb_diverse": {
        "learning_rate": 0.025, "num_leaves": 64, "min_child_samples": 25,
        "colsample_bytree": 0.85, "subsample": 0.85, "reg_lambda": 3.0,
        "scale_pos_weight": 2.0,
    },
    # extra candidate: stronger regularization and a heavier positive weight
    "lgb_strong": {
        "learning_rate": 0.03, "num_leaves": 32, "min_child_samples": 60,
        "colsample_bytree": 0.70, "subsample": 0.80, "reg_lambda": 8.0,
        "scale_pos_weight": 4.0,
    },
}
MAX_ITERATIONS_CAT = 2400
MAX_ITERATIONS_LGB = 2000
ES_ROUNDS = 150


def recency_weight(times: pd.Series, half_life: float) -> np.ndarray:
    age_days = (times.max() - times).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(np.log(0.5) * age_days / half_life)


def fit_catboost(x_tr, y_tr, cats, weight, l2_leaf_reg, depth, seed, iterations, eval_set=None):
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=float(l2_leaf_reg),
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=int(seed),
        verbose=False,
        early_stopping_rounds=ES_ROUNDS if eval_set is not None else None,
        thread_count=THREADS,
        allow_writing_files=False,
    )
    if eval_set is None:
        model.fit(x_tr, y_tr, cat_features=cats, sample_weight=weight)
        return model, iterations
    model.fit(
        x_tr, y_tr, cat_features=cats, sample_weight=weight,
        eval_set=eval_set, use_best_model=True,
    )
    return model, int(model.get_best_iteration()) + 1


def fit_lightgbm(x_tr, y_tr, cats, params, seed, iterations, eval_set=None):
    model = lgb.LGBMClassifier(
        n_estimators=iterations,
        max_depth=-1,
        objective="binary",
        random_state=int(seed),
        verbosity=-1,
        n_jobs=THREADS,
        **params,
    )
    if eval_set is None:
        model.fit(x_tr, y_tr, categorical_feature=cats)
        return model, iterations
    model.fit(
        x_tr, y_tr, categorical_feature=cats, eval_set=eval_set,
        callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)],
    )
    return model, max(int(model.best_iteration_ or iterations), 20)


def build_matrices(raw_tr: pd.DataFrame, raw_va: pd.DataFrame, mappings: dict):
    """Feature engineering fitted on `raw_tr` only, applied to both frames."""
    frame_tr, frame_va, no_id, cat_features, cats = sol.prepare_features(raw_tr, raw_va, mappings)
    x_cat_tr = sol.prepare_cat(frame_tr, cat_features, cats)
    x_cat_va = sol.prepare_cat(frame_va, cat_features, cats)
    lgb_cats = [c for c in sol.CAT_FEATURES if c in no_id]
    x_lgb_tr, x_lgb_va, lgb_cat_cols = sol.encode_lgb(frame_tr, frame_va, no_id, lgb_cats)
    y_tr = frame_tr["is_valid"].to_numpy()
    y_va = frame_va["is_valid"].to_numpy() if "is_valid" in frame_va else None
    return {
        "x_cat_tr": x_cat_tr, "x_cat_va": x_cat_va, "cats": cats,
        "x_lgb_tr": x_lgb_tr, "x_lgb_va": x_lgb_va, "lgb_cats": lgb_cat_cols,
        "y_tr": y_tr, "y_va": y_va,
    }


def component_probabilities(raw_tr: pd.DataFrame, raw_va: pd.DataFrame, mappings: dict,
                            seeds: list[int], log: dict) -> dict[str, np.ndarray]:
    """Train every component on `raw_tr` and score `raw_va`.

    Iteration counts come from an inner chronological holdout carved out of `raw_tr`;
    the validation block itself is never used for early stopping or model selection.
    """
    inner_cut = max(len(raw_tr) - BLOCK, int(0.6 * len(raw_tr)))
    inner_tr = raw_tr.iloc[:inner_cut]
    inner_va = raw_tr.iloc[inner_cut:]
    inner = build_matrices(inner_tr, inner_va, mappings)
    outer = build_matrices(raw_tr, raw_va, mappings)

    inner_times = pd.to_datetime(inner_tr["first_event_time"])
    outer_times = pd.to_datetime(raw_tr["first_event_time"])
    probabilities: dict[str, np.ndarray] = {}

    for name, cfg in CAT_PARAMS.items():
        w_inner = None if cfg["half_life"] is None else recency_weight(inner_times, cfg["half_life"])
        w_outer = None if cfg["half_life"] is None else recency_weight(outer_times, cfg["half_life"])
        started = time.time()
        _, iterations = fit_catboost(
            inner["x_cat_tr"], inner["y_tr"], inner["cats"], w_inner,
            cfg["l2_leaf_reg"], cfg["depth"], seeds[0], MAX_ITERATIONS_CAT,
            eval_set=(inner["x_cat_va"], inner["y_va"]),
        )
        log[f"{name}_iterations"] = iterations
        for seed in seeds:
            model, _ = fit_catboost(
                outer["x_cat_tr"], outer["y_tr"], outer["cats"], w_outer,
                cfg["l2_leaf_reg"], cfg["depth"], seed, iterations,
            )
            probabilities[f"{name}_s{seed}"] = model.predict_proba(outer["x_cat_va"])[:, 1]
        log[f"{name}_seconds"] = round(time.time() - started, 1)
        print(f"    {name}: iterations={iterations} ({log[f'{name}_seconds']}s)", flush=True)

    for name, params in LGB_PARAMS.items():
        started = time.time()
        _, iterations = fit_lightgbm(
            inner["x_lgb_tr"], inner["y_tr"], inner["lgb_cats"], params, seeds[0],
            MAX_ITERATIONS_LGB, eval_set=[(inner["x_lgb_va"], inner["y_va"])],
        )
        log[f"{name}_iterations"] = iterations
        for seed in seeds:
            model, _ = fit_lightgbm(
                outer["x_lgb_tr"], outer["y_tr"], outer["lgb_cats"], params, seed, iterations,
            )
            probabilities[f"{name}_s{seed}"] = model.predict_proba(outer["x_lgb_va"])[:, 1]
        log[f"{name}_seconds"] = round(time.time() - started, 1)
        print(f"    {name}: iterations={iterations} ({log[f'{name}_seconds']}s)", flush=True)

    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="42,2026,777")
    parser.add_argument("--final", action="store_true", help="fit on full train, score test.csv")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ARTIFACTS.mkdir(exist_ok=True)

    train = pd.read_csv(ROOT / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(ROOT / "test.csv")
    mappings = sol.compute_agg_counts(train, test)

    if args.final:
        log: dict = {"rows_train": len(train), "rows_test": len(test), "seeds": seeds}
        print("FINAL: full train -> test.csv", flush=True)
        probabilities = component_probabilities(train, test, mappings, seeds, log)
        np.savez_compressed(
            ARTIFACTS / "probs_test.npz",
            claim_id=test["claim_id"].to_numpy(),
            **probabilities,
        )
        (ARTIFACTS / "probs_test_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(json.dumps(log, indent=2))
        return

    total = len(train)
    for fold in range(args.folds, 0, -1):
        start = total - fold * BLOCK
        stop = start + BLOCK
        if start <= BLOCK:
            print(f"skip fold {fold}: training prefix too short")
            continue
        raw_tr = train.iloc[:start].copy()
        raw_va = train.iloc[start:stop].copy()
        log = {
            "fold": fold,
            "rows_train": len(raw_tr),
            "rows_val": len(raw_va),
            "val_start": str(raw_va["first_event_time"].min()),
            "val_stop": str(raw_va["first_event_time"].max()),
            "prevalence_train": float(raw_tr["is_valid"].mean()),
            "prevalence_val": float(raw_va["is_valid"].mean()),
            "seeds": seeds,
        }
        print(f"\nFOLD {fold}: train {len(raw_tr)} rows -> val {len(raw_va)} rows "
              f"({log['val_start']} .. {log['val_stop']}), val prevalence {log['prevalence_val']:.4f}",
              flush=True)
        probabilities = component_probabilities(raw_tr, raw_va, mappings, seeds, log)
        np.savez_compressed(
            ARTIFACTS / f"probs_fold{fold}.npz",
            y=raw_va["is_valid"].to_numpy(),
            claim_id=raw_va["claim_id"].to_numpy(),
            **probabilities,
        )
        (ARTIFACTS / f"probs_fold{fold}_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8")
        print(json.dumps(log, indent=2), flush=True)


if __name__ == "__main__":
    main()
