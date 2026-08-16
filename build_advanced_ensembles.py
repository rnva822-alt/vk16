from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

import run_experiments as exp
import solution as sol

ROOT = Path(__file__).resolve().parent
SEED = 42


def base_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    no_id = [c for c in frame.columns if c not in ["is_valid", "owner_id_cat", "content_id_cat"]]
    id_bases = {"id_content_owner", "id_content", "owner_claim_type_key", "owner_reason_key"}
    cat_features = [
        c for c in no_id
        if not (
            c.endswith("_history_count")
            or (c.endswith("_te") and c.split("_te")[0] in id_bases)
            or c in {"owner_seen_before", "content_seen_before", "event_elapsed_days"}
        )
    ]
    cats = [c for c in sol.CAT_FEATURES if c in cat_features]
    return no_id, cat_features, cats


def fit_cat(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame | None, y_va: np.ndarray | None, cats: list[str], sample_weight: np.ndarray | None, iterations: int | None) -> tuple[CatBoostClassifier, int]:
    model = CatBoostClassifier(
        iterations=1800 if iterations is None else iterations,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=120 if x_va is not None else None,
        thread_count=6,
        allow_writing_files=False,
    )
    if x_va is None:
        model.fit(x_tr, y_tr, cat_features=cats, sample_weight=sample_weight)
        return model, iterations or 250
    model.fit(x_tr, y_tr, cat_features=cats, eval_set=(x_va, y_va), use_best_model=True, sample_weight=sample_weight)
    return model, int(model.get_best_iteration() + 1)


def fit_lgb(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame | None, y_va: np.ndarray | None, cats: list[str], iterations: int | None) -> tuple[lgb.LGBMClassifier, int]:
    model = lgb.LGBMClassifier(
        n_estimators=1800 if iterations is None else iterations,
        learning_rate=0.03,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=35,
        colsample_bytree=0.85,
        subsample=0.85,
        reg_lambda=2.0,
        scale_pos_weight=4.0,
        objective="binary",
        random_state=SEED,
        verbosity=-1,
        n_jobs=6,
    )
    if x_va is None:
        model.fit(x_tr, y_tr, categorical_feature=cats)
        return model, iterations or 15
    model.fit(x_tr, y_tr, categorical_feature=cats, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(120, verbose=False)])
    return model, int(model.best_iteration_)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
        "positive_predictions": int(pred.sum()),
    }


def optimize(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    weights = np.arange(0.0, 1.01, 0.1)
    thresholds = np.arange(0.44, 0.681, 0.002)
    for w_all in weights:
        for w_recent in weights:
            w_lgb = round(1.0 - w_all - w_recent, 10)
            if w_lgb < 0:
                continue
            p = w_all * probabilities["cat_all"] + w_recent * probabilities["cat_recent"] + w_lgb * probabilities["lgb_idte"]
            best = None
            for threshold in thresholds:
                candidate = metrics(y, p, threshold)
                if best is None or candidate["f1"] > best["f1"]:
                    best = candidate
            rows.append(best | {"weight_cat_all": float(w_all), "weight_cat_recent": float(w_recent), "weight_lgb_idte": float(w_lgb)})
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def prepare_features(raw_train: pd.DataFrame, raw_other: pd.DataFrame, mappings: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    tr, other = exp.make_split_features(raw_train, raw_other, mappings)
    no_id, cat_features, cats = base_columns(tr)
    return tr, other, no_id, cat_features, cats


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(ROOT / "test.csv")
    mappings = sol.compute_agg_counts(train, test)
    tr_idx, va_idx = list(TimeSeriesSplit(n_splits=5).split(train))[-1]
    raw_tr = train.iloc[tr_idx].copy()
    raw_va = train.iloc[va_idx].copy()
    tr, va, no_id, cat_features, cats = prepare_features(raw_tr, raw_va, mappings)
    y_tr = tr["is_valid"].to_numpy()
    y_va = va["is_valid"].to_numpy()
    x_cat_tr = exp.prepare_cat(tr, cat_features, cats)
    x_cat_va = exp.prepare_cat(va, cat_features, cats)
    age_days = (pd.to_datetime(raw_tr["first_event_time"]).max() - pd.to_datetime(raw_tr["first_event_time"])).dt.total_seconds().to_numpy() / 86400.0
    recency_weight = np.exp(np.log(0.5) * age_days / 120.0)

    cat_all, it_all = fit_cat(x_cat_tr, y_tr, x_cat_va, y_va, cats, None, None)
    cat_recent, it_recent = fit_cat(x_cat_tr, y_tr, x_cat_va, y_va, cats, recency_weight, None)
    x_lgb_tr, x_lgb_va, lgb_cats = exp.encode_lgb(tr, va, no_id, [c for c in sol.CAT_FEATURES if c in no_id])
    lgb_model, it_lgb = fit_lgb(x_lgb_tr, y_tr, x_lgb_va, y_va, lgb_cats, None)
    val_probabilities = {
        "cat_all": cat_all.predict_proba(x_cat_va)[:, 1],
        "cat_recent": cat_recent.predict_proba(x_cat_va)[:, 1],
        "lgb_idte": lgb_model.predict_proba(x_lgb_va)[:, 1],
    }
    search = optimize(y_va, val_probabilities)
    search.to_csv(ROOT / "advanced_ensemble_search_latest_fold.csv", index=False)
    print("Top latest-fold configurations:")
    print(search.head(20).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # Final training on all rows using latest-fold selected iterations.
    full, test_features, full_no_id, full_cat_features, full_cats = prepare_features(train, test, mappings)
    y_full = full["is_valid"].to_numpy()
    x_full_cat = exp.prepare_cat(full, full_cat_features, full_cats)
    x_test_cat = exp.prepare_cat(test_features, full_cat_features, full_cats)
    full_age_days = (pd.to_datetime(train["first_event_time"]).max() - pd.to_datetime(train["first_event_time"])).dt.total_seconds().to_numpy() / 86400.0
    full_recency_weight = np.exp(np.log(0.5) * full_age_days / 120.0)
    final_all, _ = fit_cat(x_full_cat, y_full, None, None, full_cats, None, it_all)
    final_recent, _ = fit_cat(x_full_cat, y_full, None, None, full_cats, full_recency_weight, it_recent)
    x_full_lgb, x_test_lgb, final_lgb_cats = exp.encode_lgb(full, test_features, full_no_id, [c for c in sol.CAT_FEATURES if c in full_no_id])
    final_lgb, _ = fit_lgb(x_full_lgb, y_full, None, None, final_lgb_cats, it_lgb)
    test_probabilities = {
        "cat_all": final_all.predict_proba(x_test_cat)[:, 1],
        "cat_recent": final_recent.predict_proba(x_test_cat)[:, 1],
        "lgb_idte": final_lgb.predict_proba(x_test_lgb)[:, 1],
    }

    # Candidate 1: overall top configuration. Candidate 2: top configuration with >=40% recency model for diversity.
    selection = [search.iloc[0].to_dict()]
    diverse = search[search["weight_cat_recent"] >= 0.4]
    if len(diverse):
        selection.append(diverse.iloc[0].to_dict())
    else:
        selection.append(search.iloc[min(1, len(search) - 1)].to_dict())
    manifest = []
    for rank, config in enumerate(selection, 1):
        p = (
            config["weight_cat_all"] * test_probabilities["cat_all"]
            + config["weight_cat_recent"] * test_probabilities["cat_recent"]
            + config["weight_lgb_idte"] * test_probabilities["lgb_idte"]
        )
        labels = (p >= config["threshold"]).astype(int)
        filename = f"submission_advanced_{rank}.csv"
        pd.DataFrame({"claim_id": test["claim_id"], "is_valid": labels}).to_csv(ROOT / filename, index=False)
        manifest.append({
            "rank": rank,
            "filename": filename,
            "latest_fold_f1": config["f1"],
            "threshold": config["threshold"],
            "weight_cat_all": config["weight_cat_all"],
            "weight_cat_recent": config["weight_cat_recent"],
            "weight_lgb_idte": config["weight_lgb_idte"],
            "positive_predictions": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "iterations_cat_all": it_all,
            "iterations_cat_recent": it_recent,
            "iterations_lgb": it_lgb,
        })
    (ROOT / "advanced_ensemble_candidates.json").write_text(json.dumps({"candidates": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nGenerated candidates:")
    print(json.dumps({"candidates": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
