from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, TimeSeriesSplit

import vk_pipeline_core as sol

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SEED = 42
TE_ALPHA = 15.0
N_SPLITS = 5


def add_raw_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["owner_claim_type_key"] = out["id_content_owner"].astype(str) + "__" + out["claim_type"].astype(str)
    out["owner_reason_key"] = out["id_content_owner"].astype(str) + "__" + out["claim_reason_start"].astype(str)
    return out


def oof_target_encode(train_raw: pd.DataFrame, val_raw: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_raw.copy()
    va = val_raw.copy()
    global_mean = float(tr["is_valid"].mean())
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for col in columns:
        name = f"{col}_te"
        oof = np.full(len(tr), global_mean, dtype=float)
        for fit_idx, hold_idx in kf.split(tr):
            fit = tr.iloc[fit_idx]
            stats = fit.groupby(col)["is_valid"].agg(["sum", "count"])
            mapping = ((stats["sum"] + TE_ALPHA * global_mean) / (stats["count"] + TE_ALPHA)).to_dict()
            oof[hold_idx] = tr.iloc[hold_idx][col].map(mapping).fillna(global_mean).to_numpy()
        stats_full = tr.groupby(col)["is_valid"].agg(["sum", "count"])
        mapping_full = ((stats_full["sum"] + TE_ALPHA * global_mean) / (stats_full["count"] + TE_ALPHA)).to_dict()
        tr[name] = oof
        va[name] = va[col].map(mapping_full).fillna(global_mean).to_numpy()

        count_name = f"{col}_history_count"
        count_map = tr.groupby(col).size().to_dict()
        tr[count_name] = tr[col].map(count_map).astype(float)
        va[count_name] = va[col].map(count_map).fillna(0).astype(float)
    return tr, va


def make_split_features(raw_train: pd.DataFrame, raw_val: pd.DataFrame, agg_maps: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_with_keys = add_raw_keys(raw_train)
    val_with_keys = add_raw_keys(raw_val)

    base_train = sol.engineer_features(raw_train, agg_maps)
    base_val = sol.engineer_features(raw_val, agg_maps)
    base_train = sol.kfold_target_encoding(base_train, "is_valid", sol.TE_ALPHA)
    # Validation TE must come from the training block without leaking validation targets.
    base_val = sol.apply_te_to_new_data(base_train, base_val, "is_valid", sol.TE_ALPHA)

    te_cols = ["id_content_owner", "id_content", "owner_claim_type_key", "owner_reason_key"]
    raw_te_train, raw_te_val = oof_target_encode(train_with_keys, val_with_keys, te_cols)
    for col in te_cols:
        for suffix in ["_te", "_history_count"]:
            feature = f"{col}{suffix}"
            base_train[feature] = raw_te_train[feature].to_numpy()
            base_val[feature] = raw_te_val[feature].to_numpy()

    train_dt = pd.to_datetime(raw_train["first_event_time"])
    val_dt = pd.to_datetime(raw_val["first_event_time"])
    origin = train_dt.min()
    base_train["event_elapsed_days"] = (train_dt - origin).dt.total_seconds().to_numpy() / 86400.0
    base_val["event_elapsed_days"] = (val_dt - origin).dt.total_seconds().to_numpy() / 86400.0

    base_train["owner_seen_before"] = 1
    base_val["owner_seen_before"] = raw_val["id_content_owner"].isin(set(raw_train["id_content_owner"])).astype(int).to_numpy()
    base_train["content_seen_before"] = 1
    base_val["content_seen_before"] = raw_val["id_content"].isin(set(raw_train["id_content"])).astype(int).to_numpy()

    # Raw identifiers are kept separately for native-CatBoost experiments only.
    base_train["owner_id_cat"] = raw_train["id_content_owner"].astype(str).to_numpy()
    base_val["owner_id_cat"] = raw_val["id_content_owner"].astype(str).to_numpy()
    base_train["content_id_cat"] = raw_train["id_content"].astype(str).to_numpy()
    base_val["content_id_cat"] = raw_val["id_content"].astype(str).to_numpy()

    return base_train, base_val


def prepare_cat(df: pd.DataFrame, columns: list[str], cat_columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for col in cat_columns:
        if col in out:
            out[col] = out[col].astype(str)
    return out


def optimize_threshold(y: np.ndarray, p: np.ndarray) -> dict:
    best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0, "positive_rate": 0.0}
    for threshold in np.arange(0.05, 0.951, 0.002):
        pred = (p >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best["f1"]:
            best = {
                "f1": float(score),
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "positive_rate": float(pred.mean()),
            }
    return best


def fit_cat(name: str, x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray, cat_cols: list[str], params: dict) -> tuple[str, np.ndarray, dict]:
    model = CatBoostClassifier(
        iterations=1800,
        learning_rate=0.035,
        depth=params.get("depth", 7),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=120,
        thread_count=params.get("thread_count", 6),
        **params.get("weights", {}),
    )
    model.fit(x_tr, y_tr, cat_features=cat_cols, eval_set=(x_va, y_va), use_best_model=True)
    p = model.predict_proba(x_va)[:, 1]
    score = optimize_threshold(y_va, p)
    score["best_iteration"] = int(model.get_best_iteration() + 1)
    score["model"] = name
    return name, p, score


def fit_lgb(name: str, x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray, cat_cols: list[str]) -> tuple[str, np.ndarray, dict]:
    model = lgb.LGBMClassifier(
        n_estimators=1800,
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
    model.fit(
        x_tr,
        y_tr,
        categorical_feature=cat_cols,
        eval_set=[(x_va, y_va)],
        callbacks=[lgb.early_stopping(120, verbose=False)],
    )
    p = model.predict_proba(x_va)[:, 1]
    score = optimize_threshold(y_va, p)
    score["best_iteration"] = int(model.best_iteration_)
    score["model"] = name
    return name, p, score


def encode_lgb(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = train[feature_cols].copy()
    va = val[feature_cols].copy()
    valid_cats = [c for c in cat_cols if c in feature_cols]
    for col in valid_cats:
        values = pd.concat([tr[col], va[col]], ignore_index=True).astype(str).unique()
        mapping = {v: i for i, v in enumerate(values)}
        tr[col] = tr[col].astype(str).map(mapping).astype("category")
        va[col] = va[col].astype(str).map(mapping).astype("category")
    return tr, va, valid_cats


def run_fold(fold_name: str, raw_train: pd.DataFrame, raw_val: pd.DataFrame, agg_maps: dict) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    train_fe, val_fe = make_split_features(raw_train, raw_val, agg_maps)
    y_tr = train_fe["is_valid"].to_numpy()
    y_va = val_fe["is_valid"].to_numpy()

    no_id_cols = [c for c in train_fe.columns if c not in ["is_valid", "owner_id_cat", "content_id_cat"]]
    with_owner_cols = no_id_cols + ["owner_id_cat"]
    with_both_ids_cols = with_owner_cols + ["content_id_cat"]
    base_cat = [c for c in sol.CAT_FEATURES if c in no_id_cols]

    model_specs = [
        ("cat_base", no_id_cols, base_cat, {"depth": 7, "l2_leaf_reg": 3.0, "weights": {"auto_class_weights": "Balanced"}}),
        ("cat_id_te", no_id_cols, base_cat, {"depth": 7, "l2_leaf_reg": 4.0, "weights": {"auto_class_weights": "Balanced"}}),
        ("cat_id_te_owner_cat", with_owner_cols, base_cat + ["owner_id_cat"], {"depth": 7, "l2_leaf_reg": 5.0, "weights": {"auto_class_weights": "Balanced"}}),
        ("cat_id_te_both_ids", with_both_ids_cols, base_cat + ["owner_id_cat", "content_id_cat"], {"depth": 7, "l2_leaf_reg": 6.0, "weights": {"auto_class_weights": "Balanced"}}),
        ("cat_id_te_weight3", no_id_cols, base_cat, {"depth": 7, "l2_leaf_reg": 5.0, "weights": {"class_weights": [1.0, 3.0]}}),
        ("cat_id_te_depth8", no_id_cols, base_cat, {"depth": 8, "l2_leaf_reg": 6.0, "weights": {"auto_class_weights": "Balanced"}}),
    ]

    records: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    for name, features, cats, params in model_specs:
        if name == "cat_base":
            # The true baseline excludes all newly introduced ID / history signals.
            baseline_features = [c for c in no_id_cols if not (c.endswith("_history_count") or c.endswith("_te") and c.split("_te")[0] in {"id_content_owner", "id_content", "owner_claim_type_key", "owner_reason_key"} or c in {"owner_seen_before", "content_seen_before", "event_elapsed_days"})]
            features = baseline_features
            cats = [c for c in base_cat if c in features]
        x_tr = prepare_cat(train_fe, features, cats)
        x_va = prepare_cat(val_fe, features, cats)
        _, pred, record = fit_cat(name, x_tr, y_tr, x_va, y_va, cats, params)
        record["fold"] = fold_name
        record["n_features"] = len(features)
        records.append(record)
        preds[name] = pred

    # LightGBM only gets engineered numeric target-encoding features, not raw IDs.
    lgb_features = no_id_cols
    lgb_tr, lgb_va, lgb_cats = encode_lgb(train_fe, val_fe, lgb_features, base_cat)
    _, pred, record = fit_lgb("lgb_id_te", lgb_tr, y_tr, lgb_va, y_va, lgb_cats)
    record["fold"] = fold_name
    record["n_features"] = len(lgb_features)
    records.append(record)
    preds["lgb_id_te"] = pred

    # Test a small, pre-declared set of blends instead of choosing only a single base learner.
    for left, right in [("cat_base", "lgb_id_te"), ("cat_id_te", "lgb_id_te"), ("cat_id_te_owner_cat", "lgb_id_te"), ("cat_id_te_depth8", "lgb_id_te")]:
        best_blend = None
        for weight_left in np.arange(0.0, 1.01, 0.1):
            blend = weight_left * preds[left] + (1.0 - weight_left) * preds[right]
            score = optimize_threshold(y_va, blend)
            if best_blend is None or score["f1"] > best_blend["f1"]:
                best_blend = score | {"model": f"blend_{left}_{right}", "weight_left": float(weight_left), "fold": fold_name, "n_features": None}
                preds[f"blend_{left}_{right}"] = blend
        records.append(best_blend)

    return records, preds, y_va


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(ROOT / "test.csv")
    agg_maps = sol.compute_agg_counts(train, test)

    splitter = TimeSeriesSplit(n_splits=5)
    splits = list(splitter.split(train))
    selected = [("penultimate", splits[-2]), ("latest", splits[-1])]
    all_records: list[dict] = []
    all_predictions: dict[str, dict[str, np.ndarray]] = {}

    for fold_name, (train_idx, val_idx) in selected:
        print(f"Running {fold_name}: train={len(train_idx)}, validation={len(val_idx)}")
        records, preds, _ = run_fold(fold_name, train.iloc[train_idx].copy(), train.iloc[val_idx].copy(), agg_maps)
        all_records.extend(records)
        all_predictions[fold_name] = preds
        for row in sorted(records, key=lambda r: r["f1"], reverse=True):
            print(f"{fold_name:11s} {row['model']:40s} F1={row['f1']:.5f} thr={row['threshold']:.3f} pos={row['positive_rate']:.3f}")

    result = pd.DataFrame(all_records)
    result.to_csv(ROOT / "experiment_results.csv", index=False)
    summary = (
        result.pivot_table(index="model", columns="fold", values="f1", aggfunc="max")
        .assign(mean_recent=lambda x: x.mean(axis=1))
        .sort_values(["latest", "mean_recent"], ascending=False)
    )
    summary.to_csv(ROOT / "experiment_summary.csv")
    payload = {
        "records": all_records,
        "summary": summary.reset_index().where(pd.notna(summary.reset_index()), None).to_dict(orient="records"),
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(summary.round(6).to_string())


if __name__ == "__main__":
    main()
