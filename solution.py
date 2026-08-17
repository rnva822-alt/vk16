"""Reproducible VK tabular solution: feature engineering, training, and submission."""

from __future__ import annotations

import json
import joblib
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
N_SPLITS = 5

# Two smoothing values are retained from the original pipelines: 10 for column TE,
# 15 for entity-level OOF TE.
TE_ALPHA = 10
RAW_TE_ALPHA = 15.0

TE_COLUMNS = [
    "claim_type",
    "claim_reason_start",
    "claim_user_registered_phone_country_id",
    "platform",
]


AGE_MAP = {
    "0_13": 0, "14_17": 1, "18_24": 2, "25_34": 3,
    "35_44": 4, "45_54": 5, "55_64": 6, "65_plus": 7,
}


FRIENDS_MAP = {
    "0": 0, "1_5": 1, "6_20": 2, "21_50": 3, "51_100": 4,
    "101_250": 5, "251_500": 6, "501_1000": 7, "1001_plus": 8,
}


DROP_COLS = [
    "claim_id", "id_content", "first_event_time", "content_registered_time",
    "ip_country_id", "id_content_owner", "age_bucket", "claim_user_age_bucket",
    "friends_bucket", "claim_user_friends_bucket",
]


CAT_FEATURES = [
    "os", "platform", "sex", "claim_user_sex",
    "claim_type", "claim_reason_start",
    "registered_phone_country_id", "mobile_phone_country_id", "profile_country_id",
    "claim_user_registered_phone_country_id", "claim_user_profile_country_id",
    "claim_type_reason",
]


STRING_CAT_COLS = ["os", "claim_type_reason"]


def compute_agg_counts(train, test):
    """Compute aggregated count, frequency, and statistical features on combined train+test.

    These use only non-target information (groupby sizes, means, stds), so combining
    train+test is safe and gives both sets the same statistics.
    Useful: content owners with many complaints may be systematically different.
    """
    print("\n" + "=" * 70)
    print("Computing aggregated count/frequency/statistical features on combined train+test...")
    print("=" * 70)

    combined = pd.concat([train.drop(columns=["is_valid"]), test], axis=0, ignore_index=True)
    total_rows = len(combined)

    # --- Count features ---
    agg_mappings = {
        "content_owner_complaint_count": combined.groupby("id_content_owner").size().to_dict(),
        "content_complaint_count": combined.groupby("id_content").size().to_dict(),
        "claim_type_count": combined.groupby("claim_type").size().to_dict(),
        "claim_reason_count": combined.groupby("claim_reason_start").size().to_dict(),
    }

    # --- Frequency encoding (count / total) ---
    agg_mappings["id_content_owner_freq"] = {k: v / total_rows for k, v in agg_mappings["content_owner_complaint_count"].items()}
    agg_mappings["claim_type_freq"] = {k: v / total_rows for k, v in agg_mappings["claim_type_count"].items()}
    agg_mappings["claim_reason_freq"] = {k: v / total_rows for k, v in agg_mappings["claim_reason_count"].items()}
    agg_mappings["os_freq"] = (combined.groupby("os").size() / total_rows).to_dict()
    agg_mappings["registered_phone_country_freq"] = (combined.groupby("registered_phone_country_id").size() / total_rows).to_dict()

    # --- Statistical aggregations (computed on combined train+test, no target) ---
    agg_mappings["mean_bot_score_by_claim_type"] = combined.groupby("claim_type")["claim_user_bot_prediction_score"].mean().to_dict()
    agg_mappings["mean_bot_score_by_reason"] = combined.groupby("claim_reason_start")["claim_user_bot_prediction_score"].mean().to_dict()
    agg_mappings["mean_user_bot_by_claim_type"] = combined.groupby("claim_type")["user_bot_prediction_score"].mean().to_dict()
    agg_mappings["std_bot_score_by_claim_type"] = combined.groupby("claim_type")["claim_user_bot_prediction_score"].std().fillna(0).to_dict()
    agg_mappings["mean_likes_by_claim_type"] = combined.groupby("claim_type")["additional_likes_count"].mean().to_dict()

    print(f"  content_owner_complaint_count: {len(agg_mappings['content_owner_complaint_count'])} unique owners")
    print(f"  content_complaint_count:       {len(agg_mappings['content_complaint_count'])} unique contents")
    print(f"  claim_type_count:              {len(agg_mappings['claim_type_count'])} unique types")
    print(f"  claim_reason_count:            {len(agg_mappings['claim_reason_count'])} unique reasons")
    print("  Frequency encoding: 5 features (id_content_owner, claim_type, claim_reason, os, registered_phone_country)")
    print("  Statistical aggregations: 5 features (mean/std bot scores, mean likes by claim_type/reason)")

    return agg_mappings


def engineer_features(df, agg_mappings):
    """Perform all feature engineering on a dataframe.

    Returns a DataFrame with engineered features (datetime/dropped columns removed).
    """
    df = df.copy()

    # --- 1. Datetime parsing ---
    df["first_event_time"] = pd.to_datetime(df["first_event_time"])
    df["content_registered_time"] = pd.to_datetime(df["content_registered_time"])

    # --- 2. Temporal features from first_event_time ---
    # Useful: complaint timing patterns (e.g., late-night complaints more likely invalid)
    df["hour"] = df["first_event_time"].dt.hour
    df["dayofweek"] = df["first_event_time"].dt.dayofweek
    df["day"] = df["first_event_time"].dt.day
    df["month"] = df["first_event_time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- 3. Temporal features from content_registered_time ---
    # Useful: content age at complaint time — very new or very old content may differ
    df["content_hour"] = df["content_registered_time"].dt.hour
    df["content_dayofweek"] = df["content_registered_time"].dt.dayofweek
    df["content_month"] = df["content_registered_time"].dt.month
    df["content_age_days"] = (
        (df["first_event_time"] - df["content_registered_time"]).dt.total_seconds() / 86400
    )
    df["content_age_days"] = df["content_age_days"].clip(lower=0)

    # --- 4. Account age features ---
    # Useful: newer accounts may file more invalid complaints
    df["sender_account_age"] = (df["first_event_time"].dt.year - df["registered_year"]).clip(lower=0)
    df["claim_user_account_age"] = (df["first_event_time"].dt.year - df["claim_user_registered_year"]).clip(lower=0)

    # --- 5. Bot score features ---
    # Useful: bot-like behavior (automated spam complaints) strongly correlates with invalidity
    df["bot_score_diff"] = df["claim_user_bot_prediction_score"] - df["user_bot_prediction_score"]
    df["bot_score_sum"] = df["claim_user_bot_prediction_score"] + df["user_bot_prediction_score"]
    df["claim_user_is_high_bot"] = (df["claim_user_bot_prediction_score"] > 0.5).astype(int)
    df["user_is_high_bot"] = (df["user_bot_prediction_score"] > 0.5).astype(int)

    # --- 6. Country match features ---
    # Useful: mismatched registration/profile countries may indicate suspicious accounts
    df["country_phone_match"] = (
        df["registered_phone_country_id"] == df["claim_user_registered_phone_country_id"]
    ).astype(int)
    df["country_profile_match"] = (
        df["profile_country_id"] == df["claim_user_profile_country_id"]
    ).astype(int)

    # --- 7. Ordinal encoding for age_bucket and friends_bucket ---
    df["age_bucket_encoded"] = df["age_bucket"].map(AGE_MAP)
    df["claim_user_age_bucket_encoded"] = df["claim_user_age_bucket"].map(AGE_MAP)
    df["friends_bucket_encoded"] = df["friends_bucket"].map(FRIENDS_MAP)
    df["claim_user_friends_bucket_encoded"] = df["claim_user_friends_bucket"].map(FRIENDS_MAP)

    # --- 8. Aggregated count features ---
    # Useful: content with many complaints or repeat complainers have different validity rates
    df["content_owner_complaint_count"] = df["id_content_owner"].map(agg_mappings["content_owner_complaint_count"]).fillna(1)
    df["content_complaint_count"] = df["id_content"].map(agg_mappings["content_complaint_count"]).fillna(1)
    df["claim_type_count"] = df["claim_type"].map(agg_mappings["claim_type_count"]).fillna(1)
    df["claim_reason_count"] = df["claim_reason_start"].map(agg_mappings["claim_reason_count"]).fillna(1)

    # --- 9. Log-transform skewed counters ---
    # Useful: count features are right-skewed; log1p stabilizes variance and reduces outlier influence
    df["log_additional_likes"] = np.log1p(df["additional_likes_count"])
    df["log_additional_friend_request"] = np.log1p(df["additional_friend_request_count"])
    df["log_additional_friend_accept"] = np.log1p(df["additional_friend_accept_count"])

    # --- 10. Interaction features ---
    # Useful: certain type+reason combinations have very different validity rates
    df["claim_type_reason"] = df["claim_type"].astype(str) + "_" + df["claim_reason_start"].astype(str)
    df["sex_match"] = (df["sex"] == df["claim_user_sex"]).astype(int)
    df["age_diff"] = df["claim_user_age_bucket_encoded"] - df["age_bucket_encoded"]
    df["friends_diff"] = df["claim_user_friends_bucket_encoded"] - df["friends_bucket_encoded"]

    # --- 10b. Frequency encoding (for high-cardinality categoricals) ---
    # Useful: how common each category is in the overall population
    df["id_content_owner_freq"] = df["id_content_owner"].map(agg_mappings.get("id_content_owner_freq", {})).fillna(0)
    df["claim_type_freq"] = df["claim_type"].map(agg_mappings.get("claim_type_freq", {})).fillna(0)
    df["claim_reason_freq"] = df["claim_reason_start"].map(agg_mappings.get("claim_reason_freq", {})).fillna(0)
    df["os_freq"] = df["os"].map(agg_mappings.get("os_freq", {})).fillna(0)
    df["registered_phone_country_freq"] = df["registered_phone_country_id"].map(agg_mappings.get("registered_phone_country_freq", {})).fillna(0)

    # --- 10c. Statistical aggregations (computed on train+test combined, no target) ---
    # Useful: contextual statistics that capture category-level tendencies
    df["mean_bot_score_by_claim_type"] = df["claim_type"].map(agg_mappings.get("mean_bot_score_by_claim_type", {})).fillna(0)
    df["mean_bot_score_by_reason"] = df["claim_reason_start"].map(agg_mappings.get("mean_bot_score_by_reason", {})).fillna(0)
    df["mean_user_bot_by_claim_type"] = df["claim_type"].map(agg_mappings.get("mean_user_bot_by_claim_type", {})).fillna(0)
    df["std_bot_score_by_claim_type"] = df["claim_type"].map(agg_mappings.get("std_bot_score_by_claim_type", {})).fillna(0)
    df["mean_likes_by_claim_type"] = df["claim_type"].map(agg_mappings.get("mean_likes_by_claim_type", {})).fillna(0)

    # --- 10d. Time-based features ---
    # Useful: complaints at different times of day have different validity rates
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["content_age_bucket"] = pd.cut(
        df["content_age_days"],
        bins=[-np.inf, 1, 7, 30, 365, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["is_recent_content"] = (df["content_age_days"] <= 7).astype(int)

    # --- 10e. Registration features ---
    # Useful: registration year differences may indicate account legitimacy
    df["registration_year_diff"] = df["claim_user_registered_year"] - df["registered_year"]
    df["is_same_registration_year"] = (df["registered_year"] == df["claim_user_registered_year"]).astype(int)
    df["sender_account_age_squared"] = df["sender_account_age"] ** 2

    # --- 10f. More interaction features ---
    # Useful: additional non-linear interactions between key features
    df["claim_type_x_platform"] = df["claim_type"] * 10 + df["platform"]
    df["bot_score_x_are_friends"] = df["claim_user_bot_prediction_score"] * df["are_friends"]
    df["content_age_x_claim_type"] = df["content_age_days"] * df["claim_type"] / 1000.0
    df["is_both_high_bot"] = ((df["user_is_high_bot"] == 1) & (df["claim_user_is_high_bot"] == 1)).astype(int)
    df["friends_diff_abs"] = (df["claim_user_friends_bucket_encoded"] - df["friends_bucket_encoded"]).abs()

    # --- 11. Drop columns ---
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def kfold_target_encoding(train_df, target_col="is_valid", alpha=TE_ALPHA,
                           n_splits=5, random_state=RANDOM_SEED):
    """Apply K-fold target encoding to training data to prevent leakage.

    Each row's target encoding is computed from the OTHER folds, ensuring
    no row sees its own label in its TE feature.

    Formula: smoothed = (n * cat_mean + alpha * global_mean) / (n + alpha)
    """
    df = train_df.copy()
    global_mean = df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in TE_COLUMNS:
        te_col = f"{col}_te"
        te_values = np.full(len(df), global_mean, dtype=np.float64)

        for tr_idx, va_idx in kf.split(df):
            fold_train = df.iloc[tr_idx]
            stats = fold_train.groupby(col)[target_col].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
            mapping = smoothed.to_dict()
            te_values[va_idx] = df.iloc[va_idx][col].map(mapping).fillna(global_mean).values

        df[te_col] = te_values

    return df


def apply_te_to_new_data(train_df, new_df, target_col="is_valid", alpha=TE_ALPHA):
    """Compute TE on full training set and apply to validation/test data.

    This is correct since validation/test labels are not used.
    """
    new_df = new_df.copy()
    global_mean = train_df[target_col].mean()

    for col in TE_COLUMNS:
        stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
        mapping = smoothed.to_dict()
        te_col = f"{col}_te"
        new_df[te_col] = new_df[col].map(mapping).fillna(global_mean)

    return new_df


def add_raw_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["owner_claim_type_key"] = out["id_content_owner"].astype(str) + "__" + out["claim_type"].astype(str)
    out["owner_reason_key"] = out["id_content_owner"].astype(str) + "__" + out["claim_reason_start"].astype(str)
    return out


def oof_target_encode(train_raw: pd.DataFrame, val_raw: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_raw.copy()
    va = val_raw.copy()
    global_mean = float(tr["is_valid"].mean())
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    for col in columns:
        name = f"{col}_te"
        oof = np.full(len(tr), global_mean, dtype=float)
        for fit_idx, hold_idx in kf.split(tr):
            fit = tr.iloc[fit_idx]
            stats = fit.groupby(col)["is_valid"].agg(["sum", "count"])
            mapping = ((stats["sum"] + RAW_TE_ALPHA * global_mean) / (stats["count"] + RAW_TE_ALPHA)).to_dict()
            oof[hold_idx] = tr.iloc[hold_idx][col].map(mapping).fillna(global_mean).to_numpy()
        stats_full = tr.groupby(col)["is_valid"].agg(["sum", "count"])
        mapping_full = ((stats_full["sum"] + RAW_TE_ALPHA * global_mean) / (stats_full["count"] + RAW_TE_ALPHA)).to_dict()
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

    base_train = engineer_features(raw_train, agg_maps)
    base_val = engineer_features(raw_val, agg_maps)
    base_train = kfold_target_encoding(base_train, "is_valid", TE_ALPHA)
    base_val = apply_te_to_new_data(engineer_features(raw_train, agg_maps), base_val, "is_valid", TE_ALPHA)

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
    cats = [c for c in CAT_FEATURES if c in cat_features]
    return no_id, cat_features, cats




def prepare_features(raw_train: pd.DataFrame, raw_other: pd.DataFrame, mappings: dict):
    train_features, other_features = make_split_features(raw_train, raw_other, mappings)
    no_id, cat_features, cats = base_columns(train_features)
    return train_features, other_features, no_id, cat_features, cats


BLOCK = 7446
THREADS = 2
CAT_PARAMS = {
    "cat_all": {"depth": 7, "l2_leaf_reg": 3.0, "half_life": None},
    "cat_recent": {"depth": 7, "l2_leaf_reg": 9.0, "half_life": 105.0},
}
LGB_PARAMS = {
    "lgb_neutral": {"learning_rate": 0.03, "num_leaves": 48, "min_child_samples": 35,
                     "colsample_bytree": 0.85, "subsample": 0.85, "reg_lambda": 2.0,
                     "scale_pos_weight": 1.0},
    "lgb_diverse": {"learning_rate": 0.025, "num_leaves": 64, "min_child_samples": 25,
                     "colsample_bytree": 0.85, "subsample": 0.85, "reg_lambda": 3.0,
                     "scale_pos_weight": 2.0},
}
MAX_ITERATIONS_CAT = 2400
MAX_ITERATIONS_LGB = 2000
ES_ROUNDS = 150
SAMPLE_F1 = 0.2347083926
TEST_ROWS = 7446
TEST_POSITIVES = round(TEST_ROWS * SAMPLE_F1 / (2 - SAMPLE_F1))
RATE_MULTIPLIER = 1.8
BLEND_WEIGHTS = {
    "cat_all": 0.25,
    "cat_recent": 0.25,
    "lgb_neutral": 0.25,
    "lgb_diverse": 0.25,
}
MODEL_SIZE_LIMIT_BYTES = 15 * 1024 * 1024


def recency_weight(times: pd.Series, half_life: float) -> np.ndarray:
    age_days = (times.max() - times).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(np.log(0.5) * age_days / half_life)


def fit_catboost(x_tr, y_tr, cats, weight, l2_leaf_reg, depth, seed, iterations, eval_set=None):
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=0.03, depth=depth,
        l2_leaf_reg=float(l2_leaf_reg), loss_function="Logloss", eval_metric="Logloss",
        auto_class_weights="Balanced", random_seed=int(seed), verbose=False,
        early_stopping_rounds=ES_ROUNDS if eval_set is not None else None,
        thread_count=THREADS, allow_writing_files=False,
    )
    if eval_set is None:
        model.fit(x_tr, y_tr, cat_features=cats, sample_weight=weight)
        return model, iterations
    model.fit(x_tr, y_tr, cat_features=cats, sample_weight=weight,
              eval_set=eval_set, use_best_model=True)
    return model, int(model.get_best_iteration()) + 1


def fit_lightgbm(x_tr, y_tr, cats, params, seed, iterations, eval_set=None):
    model = lgb.LGBMClassifier(n_estimators=iterations, max_depth=-1, objective="binary",
                               random_state=int(seed), verbosity=-1, n_jobs=THREADS, **params)
    if eval_set is None:
        model.fit(x_tr, y_tr, categorical_feature=cats)
        return model, iterations
    model.fit(x_tr, y_tr, categorical_feature=cats, eval_set=eval_set,
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    return model, max(int(model.best_iteration_ or iterations), 20)


def build_matrices(raw_tr: pd.DataFrame, raw_va: pd.DataFrame, mappings: dict):
    frame_tr, frame_va, no_id, cat_features, cats = prepare_features(raw_tr, raw_va, mappings)
    x_cat_tr = prepare_cat(frame_tr, cat_features, cats)
    x_cat_va = prepare_cat(frame_va, cat_features, cats)
    lgb_cats = [c for c in CAT_FEATURES if c in no_id]
    x_lgb_tr, x_lgb_va, lgb_cat_cols = encode_lgb(frame_tr, frame_va, no_id, lgb_cats)
    y_tr = frame_tr["is_valid"].to_numpy()
    y_va = frame_va["is_valid"].to_numpy() if "is_valid" in frame_va else None
    return {"x_cat_tr": x_cat_tr, "x_cat_va": x_cat_va, "cats": cats,
            "x_lgb_tr": x_lgb_tr, "x_lgb_va": x_lgb_va, "lgb_cats": lgb_cat_cols,
            "y_tr": y_tr, "y_va": y_va}


def component_probabilities(raw_tr, raw_va, mappings, seeds, log, model_dir=None):
    inner_cut = max(len(raw_tr) - BLOCK, int(0.6 * len(raw_tr)))
    inner_tr, inner_va = raw_tr.iloc[:inner_cut], raw_tr.iloc[inner_cut:]
    inner = build_matrices(inner_tr, inner_va, mappings)
    outer = build_matrices(raw_tr, raw_va, mappings)
    inner_times = pd.to_datetime(inner_tr["first_event_time"])
    outer_times = pd.to_datetime(raw_tr["first_event_time"])
    probabilities = {}
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in CAT_PARAMS.items():
        w_inner = None if cfg["half_life"] is None else recency_weight(inner_times, cfg["half_life"])
        w_outer = None if cfg["half_life"] is None else recency_weight(outer_times, cfg["half_life"])
        started = time.time()
        _, iterations = fit_catboost(inner["x_cat_tr"], inner["y_tr"], inner["cats"], w_inner,
                                     cfg["l2_leaf_reg"], cfg["depth"], seeds[0], MAX_ITERATIONS_CAT,
                                     eval_set=(inner["x_cat_va"], inner["y_va"]))
        log[f"{name}_iterations"] = iterations
        for seed in seeds:
            model, _ = fit_catboost(outer["x_cat_tr"], outer["y_tr"], outer["cats"], w_outer,
                                    cfg["l2_leaf_reg"], cfg["depth"], seed, iterations)
            probabilities[f"{name}_s{seed}"] = model.predict_proba(outer["x_cat_va"])[:, 1]
            if model_dir is not None:
                model.save_model(str(model_dir / f"{name}_s{seed}.cbm"))
        log[f"{name}_seconds"] = round(time.time() - started, 1)
        print(f"    {name}: iterations={iterations} ({log[f'{name}_seconds']}s)", flush=True)
    for name, params in LGB_PARAMS.items():
        started = time.time()
        _, iterations = fit_lightgbm(inner["x_lgb_tr"], inner["y_tr"], inner["lgb_cats"], params,
                                     seeds[0], MAX_ITERATIONS_LGB,
                                     eval_set=[(inner["x_lgb_va"], inner["y_va"])])
        log[f"{name}_iterations"] = iterations
        for seed in seeds:
            model, _ = fit_lightgbm(outer["x_lgb_tr"], outer["y_tr"], outer["lgb_cats"], params,
                                    seed, iterations)
            probabilities[f"{name}_s{seed}"] = model.predict_proba(outer["x_lgb_va"])[:, 1]
            if model_dir is not None:
                joblib.dump(model, model_dir / f"{name}_s{seed}.pkl", compress=3)
        log[f"{name}_seconds"] = round(time.time() - started, 1)
        print(f"    {name}: iterations={iterations} ({log[f'{name}_seconds']}s)", flush=True)
    return probabilities


def _prune_models_if_needed(model_dir: Path) -> None:
    total_bytes = sum(path.stat().st_size for path in model_dir.iterdir() if path.is_file())
    if total_bytes <= MODEL_SIZE_LIMIT_BYTES:
        return
    for path in model_dir.iterdir():
        if path.is_file() and "_s42." not in path.name:
            path.unlink()


def _mean_seed_predictions(probabilities):
    return {name: np.mean([probabilities[f"{name}_s{seed}"] for seed in (42, 2026, 777)], axis=0)
            for name in ("cat_all", "cat_recent", "lgb_neutral", "lgb_diverse")}


def main():
    """Train all final components and write submission.csv.

    Complete model sets are reproduced by rerunning this script; if compressed
    persistence exceeds 15 MiB, only seed 42 models are retained.
    """
    train = pd.read_csv(ROOT / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(ROOT / "test.csv")
    mappings = compute_agg_counts(train, test)
    log = {"rows_train": len(train), "rows_test": len(test), "seeds": [42, 2026, 777]}
    print("FINAL: full train -> test.csv", flush=True)
    probabilities = component_probabilities(train, test, mappings, log["seeds"], log, ROOT / "models")
    _prune_models_if_needed(ROOT / "models")
    means = _mean_seed_predictions(probabilities)
    np.savez_compressed(ROOT / "artifacts" / "probs_test_reproduced.npz",
                        claim_id=test["claim_id"].to_numpy(), **means)
    blend = sum(means[name] * BLEND_WEIGHTS[name] for name in BLEND_WEIGHTS)
    # RATE_MULTIPLIER was selected on temporal CV (see src/rate_curve.py);
    # TEST_POSITIVES is inferred from the organizers' all-ones sample_f1.
    k = round(RATE_MULTIPLIER * TEST_POSITIVES)
    order = np.argsort(-blend, kind="mergesort")
    labels = np.zeros(len(blend), dtype=int)
    labels[order[:k]] = 1
    pd.DataFrame({"claim_id": test["claim_id"], "is_valid": labels}).to_csv(ROOT / "submission.csv", index=False)
    (ROOT / "artifacts" / "solution_final_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    print(f"submission.csv written with positives={int(labels.sum())}, k={k}")


if __name__ == "__main__":
    main()
