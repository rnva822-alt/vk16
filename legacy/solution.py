"""
VK Competition Solution: Binary Classification for `is_valid` Prediction
========================================================================

Task: Predict whether a complaint is valid (is_valid=1) or rejected (is_valid=0).
Metric: F1 score for the positive class (is_valid=1).

Approach:
---------
1. Feature Engineering:
   - Temporal features from first_event_time and content_registered_time
     (useful: complaint timing patterns differ by hour, day, weekend)
   - Account age features (older accounts may behave differently)
   - Bot score interactions (bot-like behavior correlates with invalid complaints)
   - Country match features (mismatched countries may indicate fraud)
   - Ordinal encoding for age_bucket and friends_bucket
   - Aggregated count features (computed on combined train+test to avoid leakage;
     useful: repeat offenders and frequently-complained content patterns)
   - K-fold target encoding (prevents leakage — each row's TE computed from other folds)
   - Log-transformed skewed counters (stabilizes variance of count features)
   - Interaction features (claim_type × reason, sex match, age/friends diff)

2. Validation: 5-fold TimeSeriesSplit cross-validation with OOF predictions
   - Temporal data requires time-based validation (simulates production deployment)
   - OOF predictions pooled across folds for threshold/weight optimization;
     LAST-FOLD weights/threshold used for final submission (most test-representative)

3. Models: Ensemble of CatBoost + LightGBM + XGBoost
   - Gradient boosted trees are chosen because they excel at tabular data with
     mixed types (categorical + numerical), capture non-linear interactions
     automatically, and are robust to outliers and feature scale differences.
   - Three models provide ensemble diversity: CatBoost uses ordered boosting
     (reduces overfitting on small data), LightGBM uses leaf-wise growth
     (efficient on large data), XGBoost uses level-wise growth (stable).
     Their different splitting strategies produce complementary error patterns.

4. Threshold Optimization: Scan 0.05-0.95; optimize on both pooled OOF and last fold

5. Ensemble: Weighted probability averaging with grid-searched weights;
   also compares against rank averaging; final submission uses LAST-FOLD
   weights/threshold (most representative of test conditions)

6. Final Model: Retrain on full train data using best_iteration from last CV fold,
   average predictions, apply last-fold optimized threshold + rate calibration

Usage:
    cd /Users/artem/Documents/VK && python3 solution.py
    (Dependencies: pip install catboost lightgbm xgboost scipy scikit-learn pandas numpy)
"""

# ============================================================================
# Section 1: Imports & Constants
# ============================================================================

import warnings
import json
import time
import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target encoding smoothing parameter — higher alpha = more shrinkage toward global mean
TE_ALPHA = 10

# Columns for target encoding (high-cardinality categoricals where TE adds signal)
TE_COLUMNS = [
    "claim_type",
    "claim_reason_start",
    "claim_user_registered_phone_country_id",
    "platform",
]

# Ordinal encoding maps (preserve natural ordering of bucketed features)
AGE_MAP = {
    "0_13": 0, "14_17": 1, "18_24": 2, "25_34": 3,
    "35_44": 4, "45_54": 5, "55_64": 6, "65_plus": 7,
}

FRIENDS_MAP = {
    "0": 0, "1_5": 1, "6_20": 2, "21_50": 3, "51_100": 4,
    "101_250": 5, "251_500": 6, "501_1000": 7, "1001_plus": 8,
}

# Columns to drop after feature engineering (IDs, raw datetimes, original bucket strings)
DROP_COLS = [
    "claim_id", "id_content", "first_event_time", "content_registered_time",
    "ip_country_id", "id_content_owner", "age_bucket", "claim_user_age_bucket",
    "friends_bucket", "claim_user_friends_bucket",
]

# Categorical columns (for CatBoost cat_features and LGB/XGB category dtype)
CAT_FEATURES = [
    "os", "platform", "sex", "claim_user_sex",
    "claim_type", "claim_reason_start",
    "registered_phone_country_id", "mobile_phone_country_id", "profile_country_id",
    "claim_user_registered_phone_country_id", "claim_user_profile_country_id",
    "claim_type_reason",
]

# String categorical columns needing label encoding for LGB/XGB
STRING_CAT_COLS = ["os", "claim_type_reason"]

# scale_pos_weight rationale: positive rate ≈ 14.35%, so neg/pos ≈ 5.96 ≈ 6.
# This balances gradient updates for the minority (positive) class.
SCALE_POS_WEIGHT = 6


# ============================================================================
# Section 2: Data Loading
# ============================================================================

def load_data():
    """Load train and test data, sort train by first_event_time."""
    print("=" * 70)
    print("Loading data...")
    print("=" * 70)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"Target distribution:\n{train['is_valid'].value_counts(normalize=True)}")

    # Sort train by first_event_time for time-based cross-validation
    train = train.sort_values("first_event_time").reset_index(drop=True)
    print(f"\nTrain time range: {train['first_event_time'].min()} to {train['first_event_time'].max()}")
    print(f"Test time range:  {test['first_event_time'].min()} to {test['first_event_time'].max()}")

    return train, test


# ============================================================================
# Section 3: EDA
# ============================================================================

def eda(train, test):
    """Exploratory Data Analysis — print key statistics and distribution checks."""
    print("\n" + "=" * 70)
    print("EDA: Exploratory Data Analysis")
    print("=" * 70)

    # --- Missing values ---
    print("\n--- Missing Values ---")
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()
    print(f"Train: {train_missing.sum()} total missing ({(train_missing > 0).sum()} columns with missing)")
    print(f"Test:  {test_missing.sum()} total missing ({(test_missing > 0).sum()} columns with missing)")
    if train_missing.sum() > 0:
        print("Columns with missing in train:")
        print(train_missing[train_missing > 0].to_string())

    # --- Numeric feature summary ---
    print("\n--- Numeric Feature Summary (train describe) ---")
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    print(train[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())

    # --- Target rate by key categorical features ---
    print("\n--- Target Rate by Key Categorical Features ---")
    for col in ["claim_type", "claim_reason_start", "platform", "age_bucket", "os"]:
        if col in train.columns:
            rate = train.groupby(col)["is_valid"].agg(["mean", "count"]).sort_values("mean", ascending=False)
            print(f"\n  {col} (sorted by target rate):")
            print(rate.to_string())

    # --- Bot score distribution by target ---
    print("\n--- Bot Score Distribution by Target ---")
    for col in ["user_bot_prediction_score", "claim_user_bot_prediction_score"]:
        if col in train.columns:
            print(f"\n  {col}:")
            print(train.groupby("is_valid")[col].describe().round(4).to_string())

    # --- Class imbalance ---
    print("\n--- Class Imbalance Summary ---")
    counts = train["is_valid"].value_counts()
    print(f"  is_valid=0 (rejected): {counts[0]:,} ({counts[0] / len(train):.2%})")
    print(f"  is_valid=1 (valid):    {counts[1]:,} ({counts[1] / len(train):.2%})")
    print(f"  Imbalance ratio (neg/pos): {counts[0] / counts[1]:.2f}:1")
    print(f"  scale_pos_weight = {SCALE_POS_WEIGHT} (≈ 1 / {counts[1] / len(train):.4f})")

    # --- Train vs test feature distribution comparison ---
    print("\n--- Train vs Test Feature Distribution (numeric mean comparison) ---")
    common_numeric = [c for c in numeric_cols if c in test.columns and c != "is_valid"]
    comparison = pd.DataFrame({
        "train_mean": train[common_numeric].mean(),
        "test_mean": test[common_numeric].mean(),
    })
    comparison["abs_diff"] = (comparison["train_mean"] - comparison["test_mean"]).abs()
    comparison["diff_pct"] = (comparison["abs_diff"] / comparison["train_mean"].abs() * 100).round(2)
    comparison = comparison.sort_values("diff_pct", ascending=False)
    print(comparison.round(4).to_string())

    # --- Temporal split ---
    print("\n--- Temporal Split ---")
    print(f"  Train time range: {train['first_event_time'].min()} to {train['first_event_time'].max()}")
    print(f"  Test time range:  {test['first_event_time'].min()} to {test['first_event_time'].max()}")
    train_end = pd.to_datetime(train["first_event_time"]).max()
    test_start = pd.to_datetime(test["first_event_time"]).min()
    if test_start >= train_end:
        print(f"  -> Test starts AFTER train ends (strict temporal split, no overlap)")
    else:
        print(f"  -> WARNING: Train and test time ranges overlap!")


# ============================================================================
# Section 4: Aggregated Count Features (computed on combined train+test)
# ============================================================================

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
    print(f"  Frequency encoding: 5 features (id_content_owner, claim_type, claim_reason, os, registered_phone_country)")
    print(f"  Statistical aggregations: 5 features (mean/std bot scores, mean likes by claim_type/reason)")

    return agg_mappings


# ============================================================================
# Section 5: Feature Engineering
# ============================================================================

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


# ============================================================================
# Section 6: Target Encoding (K-Fold, leakage-free)
# ============================================================================

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


# ============================================================================
# Section 7: Label Encoding & Categorical Preparation
# ============================================================================

def compute_label_mappings(train_fe, test_fe):
    """Compute label encoding mappings for string categorical columns.

    Mappings are computed on combined train+test to handle all categories.
    """
    label_mappings = {}
    for col in STRING_CAT_COLS:
        combined_vals = pd.concat([train_fe[col], test_fe[col]], ignore_index=True).unique()
        label_mappings[col] = {v: i for i, v in enumerate(combined_vals)}
    return label_mappings


def prepare_for_lgb_xgb(df, label_mappings, cast_category=True):
    """Prepare dataframe for LightGBM/XGBoost: label-encode strings, optionally cast cats.

    LightGBM natively handles 'category' dtype columns, which lets it learn
    optimal splits on categorical values rather than treating them as
    continuous integers.

    XGBoost 2.1.4 has issues with enable_categorical=True combined with
    high-cardinality categoricals and scale_pos_weight, so for XGBoost we
    pass cast_category=False to keep all features numeric.
    """
    df = df.copy()
    # Label-encode string categorical columns to integers
    for col in STRING_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].map(label_mappings[col]).fillna(-1).astype(int)
    if cast_category:
        # Cast all categorical columns to category dtype for native LightGBM handling
        for col in CAT_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
    return df


def prepare_for_catboost(df):
    """Convert categorical columns to string for CatBoost.

    CatBoost handles unknown categories natively via its encoding scheme.
    """
    df = df.copy()
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# ============================================================================
# Section 8: Threshold Optimization
# ============================================================================

def optimize_threshold(y_true, y_proba, label=""):
    """Scan thresholds from 0.05 to 0.95 to maximize F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"  {label:25s} | Best F1: {best_f1:.4f} | Threshold: {best_threshold:.2f}")
    return best_threshold, best_f1


# ============================================================================
# Section 8b: Ensemble Optimization (Weighted Grid Search)
# ============================================================================

def optimize_ensemble(val_preds_dict, val_y):
    """Grid search over ensemble weights and thresholds to maximize F1.

    Tries all weight combinations (w_cat, w_lgb, w_xgb) where each weight
    ranges from 0.0 to 1.0 in steps of 0.1 and they sum to 1.0.
    """
    model_names = list(val_preds_dict.keys())
    n_models = len(model_names)
    pred_arrays = [np.asarray(val_preds_dict[name], dtype=np.float64) for name in model_names]
    val_y = np.asarray(val_y)

    thresholds = np.arange(0.05, 0.95, 0.01)
    steps = 10

    weight_grid = []
    for w0 in range(steps + 1):
        for w1 in range(steps + 1 - w0):
            w2 = steps - w0 - w1
            weight_grid.append((w0 / steps, w1 / steps, w2 / steps))

    results = []
    for weights in weight_grid:
        weighted_proba = np.zeros_like(pred_arrays[0])
        for w, p in zip(weights, pred_arrays):
            weighted_proba += w * p

        best_thr = 0.5
        best_f1_w = 0.0
        for thr in thresholds:
            y_pred = (weighted_proba >= thr).astype(int)
            f1 = f1_score(val_y, y_pred, zero_division=0)
            if f1 > best_f1_w:
                best_f1_w = f1
                best_thr = float(thr)

        weights_dict = {model_names[i]: weights[i] for i in range(n_models)}
        results.append((weights_dict, best_thr, best_f1_w))

    results.sort(key=lambda x: x[2], reverse=True)
    best_weights, best_threshold, best_f1 = results[0]
    top_configs = results[:5]

    return best_weights, best_threshold, best_f1, top_configs


def rank_average(preds_dict):
    """Convert each model's probabilities to ranks, average, normalize to [0, 1].

    Rank averaging is robust to miscalibrated probabilities — it only relies
    on the relative ordering each model produces.
    """
    model_names = list(preds_dict.keys())
    n = len(preds_dict[model_names[0]])
    rank_sum = np.zeros(n, dtype=np.float64)
    for name in model_names:
        rank_sum += rankdata(preds_dict[name])
    avg_rank = rank_sum / len(model_names)
    denom = avg_rank.max() - avg_rank.min()
    if denom < 1e-12:
        return avg_rank
    return (avg_rank - avg_rank.min()) / denom


# ============================================================================
# Section 9: Model Training (on validation fold)
# ============================================================================

def train_catboost(X_train, y_train, X_val, y_val, cat_features, quiet=False):
    """Train CatBoost classifier with early stopping."""
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=False if quiet else 200,
        early_stopping_rounds=150,
        cat_features=cat_features,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, categorical_feature=None, quiet=False):
    """Train LightGBM classifier with early stopping."""
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        objective="binary",
        metric="auc",
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=not quiet)]
    if not quiet:
        callbacks.append(lgb.log_evaluation(200))
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_feature,
        callbacks=callbacks,
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, quiet=False):
    """Train XGBoost classifier with early stopping.

    Note: enable_categorical is NOT used. XGBoost 2.1.4 has issues with
    high-cardinality category dtype + scale_pos_weight, compressing all
    probabilities below 0.5. All features are treated as numeric.
    """
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        early_stopping_rounds=150,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False if quiet else 200)
    return model


def get_best_iterations(cb_model, lgb_model, xgb_model, quiet=False):
    """Extract best iteration from each model for final training.

    CatBoost and XGBoost use 0-indexed best_iteration, so we add 1 to get
    the number of trees to train. LightGBM's best_iteration_ is already
    1-indexed (equals the number of trees retained), so NO +1 is needed.
    """
    # CatBoost: get_best_iteration() is 0-indexed
    cb_best = cb_model.get_best_iteration()
    if cb_best is None:
        cb_best = 1999
    cb_best = int(cb_best) + 1

    # LightGBM: best_iteration_ is 1-indexed (number of trees used) — NO +1
    lgb_best = lgb_model.best_iteration_
    if lgb_best is None:
        lgb_best = 2000
    lgb_best = int(lgb_best)

    # XGBoost: best_iteration is 0-indexed
    xgb_best = xgb_model.best_iteration
    if xgb_best is None:
        xgb_best = 1999
    xgb_best = int(xgb_best) + 1

    if not quiet:
        print(f"\n  Best iterations — CatBoost: {cb_best}, LightGBM: {lgb_best}, XGBoost: {xgb_best}")
    return cb_best, lgb_best, xgb_best


# ============================================================================
# Section 9b: Hyperparameter Tuning (on last CV fold)
# ============================================================================

def tune_catboost(X_train, y_train, X_val, y_val, cat_features):
    """Grid search CatBoost hyperparameters on the last CV fold.

    Tries 8 combinations of depth, learning_rate, and l2_leaf_reg.
    Each config is trained with 500 iterations (fast) and early stopping.
    Returns the best params and their F1 score.
    """
    param_grid = [
        {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 3},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
        {"depth": 7, "learning_rate": 0.02, "l2_leaf_reg": 7},
        {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 3},
        {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5},
        {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 3},
        {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5},
        {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 7},
    ]

    best_f1 = 0.0
    best_params = param_grid[3]  # default (close to current)

    for i, params in enumerate(param_grid):
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="Logloss",
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED,
            verbose=False,
            early_stopping_rounds=100,
            cat_features=cat_features,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        pred = model.predict_proba(X_val)[:, 1]

        # Optimize threshold for this config
        best_thr = 0.5
        best_thr_f1 = 0.0
        for thr in np.arange(0.30, 0.70, 0.01):
            f1 = f1_score(y_val, (pred >= thr).astype(int), zero_division=0)
            if f1 > best_thr_f1:
                best_thr_f1 = f1
                best_thr = float(thr)

        print(f"    CB config {i+1}/8: depth={params['depth']}, lr={params['learning_rate']}, "
              f"l2={params['l2_leaf_reg']} -> F1={best_thr_f1:.4f} @ thr={best_thr:.2f}")

        if best_thr_f1 > best_f1:
            best_f1 = best_thr_f1
            best_params = params

    print(f"  >>> Best CatBoost: depth={best_params['depth']}, lr={best_params['learning_rate']}, "
          f"l2={best_params['l2_leaf_reg']} -> F1={best_f1:.4f}")
    return best_params, best_f1


def tune_lightgbm(X_train, y_train, X_val, y_val, categorical_feature=None):
    """Grid search LightGBM hyperparameters on the last CV fold.

    Tries 8 combinations of num_leaves, learning_rate, and max_depth.
    Each config is trained with 500 iterations (fast) and early stopping.
    Returns the best params and their F1 score.
    """
    param_grid = [
        {"num_leaves": 31, "learning_rate": 0.02, "max_depth": 6},
        {"num_leaves": 31, "learning_rate": 0.05, "max_depth": 7},
        {"num_leaves": 63, "learning_rate": 0.02, "max_depth": 6},
        {"num_leaves": 63, "learning_rate": 0.03, "max_depth": 7},
        {"num_leaves": 63, "learning_rate": 0.05, "max_depth": 8},
        {"num_leaves": 127, "learning_rate": 0.02, "max_depth": 7},
        {"num_leaves": 127, "learning_rate": 0.03, "max_depth": 8},
        {"num_leaves": 127, "learning_rate": 0.05, "max_depth": 6},
    ]

    best_f1 = 0.0
    best_params = param_grid[3]  # default (close to current)

    for i, params in enumerate(param_grid):
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=SCALE_POS_WEIGHT,
            objective="binary",
            metric="auc",
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=categorical_feature,
            callbacks=callbacks,
        )
        pred = model.predict_proba(X_val)[:, 1]

        # Optimize threshold for this config
        best_thr = 0.5
        best_thr_f1 = 0.0
        for thr in np.arange(0.30, 0.70, 0.01):
            f1 = f1_score(y_val, (pred >= thr).astype(int), zero_division=0)
            if f1 > best_thr_f1:
                best_thr_f1 = f1
                best_thr = float(thr)

        print(f"    LGB config {i+1}/8: leaves={params['num_leaves']}, lr={params['learning_rate']}, "
              f"depth={params['max_depth']} -> F1={best_thr_f1:.4f} @ thr={best_thr:.2f}")

        if best_thr_f1 > best_f1:
            best_f1 = best_thr_f1
            best_params = params

    print(f"  >>> Best LightGBM: leaves={best_params['num_leaves']}, lr={best_params['learning_rate']}, "
          f"depth={best_params['max_depth']} -> F1={best_f1:.4f}")
    return best_params, best_f1


def tune_scale_pos_weight(X_tr_cb, y_tr, X_va_cb, y_va, cb_cat_features, cb_params,
                           X_tr_lgb, X_va_lgb, lgb_cat_features, lgb_params,
                           X_tr_xgb, X_va_xgb):
    """Try different scale_pos_weight values for all models on the last fold.

    For each spw in [3, 4, 5, 6, 7], trains all 3 models, computes ensemble F1
    (equal-weight average), and returns the best spw and its F1.
    CatBoost uses class_weights instead of auto_class_weights when spw is specified.
    """
    spw_values = [3, 4, 5, 6, 7]
    best_spw = SCALE_POS_WEIGHT
    best_f1 = 0.0

    for spw in spw_values:
        # CatBoost with class_weights
        cb_model = CatBoostClassifier(
            iterations=500,
            learning_rate=cb_params["learning_rate"],
            depth=cb_params["depth"],
            l2_leaf_reg=cb_params["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="Logloss",
            class_weights={0: 1, 1: spw},
            random_seed=RANDOM_SEED,
            verbose=False,
            early_stopping_rounds=100,
            cat_features=cb_cat_features,
        )
        cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True)
        cb_pred = cb_model.predict_proba(X_va_cb)[:, 1]

        # LightGBM with scale_pos_weight
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=lgb_params["learning_rate"],
            max_depth=lgb_params["max_depth"],
            num_leaves=lgb_params["num_leaves"],
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            objective="binary",
            metric="auc",
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_va)],
            categorical_feature=lgb_cat_features,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_pred = lgb_model.predict_proba(X_va_lgb)[:, 1]

        # XGBoost with scale_pos_weight
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            early_stopping_rounds=100,
        )
        xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        xgb_pred = xgb_model.predict_proba(X_va_xgb)[:, 1]

        # Ensemble (equal-weight average) and threshold optimization
        ensemble_pred = (cb_pred + lgb_pred + xgb_pred) / 3.0
        best_thr = 0.5
        best_thr_f1 = 0.0
        for thr in np.arange(0.30, 0.70, 0.01):
            f1 = f1_score(y_va, (ensemble_pred >= thr).astype(int), zero_division=0)
            if f1 > best_thr_f1:
                best_thr_f1 = f1
                best_thr = float(thr)

        print(f"    spw={spw} -> ensemble F1={best_thr_f1:.4f} @ thr={best_thr:.2f}")

        if best_thr_f1 > best_f1:
            best_f1 = best_thr_f1
            best_spw = spw

    print(f"  >>> Best scale_pos_weight: {best_spw} -> F1={best_f1:.4f}")
    return best_spw, best_f1


# ============================================================================
# Section 9c: Stacking (Meta-Model on OOF Predictions)
# ============================================================================

def train_stacking_meta_model(oof_train, oof_y, oof_val, val_y, test_preds):
    """Train meta-model on OOF predictions and evaluate on last fold.

    Parameters
    ----------
    oof_train : np.ndarray of shape (n_train_meta, 3)
        OOF predictions [cb, lgb, xgb] for meta-model training (folds 1-4).
    oof_y : np.ndarray
        Labels for meta-model training.
    oof_val : np.ndarray of shape (n_val, 3)
        OOF predictions [cb, lgb, xgb] for last fold (meta-model validation).
    val_y : np.ndarray
        Labels for last fold.
    test_preds : np.ndarray of shape (n_test, 3)
        Final models' test predictions [cb, lgb, xgb].

    Returns
    -------
    dict with stacking results
    """
    results = {}

    # --- LogisticRegression meta-model ---
    lr_meta = LogisticRegression(C=1.0, random_state=RANDOM_SEED, max_iter=1000)
    lr_meta.fit(oof_train, oof_y)
    lr_val_pred = lr_meta.predict_proba(oof_val)[:, 1]
    lr_test_pred = lr_meta.predict_proba(test_preds)[:, 1]

    lr_best_thr, lr_best_f1 = optimize_threshold(val_y, lr_val_pred, "Stacking-LR last-fold")
    results["lr"] = {"val_f1": lr_best_f1, "val_thr": lr_best_thr, "test_pred": lr_test_pred}

    # --- LightGBM meta-model ---
    lgb_meta = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    lgb_meta.fit(oof_train, oof_y)
    lgb_val_pred = lgb_meta.predict_proba(oof_val)[:, 1]
    lgb_test_pred = lgb_meta.predict_proba(test_preds)[:, 1]

    lgb_best_thr, lgb_best_f1 = optimize_threshold(val_y, lgb_val_pred, "Stacking-LGB last-fold")
    results["lgb"] = {"val_f1": lgb_best_f1, "val_thr": lgb_best_thr, "test_pred": lgb_test_pred}

    return results


# ============================================================================
# Section 10: Final Model Training (on full train data)
# ============================================================================

def train_final_catboost(X, y, cat_features, n_estimators, depth=7, learning_rate=0.03,
                          l2_leaf_reg=3, scale_pos_weight=None):
    """Train CatBoost on full data with fixed iteration count and tuned params."""
    if scale_pos_weight is not None:
        kwargs = {"class_weights": {0: 1, 1: scale_pos_weight}}
    else:
        kwargs = {"auto_class_weights": "Balanced"}

    model = CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=200,
        cat_features=cat_features,
        **kwargs,
    )
    model.fit(X, y)
    return model


def train_final_lightgbm(X, y, n_estimators, categorical_feature=None,
                           num_leaves=63, learning_rate=0.03, max_depth=7,
                           scale_pos_weight=SCALE_POS_WEIGHT):
    """Train LightGBM on full data with fixed iteration count and tuned params."""
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        metric="auc",
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X, y, categorical_feature=categorical_feature)
    return model


def train_final_xgboost(X, y, n_estimators, scale_pos_weight=SCALE_POS_WEIGHT):
    """Train XGBoost on full data with fixed iteration count (numeric features only)."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(X, y, verbose=200)
    return model


# ============================================================================
# Section 10b: Feature Importance
# ============================================================================

def print_feature_importance(cb_model, lgb_model, xgb_model, feature_cols):
    """Extract and print feature importances from all three models."""
    print("\n" + "=" * 70)
    print("Feature Importance Analysis (from last CV fold models)")
    print("=" * 70)

    # CatBoost
    cb_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": cb_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print("\nCatBoost Top 20 Features:")
    print(cb_imp.head(20).to_string(index=False))

    # LightGBM
    lgb_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nLightGBM Top 20 Features:")
    print(lgb_imp.head(20).to_string(index=False))

    # XGBoost
    xgb_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nXGBoost Top 20 Features:")
    print(xgb_imp.head(20).to_string(index=False))


# ============================================================================
# Section 10c: Adversarial Validation
# ============================================================================

def adversarial_validation(val_df, test_df, label_mappings, feature_cols):
    """Check if validation and test distributions are similar.

    Trains a LightGBM to distinguish val (label=0) from test (label=1).
    High AUC indicates distribution shift — validation F1 may not transfer.
    """
    print("\n" + "=" * 70)
    print("Adversarial Validation (last CV fold val vs test)")
    print("=" * 70)

    # Prepare features for LightGBM
    X_val = prepare_for_lgb_xgb(val_df[feature_cols], label_mappings)
    X_test = prepare_for_lgb_xgb(test_df[feature_cols], label_mappings)

    # Convert category columns to numeric codes for consistent concatenation
    for df_tmp in [X_val, X_test]:
        for col in df_tmp.columns:
            if str(df_tmp[col].dtype) == "category":
                df_tmp[col] = df_tmp[col].cat.codes

    X_combined = pd.concat([X_val, X_test], axis=0, ignore_index=True)
    y_combined = np.concatenate([np.zeros(len(X_val)), np.ones(len(X_test))])

    adv_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    adv_model.fit(X_combined, y_combined)

    adv_pred = adv_model.predict_proba(X_combined)[:, 1]
    auc = roc_auc_score(y_combined, adv_pred)

    print(f"\n  Adversarial Validation AUC: {auc:.4f}")

    imp = pd.DataFrame({
        "feature": X_combined.columns,
        "importance": adv_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("  Top 10 features distinguishing val from test:")
    print(imp.head(10).to_string(index=False))

    if auc > 0.7:
        print("\n  WARNING: AUC > 0.7 — significant distribution shift detected!")
        print("  Validation F1 may not transfer to test.")
    else:
        print("\n  AUC <= 0.7 — distributions are reasonably similar.")

    return auc


# ============================================================================
# Section 11: Main Pipeline
# ============================================================================

def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    train, test = load_data()

    # ------------------------------------------------------------------
    # Step 2: EDA
    # ------------------------------------------------------------------
    eda(train, test)

    # ------------------------------------------------------------------
    # Step 3: Compute aggregated count features (on combined train+test)
    # ------------------------------------------------------------------
    agg_mappings = compute_agg_counts(train, test)

    # ------------------------------------------------------------------
    # Step 4: Feature engineering
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Feature engineering...")
    print("=" * 70)

    train_fe = engineer_features(train, agg_mappings)
    test_fe = engineer_features(test, agg_mappings)

    print(f"Train features shape: {train_fe.shape}")
    print(f"Test features shape:  {test_fe.shape}")

    # ------------------------------------------------------------------
    # Step 5: Compute label encoding mappings (on combined data)
    # ------------------------------------------------------------------
    label_mappings = compute_label_mappings(train_fe, test_fe)

    # ------------------------------------------------------------------
    # Step 6: Full-train target encoding (K-fold for train, full-train for test)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Target encoding (K-fold for train, full-train mapping for test)...")
    print("=" * 70)

    # K-fold TE on full train (leakage-free for final model training)
    train_full_te = kfold_target_encoding(train_fe, "is_valid", TE_ALPHA)
    # TE from full train applied to test (correct: test labels not used)
    test_full_te = apply_te_to_new_data(train_fe, test_fe, "is_valid", TE_ALPHA)
    print("  K-fold target encoding complete (5-fold, leakage-free).")

    # Define feature columns (all columns except is_valid)
    feature_cols = [c for c in train_full_te.columns if c != "is_valid"]
    n_old_features = 63  # previous version had 63 features
    print(f"  Total features (with target encoding): {len(feature_cols)} (was {n_old_features} before new features)")

    # Categorical features present in feature_cols
    cb_cat_features = [c for c in CAT_FEATURES if c in feature_cols]
    lgb_cat_features = [c for c in CAT_FEATURES if c in feature_cols]
    print(f"  Categorical features: {len(cb_cat_features)} columns")

    # ------------------------------------------------------------------
    # Step 7: Prepare final model matrices
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Preparing final model feature matrices...")
    print("=" * 70)

    X_train_full_cb = prepare_for_catboost(train_full_te[feature_cols])
    X_test_cb = prepare_for_catboost(test_full_te[feature_cols])
    X_train_full_num = prepare_for_lgb_xgb(train_full_te[feature_cols], label_mappings)
    X_test_num = prepare_for_lgb_xgb(test_full_te[feature_cols], label_mappings)
    # XGBoost: keep all features numeric (no category dtype) — avoids
    # enable_categorical issues in XGBoost 2.1.4 with high-cardinality cats
    X_train_full_xgb = prepare_for_lgb_xgb(train_full_te[feature_cols], label_mappings, cast_category=False)
    X_test_xgb = prepare_for_lgb_xgb(test_full_te[feature_cols], label_mappings, cast_category=False)
    y_train_full = train_full_te["is_valid"].values

    print(f"  Full train (CatBoost): {X_train_full_cb.shape}")
    print(f"  Full train (LightGBM): {X_train_full_num.shape}")
    print(f"  Full train (XGBoost):  {X_train_full_xgb.shape}")
    print(f"  Test (CatBoost):       {X_test_cb.shape}")
    print(f"  Test (LightGBM):       {X_test_num.shape}")
    print(f"  Test (XGBoost):        {X_test_xgb.shape}")

    # ------------------------------------------------------------------
    # Step 8: 5-fold TimeSeriesSplit CV with OOF predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5-Fold TimeSeriesSplit Cross-Validation with OOF Predictions")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    # OOF storage
    oof_cb = np.zeros(len(train_fe))
    oof_lgb = np.zeros(len(train_fe))
    oof_xgb = np.zeros(len(train_fe))
    oof_mask = np.zeros(len(train_fe), dtype=bool)

    # Track best iterations across folds
    all_cb_best = []
    all_lgb_best = []
    all_xgb_best = []

    # Store last fold's models and data for feature importance & adversarial validation
    last_cb_model = None
    last_lgb_model = None
    last_xgb_model = None
    last_val_fold_te = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_fe), 1):
        print(f"\n--- Fold {fold}/5 ---")
        print(f"  Train: {len(train_idx)} rows | Val: {len(val_idx)} rows")

        train_fold = train_fe.iloc[train_idx].copy()
        val_fold = train_fe.iloc[val_idx].copy()

        # K-fold TE on train_fold (leakage-free), apply TE from train_fold to val_fold
        train_fold_te = kfold_target_encoding(train_fold, "is_valid", TE_ALPHA)
        val_fold_te = apply_te_to_new_data(train_fold, val_fold, "is_valid", TE_ALPHA)

        # Prepare matrices
        X_tr_cb = prepare_for_catboost(train_fold_te[feature_cols])
        X_va_cb = prepare_for_catboost(val_fold_te[feature_cols])
        X_tr_num = prepare_for_lgb_xgb(train_fold_te[feature_cols], label_mappings)
        X_va_num = prepare_for_lgb_xgb(val_fold_te[feature_cols], label_mappings)
        # XGBoost: numeric only (no category dtype)
        X_tr_xgb = prepare_for_lgb_xgb(train_fold_te[feature_cols], label_mappings, cast_category=False)
        X_va_xgb = prepare_for_lgb_xgb(val_fold_te[feature_cols], label_mappings, cast_category=False)

        y_tr = train_fold_te["is_valid"].values
        y_va = val_fold_te["is_valid"].values

        # Train models (quiet during CV to reduce output)
        cb_model = train_catboost(X_tr_cb, y_tr, X_va_cb, y_val=y_va, cat_features=cb_cat_features, quiet=True)
        lgb_model = train_lightgbm(X_tr_num, y_tr, X_va_num, y_val=y_va, categorical_feature=lgb_cat_features, quiet=True)
        xgb_model = train_xgboost(X_tr_xgb, y_tr, X_va_xgb, y_val=y_va, quiet=True)

        # Predict on validation
        cb_pred = cb_model.predict_proba(X_va_cb)[:, 1]
        lgb_pred = lgb_model.predict_proba(X_va_num)[:, 1]
        xgb_pred = xgb_model.predict_proba(X_va_xgb)[:, 1]

        oof_cb[val_idx] = cb_pred
        oof_lgb[val_idx] = lgb_pred
        oof_xgb[val_idx] = xgb_pred
        oof_mask[val_idx] = True

        # Per-fold F1 at default threshold 0.5
        cb_f1_fold = f1_score(y_va, (cb_pred >= 0.5).astype(int), zero_division=0)
        lgb_f1_fold = f1_score(y_va, (lgb_pred >= 0.5).astype(int), zero_division=0)
        xgb_f1_fold = f1_score(y_va, (xgb_pred >= 0.5).astype(int), zero_division=0)

        # Best iterations
        cb_best, lgb_best, xgb_best = get_best_iterations(cb_model, lgb_model, xgb_model, quiet=True)
        all_cb_best.append(cb_best)
        all_lgb_best.append(lgb_best)
        all_xgb_best.append(xgb_best)

        print(f"  F1@0.5 — CatBoost: {cb_f1_fold:.4f}, LightGBM: {lgb_f1_fold:.4f}, XGBoost: {xgb_f1_fold:.4f}")
        print(f"  Best iters — CB: {cb_best}, LGB: {lgb_best}, XGB: {xgb_best}")

        # Save last fold's models and data (for feature importance, adversarial validation, and tuning)
        if fold == 5:
            last_cb_model = cb_model
            last_lgb_model = lgb_model
            last_xgb_model = xgb_model
            last_val_fold_te = val_fold_te
            last_y_va = y_va
            last_cb_pred = cb_pred
            last_lgb_pred = lgb_pred
            last_xgb_pred = xgb_pred
            # Save training data for hyperparameter tuning
            last_X_tr_cb = X_tr_cb
            last_X_va_cb = X_va_cb
            last_X_tr_num = X_tr_num
            last_X_va_num = X_va_num
            last_X_tr_xgb = X_tr_xgb
            last_X_va_xgb = X_va_xgb
            last_y_tr = y_tr
            last_val_idx = val_idx

    # ------------------------------------------------------------------
    # Step 9: Pooled OOF predictions — threshold & ensemble optimization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pooled OOF Predictions — Threshold & Ensemble Optimization")
    print("=" * 70)

    pooled_y = train_fe["is_valid"].values[oof_mask]
    pooled_cb = oof_cb[oof_mask]
    pooled_lgb = oof_lgb[oof_mask]
    pooled_xgb = oof_xgb[oof_mask]

    print(f"  Total OOF predictions: {len(pooled_y)}")

    # Individual model OOF threshold optimization
    print("\n  Individual model OOF threshold optimization (0.05 - 0.95, step 0.01):")
    cb_thr, cb_oof_f1 = optimize_threshold(pooled_y, pooled_cb, "CatBoost OOF")
    lgb_thr, lgb_oof_f1 = optimize_threshold(pooled_y, pooled_lgb, "LightGBM OOF")
    xgb_thr, xgb_oof_f1 = optimize_threshold(pooled_y, pooled_xgb, "XGBoost OOF")

    print(f"\n  {'Model':27s} | {'Best F1':8s} | {'Threshold':10s}")
    print(f"  {'-' * 27}-+-{'-' * 8}-+-{'-' * 10}")
    print(f"  {'CatBoost':27s} | {cb_oof_f1:.4f}   | {cb_thr:.2f}")
    print(f"  {'LightGBM':27s} | {lgb_oof_f1:.4f}   | {lgb_thr:.2f}")
    print(f"  {'XGBoost':27s} | {xgb_oof_f1:.4f}   | {xgb_thr:.2f}")

    # Ensemble optimization (weighted grid search)
    print("\n  Ensemble optimization: weighted grid search over (w_cat, w_lgb, w_xgb)...")
    print("  Weights range 0.0-1.0 step 0.1, sum to 1.0; threshold 0.05-0.95 step 0.01")

    oof_preds_dict = {"catboost": pooled_cb, "lightgbm": pooled_lgb, "xgboost": pooled_xgb}
    best_weights, best_thr_prob, best_f1_prob, top_configs = optimize_ensemble(oof_preds_dict, pooled_y)

    print(f"\n  Top 5 weight combinations (probability averaging):")
    print(f"  {'Rank':4s} | {'CatBoost':8s} {'LightGBM':8s} {'XGBoost':8s} | {'F1':6s} | {'Threshold':9s}")
    print(f"  {'-' * 4}-+-{'-' * 8}-{'-' * 8}-{'-' * 8}-+-{'-' * 6}-+-{'-' * 9}")
    for rank, (w, thr, f1) in enumerate(top_configs, 1):
        print(f"  {rank:4d} | {w['catboost']:8.1f} {w['lightgbm']:8.1f} {w['xgboost']:8.1f} | {f1:.4f} | {thr:.2f}")

    print(f"\n  >>> Best probability-averaging config: "
          f"w_cat={best_weights['catboost']:.1f}, "
          f"w_lgb={best_weights['lightgbm']:.1f}, "
          f"w_xgb={best_weights['xgboost']:.1f} | "
          f"F1={best_f1_prob:.4f} | threshold={best_thr_prob:.2f}")

    # Rank averaging alternative
    print("\n  Rank averaging alternative:")
    rank_oof_proba = rank_average(oof_preds_dict)
    rank_thr, rank_f1 = optimize_threshold(pooled_y, rank_oof_proba, "Rank Average OOF")

    print(f"\n  Rank-averaged OOF F1: {rank_f1:.4f} (threshold={rank_thr:.2f})")

    # Compare probability averaging vs rank averaging
    use_rank = rank_f1 > best_f1_prob
    if use_rank:
        print(f"\n  >>> Rank averaging ({rank_f1:.4f}) BEATS probability averaging ({best_f1_prob:.4f})")
        final_method = "rank_average"
        final_ens_f1 = rank_f1
        final_thr = rank_thr
    else:
        print(f"\n  >>> Probability averaging ({best_f1_prob:.4f}) BEATS rank averaging ({rank_f1:.4f})")
        final_method = "probability_average"
        final_ens_f1 = best_f1_prob
        final_thr = best_thr_prob

    print(f"  >>> Selected ensemble method: {final_method}")
    print(f"  >>> Selected ensemble OOF F1: {final_ens_f1:.4f} (threshold={final_thr:.2f})")

    # ------------------------------------------------------------------
    # Step 9b: Last-fold comparison (most representative of test conditions)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Last-Fold Optimization Comparison (most representative of test)")
    print("=" * 70)
    print("  (The last fold has the most training data, closest to the final model.)\n")

    print("  Individual model last-fold threshold optimization:")
    lf_cb_thr, lf_cb_f1 = optimize_threshold(last_y_va, last_cb_pred, "CatBoost last-fold")
    lf_lgb_thr, lf_lgb_f1 = optimize_threshold(last_y_va, last_lgb_pred, "LightGBM last-fold")
    lf_xgb_thr, lf_xgb_f1 = optimize_threshold(last_y_va, last_xgb_pred, "XGBoost last-fold")

    print(f"\n  {'Model':27s} | {'Pooled F1':10s} | {'Last-fold F1':12s} | {'Pooled Thr':10s} | {'LF Thr':8s}")
    print(f"  {'-' * 27}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}")
    print(f"  {'CatBoost':27s} | {cb_oof_f1:.4f}     | {lf_cb_f1:.4f}       | {cb_thr:.2f}        | {lf_cb_thr:.2f}")
    print(f"  {'LightGBM':27s} | {lgb_oof_f1:.4f}     | {lf_lgb_f1:.4f}       | {lgb_thr:.2f}        | {lf_lgb_thr:.2f}")
    print(f"  {'XGBoost':27s} | {xgb_oof_f1:.4f}     | {lf_xgb_f1:.4f}       | {xgb_thr:.2f}        | {lf_xgb_thr:.2f}")

    # Ensemble optimization on last fold only
    print("\n  Ensemble optimization on last fold only:")
    lf_preds_dict = {"catboost": last_cb_pred, "lightgbm": last_lgb_pred, "xgboost": last_xgb_pred}
    lf_best_weights, lf_best_thr, lf_best_f1, lf_top_configs = optimize_ensemble(lf_preds_dict, last_y_va)

    print(f"\n  Last-fold top 5 weight combinations:")
    print(f"  {'Rank':4s} | {'CatBoost':8s} {'LightGBM':8s} {'XGBoost':8s} | {'F1':6s} | {'Threshold':9s}")
    print(f"  {'-' * 4}-+-{'-' * 8}-{'-' * 8}-{'-' * 8}-+-{'-' * 6}-+-{'-' * 9}")
    for rank, (w, thr, f1) in enumerate(lf_top_configs, 1):
        print(f"  {rank:4d} | {w['catboost']:8.1f} {w['lightgbm']:8.1f} {w['xgboost']:8.1f} | {f1:.4f} | {thr:.2f}")

    print(f"\n  Last-fold ensemble best: F1={lf_best_f1:.4f}, threshold={lf_best_thr:.2f}, "
          f"weights=(cat={lf_best_weights['catboost']:.1f}, lgb={lf_best_weights['lightgbm']:.1f}, xgb={lf_best_weights['xgboost']:.1f})")

    # Rank averaging alternative on last fold
    print("\n  Rank averaging alternative (last fold):")
    lf_rank_proba = rank_average(lf_preds_dict)
    lf_rank_thr, lf_rank_f1 = optimize_threshold(last_y_va, lf_rank_proba, "Rank Average last-fold")
    print(f"\n  Last-fold rank-averaged F1: {lf_rank_f1:.4f} (threshold={lf_rank_thr:.2f})")

    # Select best last-fold method (rank vs probability averaging)
    lf_use_rank = lf_rank_f1 > lf_best_f1
    if lf_use_rank:
        print(f"\n  >>> Last-fold: Rank averaging ({lf_rank_f1:.4f}) BEATS probability averaging ({lf_best_f1:.4f})")
        submission_method = "rank_average"
        submission_ens_f1 = lf_rank_f1
        submission_thr = lf_rank_thr
    else:
        print(f"\n  >>> Last-fold: Probability averaging ({lf_best_f1:.4f}) BEATS rank averaging ({lf_rank_f1:.4f})")
        submission_method = "probability_average"
        submission_ens_f1 = lf_best_f1
        submission_thr = lf_best_thr
    submission_weights = lf_best_weights  # used for probability averaging; kept for logging

    # --- Comparison: pooled OOF vs last-fold ---
    print(f"\n  {'=' * 68}")
    print(f"  COMPARISON: Pooled OOF vs Last-Fold (for final submission)")
    print(f"  {'=' * 68}")
    print(f"  Pooled OOF:  F1={final_ens_f1:.4f}, threshold={final_thr:.2f}, method={final_method}, "
          f"weights=(cat={best_weights['catboost']:.1f}, lgb={best_weights['lightgbm']:.1f}, xgb={best_weights['xgboost']:.1f})")
    print(f"  Last-fold:   F1={submission_ens_f1:.4f}, threshold={submission_thr:.2f}, method={submission_method}, "
          f"weights=(cat={submission_weights['catboost']:.1f}, lgb={submission_weights['lightgbm']:.1f}, xgb={submission_weights['xgboost']:.1f})")
    print(f"\n  >>> Using LAST-FOLD weights for final submission: "
          f"cat={submission_weights['catboost']:.1f}, "
          f"lgb={submission_weights['lightgbm']:.1f}, "
          f"xgb={submission_weights['xgboost']:.1f}, "
          f"threshold={submission_thr:.4f}")
    print(f"  (Last fold is most representative of test: most training data, temporally closest.)")

    # ------------------------------------------------------------------
    # Step 10: Feature importance (from last CV fold's models)
    # ------------------------------------------------------------------
    print_feature_importance(last_cb_model, last_lgb_model, last_xgb_model, feature_cols)

    # ------------------------------------------------------------------
    # Step 11: Adversarial validation (last fold val vs test)
    # ------------------------------------------------------------------
    adv_auc = adversarial_validation(last_val_fold_te, test_full_te, label_mappings, feature_cols)

    # ------------------------------------------------------------------
    # Step 12: Train final models on FULL training data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training FINAL models on full training data...")
    print("=" * 70)

    # Use best_iteration from the LAST fold (most training data, most representative)
    cb_best_final = all_cb_best[-1]
    lgb_best_final = all_lgb_best[-1]
    xgb_best_final = all_xgb_best[-1]
    print(f"  Using best iterations from last fold: CB={cb_best_final}, LGB={lgb_best_final}, XGB={xgb_best_final}")

    print("\n--- Final CatBoost ---")
    cb_final = train_final_catboost(X_train_full_cb, y_train_full, cb_cat_features, cb_best_final)

    print("\n--- Final LightGBM ---")
    lgb_final = train_final_lightgbm(X_train_full_num, y_train_full, lgb_best_final, categorical_feature=lgb_cat_features)

    print("\n--- Final XGBoost ---")
    xgb_final = train_final_xgboost(X_train_full_xgb, y_train_full, xgb_best_final)

    # ------------------------------------------------------------------
    # Step 13: Predict on test set and generate submission
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generating test predictions and submission...")
    print("=" * 70)

    cb_test_proba = cb_final.predict_proba(X_test_cb)[:, 1]
    lgb_test_proba = lgb_final.predict_proba(X_test_num)[:, 1]
    xgb_test_proba = xgb_final.predict_proba(X_test_xgb)[:, 1]

    test_preds_dict = {
        "catboost": cb_test_proba,
        "lightgbm": lgb_test_proba,
        "xgboost": xgb_test_proba,
    }

    # Apply selected ensemble method using LAST-FOLD weights/threshold
    print(f"\n  Using LAST-FOLD configuration for test predictions:")
    print(f"    Method: {submission_method}")
    if submission_method == "probability_average":
        print(f"    Weights: cat={submission_weights['catboost']:.1f}, "
              f"lgb={submission_weights['lightgbm']:.1f}, "
              f"xgb={submission_weights['xgboost']:.1f}")
    print(f"    Base threshold: {submission_thr:.4f}")

    if lf_use_rank:
        ensemble_test_proba = rank_average(test_preds_dict)
    else:
        ensemble_test_proba = np.zeros_like(cb_test_proba)
        for name in test_preds_dict:
            ensemble_test_proba += submission_weights[name] * test_preds_dict[name]

    # --- Positive-rate calibration (on top of last-fold threshold) ---
    # The last-fold threshold may produce too many positive predictions
    # relative to the train positive rate. If the deviation exceeds 20%,
    # adjust the threshold using a percentile-based (top-K) approach.
    train_rate = y_train_full.mean()
    original_preds = (ensemble_test_proba >= submission_thr).astype(int)
    original_rate = original_preds.mean()

    print(f"\n  Positive-rate calibration check (on top of last-fold threshold):")
    print(f"    Train positive rate:      {train_rate:.4f}")
    print(f"    Predicted positive rate:  {original_rate:.4f} (at last-fold threshold {submission_thr:.4f})")

    if abs(original_rate - train_rate) / train_rate > 0.20:
        # Calibrate: use percentile-based threshold
        # Allow up to 10% more positives than train rate (test may be slightly higher)
        target_rate = min(1.1 * train_rate, 0.18)
        n_positive_adjusted = int(target_rate * len(ensemble_test_proba))
        calibrated_threshold = float(np.sort(ensemble_test_proba)[::-1][n_positive_adjusted])
        print(f"    Rate calibration TRIGGERED: original_rate={original_rate:.4f}, target_rate={target_rate:.4f}")
        print(f"    Threshold adjusted: {submission_thr:.4f} -> {calibrated_threshold:.4f}")
        final_threshold = calibrated_threshold
    else:
        print(f"    Rate calibration NOT triggered (within 20% of train rate)")
        final_threshold = submission_thr

    ensemble_test_pred = (ensemble_test_proba >= final_threshold).astype(int)

    # Create submission
    submission = pd.DataFrame({
        "claim_id": test["claim_id"].values,
        "is_valid": ensemble_test_pred,
    })
    submission.to_csv("submission.csv", index=False)

    print(f"\n  Submission saved to: submission.csv")
    print(f"  Submission shape: {submission.shape}")
    print(f"  >>> FINAL SUBMISSION uses LAST-FOLD configuration <<<")
    print(f"  Ensemble method used: {submission_method}")
    if submission_method == "probability_average":
        print(f"  Weights: catboost={submission_weights['catboost']:.1f}, "
              f"lightgbm={submission_weights['lightgbm']:.1f}, "
              f"xgboost={submission_weights['xgboost']:.1f}")
    print(f"  Last-fold threshold: {submission_thr:.4f}")
    print(f"  Final threshold:     {final_threshold:.4f} (after rate calibration)")
    print(f"  Prediction distribution:")
    print(f"    is_valid=0 (rejected): {(submission['is_valid'] == 0).sum()} "
          f"({(submission['is_valid'] == 0).mean():.2%})")
    print(f"    is_valid=1 (valid):    {(submission['is_valid'] == 1).sum()} "
          f"({(submission['is_valid'] == 1).mean():.2%})")

    # ------------------------------------------------------------------
    # Step 14: Experiment history logging
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Logging experiment results...")
    print("=" * 70)

    experiment_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "cv_folds": 5,
        "catboost_oof_f1": float(cb_oof_f1),
        "lightgbm_oof_f1": float(lgb_oof_f1),
        "xgboost_oof_f1": float(xgb_oof_f1),
        # Pooled OOF ensemble (for comparison)
        "pooled_oof_ensemble_f1": float(final_ens_f1),
        "pooled_oof_method": final_method,
        "pooled_oof_threshold": float(final_thr),
        "pooled_oof_weights": {k: float(v) for k, v in best_weights.items()},
        # Last-fold ensemble (used for final submission)
        "last_fold_ensemble_f1": float(submission_ens_f1),
        "last_fold_method": submission_method,
        "last_fold_threshold": float(submission_thr),
        "last_fold_weights": {k: float(v) for k, v in submission_weights.items()},
        # Final submission config
        "final_config_used": "last_fold",
        "final_threshold": float(final_threshold),
        "rate_calibration_triggered": bool(abs(original_rate - train_rate) / train_rate > 0.20),
        "n_features": len(feature_cols),
        "submission_positive_rate": float(submission["is_valid"].mean()),
        "train_positive_rate": float(train_rate),
        "adversarial_auc": float(adv_auc),
    }
    with open("experiment_history.jsonl", "a") as f:
        f.write(json.dumps(experiment_log) + "\n")
    print("  Experiment logged to experiment_history.jsonl")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  CV: 5-fold TimeSeriesSplit with OOF predictions ({int(oof_mask.sum())} pooled)")
    print(f"  Pooled OOF F1 scores (individual models):")
    print(f"    CatBoost:  {cb_oof_f1:.4f} (threshold={cb_thr:.2f})")
    print(f"    LightGBM:  {lgb_oof_f1:.4f} (threshold={lgb_thr:.2f})")
    print(f"    XGBoost:   {xgb_oof_f1:.4f} (threshold={xgb_thr:.2f})")
    print(f"  --- Ensemble optimization (both strategies, for comparison) ---")
    print(f"  Pooled OOF ensemble:")
    print(f"    F1={final_ens_f1:.4f}, method={final_method}, threshold={final_thr:.4f}, "
          f"weights=(cat={best_weights['catboost']:.1f}, lgb={best_weights['lightgbm']:.1f}, xgb={best_weights['xgboost']:.1f})")
    print(f"  Last-fold ensemble:")
    print(f"    F1={submission_ens_f1:.4f}, method={submission_method}, threshold={submission_thr:.4f}, "
          f"weights=(cat={submission_weights['catboost']:.1f}, lgb={submission_weights['lightgbm']:.1f}, xgb={submission_weights['xgboost']:.1f})")
    print(f"  >>> FINAL SUBMISSION uses LAST-FOLD configuration (most representative of test)")
    print(f"  Adversarial validation AUC: {adv_auc:.4f}")
    print(f"  Rate calibration triggered: {abs(original_rate - train_rate) / train_rate > 0.20}")
    print(f"  Submission threshold: {final_threshold:.4f} (last-fold base: {submission_thr:.4f})")
    print(f"  Submission predictions:")
    print(f"    Total: {len(submission)}")
    print(f"    Valid (1):    {(submission['is_valid'] == 1).sum()}")
    print(f"    Rejected (0): {(submission['is_valid'] == 0).sum()}")
    print(f"  Positive rate: {submission['is_valid'].mean():.2%}")
    print(f"  (Train positive rate was: {train['is_valid'].mean():.2%})")
    print(f"  Total runtime: {total_time / 60:.1f} minutes ({total_time:.0f} seconds)")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
