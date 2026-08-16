from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

RANDOM_SEED = 42
TE_ALPHA = 10

TE_COLUMNS = [
    "claim_type",
    "claim_reason_start",
    "claim_user_registered_phone_country_id",
    "platform",
]

AGE_MAP = {
    "0_13": 0,
    "14_17": 1,
    "18_24": 2,
    "25_34": 3,
    "35_44": 4,
    "45_54": 5,
    "55_64": 6,
    "65_plus": 7,
}

FRIENDS_MAP = {
    "0": 0,
    "1_5": 1,
    "6_20": 2,
    "21_50": 3,
    "51_100": 4,
    "101_250": 5,
    "251_500": 6,
    "501_1000": 7,
    "1001_plus": 8,
}

DROP_COLS = [
    "claim_id",
    "id_content",
    "first_event_time",
    "content_registered_time",
    "ip_country_id",
    "id_content_owner",
    "age_bucket",
    "claim_user_age_bucket",
    "friends_bucket",
    "claim_user_friends_bucket",
]

CAT_FEATURES = [
    "os",
    "platform",
    "sex",
    "claim_user_sex",
    "claim_type",
    "claim_reason_start",
    "registered_phone_country_id",
    "mobile_phone_country_id",
    "profile_country_id",
    "claim_user_registered_phone_country_id",
    "claim_user_profile_country_id",
    "claim_type_reason",
]


def compute_agg_counts(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    combined = pd.concat([train.drop(columns=["is_valid"]), test], axis=0, ignore_index=True)
    total_rows = len(combined)

    agg_mappings = {
        "content_owner_complaint_count": combined.groupby("id_content_owner").size().to_dict(),
        "content_complaint_count": combined.groupby("id_content").size().to_dict(),
        "claim_type_count": combined.groupby("claim_type").size().to_dict(),
        "claim_reason_count": combined.groupby("claim_reason_start").size().to_dict(),
    }

    agg_mappings["id_content_owner_freq"] = {
        key: value / total_rows
        for key, value in agg_mappings["content_owner_complaint_count"].items()
    }
    agg_mappings["claim_type_freq"] = {
        key: value / total_rows for key, value in agg_mappings["claim_type_count"].items()
    }
    agg_mappings["claim_reason_freq"] = {
        key: value / total_rows for key, value in agg_mappings["claim_reason_count"].items()
    }
    agg_mappings["os_freq"] = (combined.groupby("os").size() / total_rows).to_dict()
    agg_mappings["registered_phone_country_freq"] = (
        combined.groupby("registered_phone_country_id").size() / total_rows
    ).to_dict()

    agg_mappings["mean_bot_score_by_claim_type"] = (
        combined.groupby("claim_type")["claim_user_bot_prediction_score"].mean().to_dict()
    )
    agg_mappings["mean_bot_score_by_reason"] = (
        combined.groupby("claim_reason_start")["claim_user_bot_prediction_score"].mean().to_dict()
    )
    agg_mappings["mean_user_bot_by_claim_type"] = (
        combined.groupby("claim_type")["user_bot_prediction_score"].mean().to_dict()
    )
    agg_mappings["std_bot_score_by_claim_type"] = (
        combined.groupby("claim_type")["claim_user_bot_prediction_score"].std().fillna(0).to_dict()
    )
    agg_mappings["mean_likes_by_claim_type"] = (
        combined.groupby("claim_type")["additional_likes_count"].mean().to_dict()
    )
    return agg_mappings


def engineer_features(df: pd.DataFrame, agg_mappings: dict) -> pd.DataFrame:
    df = df.copy()
    df["first_event_time"] = pd.to_datetime(df["first_event_time"])
    df["content_registered_time"] = pd.to_datetime(df["content_registered_time"])

    df["hour"] = df["first_event_time"].dt.hour
    df["dayofweek"] = df["first_event_time"].dt.dayofweek
    df["day"] = df["first_event_time"].dt.day
    df["month"] = df["first_event_time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["content_hour"] = df["content_registered_time"].dt.hour
    df["content_dayofweek"] = df["content_registered_time"].dt.dayofweek
    df["content_month"] = df["content_registered_time"].dt.month
    df["content_age_days"] = (
        (df["first_event_time"] - df["content_registered_time"]).dt.total_seconds() / 86400.0
    ).clip(lower=0)

    df["sender_account_age"] = (df["first_event_time"].dt.year - df["registered_year"]).clip(lower=0)
    df["claim_user_account_age"] = (
        df["first_event_time"].dt.year - df["claim_user_registered_year"]
    ).clip(lower=0)

    df["bot_score_diff"] = df["claim_user_bot_prediction_score"] - df["user_bot_prediction_score"]
    df["bot_score_sum"] = df["claim_user_bot_prediction_score"] + df["user_bot_prediction_score"]
    df["claim_user_is_high_bot"] = (df["claim_user_bot_prediction_score"] > 0.5).astype(int)
    df["user_is_high_bot"] = (df["user_bot_prediction_score"] > 0.5).astype(int)

    df["country_phone_match"] = (
        df["registered_phone_country_id"] == df["claim_user_registered_phone_country_id"]
    ).astype(int)
    df["country_profile_match"] = (
        df["profile_country_id"] == df["claim_user_profile_country_id"]
    ).astype(int)

    df["age_bucket_encoded"] = df["age_bucket"].map(AGE_MAP)
    df["claim_user_age_bucket_encoded"] = df["claim_user_age_bucket"].map(AGE_MAP)
    df["friends_bucket_encoded"] = df["friends_bucket"].map(FRIENDS_MAP)
    df["claim_user_friends_bucket_encoded"] = df["claim_user_friends_bucket"].map(FRIENDS_MAP)

    df["content_owner_complaint_count"] = (
        df["id_content_owner"].map(agg_mappings["content_owner_complaint_count"]).fillna(1)
    )
    df["content_complaint_count"] = df["id_content"].map(agg_mappings["content_complaint_count"]).fillna(1)
    df["claim_type_count"] = df["claim_type"].map(agg_mappings["claim_type_count"]).fillna(1)
    df["claim_reason_count"] = df["claim_reason_start"].map(agg_mappings["claim_reason_count"]).fillna(1)

    df["log_additional_likes"] = np.log1p(df["additional_likes_count"])
    df["log_additional_friend_request"] = np.log1p(df["additional_friend_request_count"])
    df["log_additional_friend_accept"] = np.log1p(df["additional_friend_accept_count"])

    df["claim_type_reason"] = (
        df["claim_type"].astype(str) + "_" + df["claim_reason_start"].astype(str)
    )
    df["sex_match"] = (df["sex"] == df["claim_user_sex"]).astype(int)
    df["age_diff"] = df["claim_user_age_bucket_encoded"] - df["age_bucket_encoded"]
    df["friends_diff"] = (
        df["claim_user_friends_bucket_encoded"] - df["friends_bucket_encoded"]
    )

    df["id_content_owner_freq"] = (
        df["id_content_owner"].map(agg_mappings["id_content_owner_freq"]).fillna(0)
    )
    df["claim_type_freq"] = df["claim_type"].map(agg_mappings["claim_type_freq"]).fillna(0)
    df["claim_reason_freq"] = df["claim_reason_start"].map(agg_mappings["claim_reason_freq"]).fillna(0)
    df["os_freq"] = df["os"].map(agg_mappings["os_freq"]).fillna(0)
    df["registered_phone_country_freq"] = (
        df["registered_phone_country_id"].map(agg_mappings["registered_phone_country_freq"]).fillna(0)
    )

    df["mean_bot_score_by_claim_type"] = (
        df["claim_type"].map(agg_mappings["mean_bot_score_by_claim_type"]).fillna(0)
    )
    df["mean_bot_score_by_reason"] = (
        df["claim_reason_start"].map(agg_mappings["mean_bot_score_by_reason"]).fillna(0)
    )
    df["mean_user_bot_by_claim_type"] = (
        df["claim_type"].map(agg_mappings["mean_user_bot_by_claim_type"]).fillna(0)
    )
    df["std_bot_score_by_claim_type"] = (
        df["claim_type"].map(agg_mappings["std_bot_score_by_claim_type"]).fillna(0)
    )
    df["mean_likes_by_claim_type"] = (
        df["claim_type"].map(agg_mappings["mean_likes_by_claim_type"]).fillna(0)
    )

    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["content_age_bucket"] = pd.cut(
        df["content_age_days"],
        bins=[-np.inf, 1, 7, 30, 365, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["is_recent_content"] = (df["content_age_days"] <= 7).astype(int)

    df["registration_year_diff"] = df["claim_user_registered_year"] - df["registered_year"]
    df["is_same_registration_year"] = (
        df["registered_year"] == df["claim_user_registered_year"]
    ).astype(int)
    df["sender_account_age_squared"] = df["sender_account_age"] ** 2

    df["claim_type_x_platform"] = df["claim_type"] * 10 + df["platform"]
    df["bot_score_x_are_friends"] = df["claim_user_bot_prediction_score"] * df["are_friends"]
    df["content_age_x_claim_type"] = df["content_age_days"] * df["claim_type"] / 1000.0
    df["is_both_high_bot"] = (
        (df["user_is_high_bot"] == 1) & (df["claim_user_is_high_bot"] == 1)
    ).astype(int)
    df["friends_diff_abs"] = (
        df["claim_user_friends_bucket_encoded"] - df["friends_bucket_encoded"]
    ).abs()

    return df.drop(columns=[column for column in DROP_COLS if column in df.columns])


def kfold_target_encoding(
    train_df: pd.DataFrame,
    target_col: str = "is_valid",
    alpha: float = TE_ALPHA,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    df = train_df.copy()
    global_mean = df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in TE_COLUMNS:
        te_col = f"{col}_te"
        te_values = np.full(len(df), global_mean, dtype=np.float64)
        for tr_idx, va_idx in kf.split(df):
            fold_train = df.iloc[tr_idx]
            stats = fold_train.groupby(col)[target_col].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (
                stats["count"] + alpha
            )
            te_values[va_idx] = df.iloc[va_idx][col].map(smoothed.to_dict()).fillna(global_mean).values
        df[te_col] = te_values
    return df


def apply_te_to_new_data(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    target_col: str = "is_valid",
    alpha: float = TE_ALPHA,
) -> pd.DataFrame:
    encoded = new_df.copy()
    global_mean = train_df[target_col].mean()

    for col in TE_COLUMNS:
        stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (
            stats["count"] + alpha
        )
        encoded[f"{col}_te"] = encoded[col].map(smoothed.to_dict()).fillna(global_mean)
    return encoded
