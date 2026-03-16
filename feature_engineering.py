import pandas as pd
import numpy as np
from data_loader import load_data, temporal_split


def add_transaction_features(df):
    df = df.copy()
    # balance changes
    df["deltaOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["deltaDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # amount ratios
    df["amount_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["balance_left_ratio_orig"] = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1)

    # fraud pattern flags
    df["orig_drained"] = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_unchanged"] = (df["deltaDest"] == 0).astype(int)
    df["is_fraud_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    df["double_red_flag"] = (
        (df["orig_drained"] == 1) & (df["dest_unchanged"] == 1)
    ).astype(int)

    # account type
    df["orig_is_customer"] = df["nameOrig"].str.startswith("C").astype(int)
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(int)

    # time features
    df["hour_of_day"] = df["step"] % 24
    df["day_of_sim"] = df["step"] // 24

    # log amount (helps tree models)
    df["log_amount"] = np.log1p(df["amount"])

    return df


def add_account_stats(train_df, df):
    stats = train_df.groupby("nameOrig").agg(
        sender_tx_count=("amount", "count"),
        sender_avg_amount=("amount", "mean"),
        sender_max_amount=("amount", "max"),
        sender_unique_dest=("nameDest", "nunique")
    ).reset_index()

    df = df.merge(stats, on="nameOrig", how="left")

    stat_cols = [
        "sender_tx_count",
        "sender_avg_amount",
        "sender_max_amount",
        "sender_unique_dest"
    ]
    df[stat_cols] = df[stat_cols].fillna(0)

    return df


def build_features(train_df, test_df):
    train_feat = add_transaction_features(train_df)
    train_feat = add_account_stats(train_df, train_feat)
    test_feat = add_transaction_features(test_df)
    test_feat = add_account_stats(train_df, test_feat)

    print("Train shape:", train_feat.shape)
    print("Test shape:", test_feat.shape)

    return train_feat, test_feat


def get_feature_columns(df):
    exclude = ["step", "isFraud", "nameOrig", "nameDest", "type"]

    features = [
        col for col in df.columns
        if col not in exclude and df[col].dtype != "object"
    ]

    return features


if __name__ == "__main__":
    df = load_data("paysim.csv")
    train_df, test_df = temporal_split(df)
    train_feat, test_feat = build_features(train_df, test_df)
    feature_cols = get_feature_columns(train_feat)

    print("\nFeature columns:", len(feature_cols))
    print(feature_cols)