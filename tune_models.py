import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_data, temporal_split
from feature_engineering import build_features, get_feature_columns


def main():

    df = load_data("paysim.csv")
    train_df, test_df = temporal_split(df)
    train_feat, test_feat = build_features(train_df, test_df)
    feature_cols = get_feature_columns(train_feat)

    X_train = train_feat[feature_cols]
    y_train = train_feat["isFraud"]

    print("Starting hyperparameter tuning")

    # RandomForest tuning
    rf = RandomForestClassifier()

    rf_params = {
        "n_estimators": [10, 15, 20],
        "max_depth": [3,4,5],
        "min_samples_split": [2,5,10]
    }

    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=5,
        scoring="average_precision",
        cv=3,
        verbose=1
    )

    rf_search.fit(X_train, y_train)

    print("Best RandomForest params")
    print(rf_search.best_params_)

    # XGBoost tuning

    xgb = XGBClassifier(eval_metric="aucpr")

    xgb_params = {
        "max_depth": [3,4,5,6],
        "learning_rate": [0.01,0.05,0.1],
        "n_estimators": [100,200]
    }

    xgb_search = RandomizedSearchCV(
        xgb,
        xgb_params,
        n_iter=5,
        scoring="average_precision",
        cv=3,
        verbose=1
    )

    xgb_search.fit(X_train, y_train)

    print("Best XGBoost params")
    print(xgb_search.best_params_)


if __name__ == "__main__":
    main()