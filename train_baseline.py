import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_loader import load_data, temporal_split
from feature_engineering import build_features, get_feature_columns


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def get_models(scale_pos_weight):

    return {

        "random_forest": RandomForestClassifier(
            n_estimators=20,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42
        )
    }


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    print(name)
    print("ROC-AUC:", round(roc,4))
    print("PR-AUC :", round(pr,4))
    print()
    print(classification_report(y_test, y_pred))

    return {"model": name, "roc_auc": roc, "pr_auc": pr}


def save_model(model, name):

    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print("Saved:", path)


def main():

    df = load_data("paysim.csv")
    train_df, test_df = temporal_split(df)
    train_feat, test_feat = build_features(train_df, test_df)
    feature_cols = get_feature_columns(train_feat)

    X_train = train_feat[feature_cols]
    y_train = train_feat["isFraud"]
    X_test = test_feat[feature_cols]
    y_test = test_feat["isFraud"]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("Samples:", len(X_train))
    print("Fraud rate:", y_train.mean())

    models = get_models(scale_pos_weight)

    results = []

    for name, model in models.items():

        print("\nTraining", name)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, name)
        save_model(model, name)
        results.append(metrics)

    print("\nSummary")
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print(results_df)

if __name__ == "__main__":
    main()