import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

from data_loader import load_data, temporal_split
from feature_engineering import build_features, get_feature_columns


MODELS = ["random_forest", "xgboost", "lightgbm"]


def load_model(name):
    with open(f"models/{name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    df = load_data("data/raw/paysim.csv")
    train_df, test_df = temporal_split(df)
    train_feat, test_feat = build_features(train_df, test_df)
    feature_cols = get_feature_columns(train_feat)

    X_test = test_feat[feature_cols]
    y_test = test_feat["isFraud"]
    results = []

    plt.figure(figsize=(8,6))
    for name in MODELS:
        print("\nEvaluating:", name)

        model = load_model(name)
        proba = model.predict_proba(X_test)[:,1]
        preds = (proba > 0.5).astype(int)
        roc = roc_auc_score(y_test, proba)
        pr = average_precision_score(y_test, proba)

        print("ROC-AUC:", round(roc,4))
        print("PR-AUC :", round(pr,4))
        print(classification_report(y_test, preds))

        results.append({
            "model": name,
            "roc_auc": roc,
            "pr_auc": pr
        })

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0,1],[0,1],"k--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Precision-Recall curves
    plt.figure(figsize=(8,6))

    for name in MODELS:
        model = load_model(name)
        proba = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        plt.plot(recall, precision, label=name)

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # Summary table
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)

    print("\nModel Comparison")
    print(results_df)


if __name__ == "__main__":
    main()