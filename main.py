import pandas as pd
from data_loader import load_data, temporal_split
from feature_engineering import build_features
from graph_builder import build_graph, compute_node_features
import tune_models
import train_baseline
import train_gnn
import evaluate


def main():
    print("\nGraph Fraud Detection Pipeline\n")

    # step 1 — Load data
    print("Step 1: Load dataset")
    df = load_data("paysim.csv")

    # step 2 — Train/test split
    print("\nStep 2: Temporal split")
    train_df, test_df = temporal_split(df)

    # step 3 — Feature engineering
    print("\nStep 3: Feature engineering")
    train_feat, test_feat = build_features(train_df, test_df)

    # step 4 — Graph construction
    print("\nStep 4: Build transaction graph")
    full_df = pd.concat([train_feat, test_feat])
    G = build_graph(full_df)
    node_df = compute_node_features(G)
    print("Graph nodes:", len(node_df))

    # step 5 — Hyperparameter tuning
    print("\nStep 5: Hyperparameter tuning")
    tune_models.main()

    # step 6 — Train baseline models
    print("\nStep 6: Train baseline models")
    train_baseline.main()

    # step 7 — Train graph models
    print("\nStep 7: Train graph neural networks")
    train_gnn.main()

    # step 8 — Evaluate models
    print("\nStep 8: Evaluate models")
    evaluate.main()
    print("\nPipeline finished successfully")

if __name__ == "__main__":
    main()