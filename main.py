import pandas as pd
from data_loader import load_data, temporal_split
from feature_engineering import build_features
from graph_builder import build_graph, compute_node_features
import train_baseline
import train_gnn
import evaluate


def main():
    print("Starting fraud detection pipeline\n")
    # 1. load data
    print("Step 1: Load data")
    df = load_data("data/raw/paysim.csv")

    # 2. train / test split
    print("\nStep 2: Split dataset")
    train_df, test_df = temporal_split(df)

    # 3. feature engineering
    print("\nStep 3: Feature engineering")
    train_feat, test_feat = build_features(train_df, test_df)

    # 4. build graph
    print("\nStep 4: Build transaction graph")
    full_df = pd.concat([train_feat, test_feat])
    G = build_graph(full_df)
    node_df = compute_node_features(G)
    print("Graph nodes:", len(node_df))

    # 5. train baseline models
    print("\nStep 5: Train baseline models")
    train_baseline.main()

    # 6. train GNN models
    print("\nStep 6: Train GNN models")
    train_gnn.main()

    # 7. evaluate models
    print("\nStep 7: Evaluate models")
    evaluate.main()
    print("\nPipeline finished")

if __name__ == "__main__":
    main()