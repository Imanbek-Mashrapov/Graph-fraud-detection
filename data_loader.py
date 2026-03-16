import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load PaySim dataset from CSV and perform basic cleaning.
    """
    print("Loading data...")
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)

    if "isFlaggedFraud" in df.columns:
        df = df.drop(columns=["isFlaggedFraud"])

    df = df.drop_duplicates()

    # Fix negative balances (artifact of PaySim generator)
    balance_cols = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    for col in balance_cols:
        df[col] = df[col].clip(lower=0)

    # Remove zero-amount transactions
    df = df[df["amount"] > 0]

    # Sort by time
    df = df.sort_values("step").reset_index(drop=True)

    fraud_rate = df["isFraud"].mean() * 100
    print(f"Fraud rate: {fraud_rate:.3f}%")

    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Split dataset by time to avoid data leakage.
    Earlier transactions go to train, later to test.
    """

    cutoff = df["step"].quantile(train_frac)

    train = df[df["step"] <= cutoff].reset_index(drop=True)
    test = df[df["step"] > cutoff].reset_index(drop=True)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    return train, test


def save_parquet(df: pd.DataFrame, path: str):
    """
    Save dataframe in parquet format.
    """
    df.to_parquet(path, index=False)
    print("Saved:", path)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load parquet file.
    """
    df = pd.read_parquet(path)
    print("Loaded:", path)

    return df


if __name__ == "__main__":

    df = load_data("data/raw/paysim.csv")

    save_parquet(df, "data/processed/transactions_clean.parquet")

    train_df, test_df = temporal_split(df)

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)