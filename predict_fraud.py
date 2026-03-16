import pickle
import pandas as pd

from feature_engineering import add_transaction_features


def load_model():

    with open("models/xgboost.pkl","rb") as f:
        model = pickle.load(f)

    return model


def predict(transaction_dict):

    model = load_model()

    df = pd.DataFrame([transaction_dict])

    df = add_transaction_features(df)

    features = [
        col for col in df.columns
        if df[col].dtype != "object" and col not in ["isFraud"]
    ]

    proba = model.predict_proba(df[features])[:,1][0]

    return proba


if __name__ == "__main__":

    example_tx = {

        "step": 10,
        "type": "TRANSFER",
        "amount": 5000,
        "nameOrig": "C123",
        "oldbalanceOrg": 6000,
        "newbalanceOrig": 1000,
        "nameDest": "C456",
        "oldbalanceDest": 2000,
        "newbalanceDest": 7000
    }

    p = predict(example_tx)

    print("Fraud probability:", p)