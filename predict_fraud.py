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

    df["sender_tx_count"] = 0
    df["sender_avg_amount"] = 0
    df["sender_max_amount"] = 0
    df["sender_unique_dest"] = 0

    if "step" in df.columns:
        df = df.drop(columns=["step"])

    features = model.get_booster().feature_names

    proba = model.predict_proba(df[features])[:,1][0]

    return proba


if __name__ == "__main__":

    
    example_tx = {

        "step": 200,
        "type": "TRANSFER",
        "amount": 7500,

        "nameOrig": "C999999",
        "oldbalanceOrg": 7500,
        "newbalanceOrig": 0,

        "nameDest": "C888888",
        "oldbalanceDest": 12000,
        "newbalanceDest": 12000
    }


    p = predict(example_tx)

    print("Fraud probability:", p)