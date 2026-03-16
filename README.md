# PaySim Fraud Detection

A machine learning pipeline for detecting financial fraud in mobile money transactions using the [Kaggle PaySim1 dataset](https://www.kaggle.com/datasets/ealaxi/paysim1). The project combines classical tree-based classifiers with Graph Neural Networks (GNNs) to identify fraudulent TRANSFER and CASH_OUT transactions.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Rows | ~6.36 million transactions |
| Time span | 744 steps (simulated 31 days) |
| Fraud rate | ~0.13% (highly imbalanced) |
| Fraud types | TRANSFER and CASH_OUT only |

Download `paysim.csv` from Kaggle and place it in the project root before running anything.

---

## Project Structure

```
├── data_loader.py          # Load, clean, and split the dataset
├── feature_engineering.py  # Transaction and account-level features
├── graph_builder.py        # Build NetworkX graph and compute node features
├── train_baseline.py       # Train RandomForest and XGBoost baselines
├── train_gnn.py            # Train GraphSAGE and GAT edge classifiers
├── tune_models.py          # Hyperparameter search via RandomizedSearchCV
├── evaluate.py             # Metrics, ROC/PR curves, feature importance
├── predict_fraud.py        # Score a single transaction using saved model
├── main.py                 # End-to-end pipeline runner
├── EDA_fraud_paysim.ipynb  # Exploratory data analysis notebook
├── requirements.txt
└── models/                 # Saved model files (created at runtime)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **PyTorch Geometric** requires a separate install step matching your PyTorch and CUDA version.  
> See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Running the Full Pipeline

```bash
python main.py
```

This runs all steps in order:

1. Load and clean the dataset
2. Temporal train/test split (80/20 by time step — no leakage)
3. Feature engineering
4. Build the transaction graph (NetworkX DiGraph)
5. Hyperparameter tuning (RandomizedSearchCV)
6. Train baseline models (RandomForest, XGBoost)
7. Train GNN models (GraphSAGE, GAT)
8. Evaluate all models (ROC-AUC, PR-AUC, classification report, curves)

Trained models are saved to `models/`.

---

## Running Steps Individually

```bash
# EDA
jupyter notebook EDA_fraud_paysim.ipynb

# Hyperparameter tuning
python tune_models.py

# Train baselines only
python train_baseline.py

# Train GNN models only
python train_gnn.py

# Evaluate saved models
python evaluate.py
```

---

## Predicting on a Single Transaction

Use `predict_fraud.py` to score any transaction with the saved XGBoost model.

### From the command line

```bash
python predict_fraud.py
```

This runs the built-in example transaction and prints the fraud probability:

```
Fraud probability: 0.9732
```

### From your own code

```python
from predict_fraud import predict

transaction = {
    "step": 200,
    "type": "TRANSFER",
    "amount": 7500,
    "nameOrig": "C123456",
    "oldbalanceOrg": 7500,
    "newbalanceOrig": 0,
    "nameDest": "C654321",
    "oldbalanceDest": 12000,
    "newbalanceDest": 12000,
}

fraud_probability = predict(transaction)
print(f"Fraud probability: {fraud_probability:.4f}")

# Apply a threshold
if fraud_probability > 0.5:
    print("⚠️  Transaction flagged as FRAUD")
else:
    print("✅  Transaction looks legitimate")
```

### Required input fields

| Field | Type | Description |
|---|---|---|
| `step` | int | Time step (1–744) |
| `type` | str | Transaction type: `TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `TRANSFER` |
| `amount` | float | Transaction amount |
| `nameOrig` | str | Sender account ID (e.g. `C123456`) |
| `oldbalanceOrg` | float | Sender balance before transaction |
| `newbalanceOrig` | float | Sender balance after transaction |
| `nameDest` | str | Receiver account ID (`C` = customer, `M` = merchant) |
| `oldbalanceDest` | float | Receiver balance before transaction |
| `newbalanceDest` | float | Receiver balance after transaction |

> **Note:** `predict_fraud.py` uses only transaction-level features. Account-level stats (e.g. `sender_tx_count`) are set to 0 for single-transaction scoring since there is no historical context. For batch scoring with full history, use `evaluate.py` instead.

### High-risk transaction patterns

The model is most sensitive to these fraud signals (derived from EDA):

- Sender balance fully drained (`newbalanceOrig == 0` after transfer)
- Destination balance unchanged after receiving funds (`deltaDest == 0`)
- Transaction type is `TRANSFER` or `CASH_OUT`
- Both drain and unchanged-destination flags set simultaneously (`double_red_flag`)

---

## Models

| Model | Type | Imbalance handling |
|---|---|---|
| RandomForest | Tree ensemble | `class_weight="balanced"` |
| XGBoost | Gradient boosting | `scale_pos_weight` |
| GraphSAGE | Graph Neural Network | BCEWithLogitsLoss |
| GAT | Graph Neural Network | BCEWithLogitsLoss |

**Evaluation metric:** PR-AUC (average precision) is the primary metric due to severe class imbalance. ROC-AUC is tracked as a secondary metric.

---

## Key Features Engineered

- `deltaOrig` / `deltaDest` — balance change for sender and receiver
- `amount_ratio_orig` — amount relative to sender's starting balance
- `orig_drained` — flag: sender balance reaches zero
- `dest_unchanged` — flag: receiver balance does not change
- `double_red_flag` — both drain flags set simultaneously
- `is_fraud_type` — transaction type is TRANSFER or CASH_OUT
- `log_amount` — log-transformed amount
- `sender_tx_count`, `sender_avg_amount`, `sender_max_amount`, `sender_unique_dest` — account velocity features (built from training data only)
- Graph node features: `in_degree`, `out_degree`, `pagerank`, `is_merchant`

---

## License

This project uses the PaySim dataset which is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
