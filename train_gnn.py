import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv

from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader import load_data, temporal_split
from feature_engineering import build_features, get_feature_columns
from graph_builder import build_graph, compute_node_features


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
LR = 0.001
HIDDEN = 64


class GraphSAGE(nn.Module):

    def __init__(self, in_channels, hidden):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):

    def __init__(self, in_channels, hidden):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden)
        self.conv2 = GATConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class EdgeModel(nn.Module):

    def __init__(self, gnn, hidden):
        super().__init__()
        self.gnn = gnn
        self.linear = nn.Linear(hidden * 2, 1)

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        src, dst = data.edge_index
        edge_embed = torch.cat([x[src], x[dst]], dim=1)
        out = self.linear(edge_embed)
        return out.squeeze()



def build_pyg_data(edge_df, node_df, node_features):

    accounts = node_df["account"].tolist()
    mapping = {a: i for i, a in enumerate(accounts)}
    x = torch.tensor(node_df[node_features].values, dtype=torch.float)
    src = torch.tensor(edge_df["nameOrig"].map(mapping).values)
    dst = torch.tensor(edge_df["nameDest"].map(mapping).values)
    edge_index = torch.stack([src, dst], dim=0)

    y = torch.tensor(edge_df["isFraud"].values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data.to(DEVICE)


def train(model, data):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    y = data.y

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, "loss:", round(loss.item(),4))


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy()
    y = data.y.cpu().numpy()
    roc = roc_auc_score(y, probs)
    pr = average_precision_score(y, probs)
    print("ROC-AUC:", round(roc,4))
    print("PR-AUC :", round(pr,4))


def main():

    df = load_data("data/raw/paysim.csv")
    train_df, test_df = temporal_split(df)
    train_feat, test_feat = build_features(train_df, test_df)
    # Build graph
    full_df = pd.concat([train_feat, test_feat])
    G = build_graph(full_df)
    node_df = compute_node_features(G)
    node_features = ["in_degree", "out_degree", "total_degree", "pagerank", "is_merchant"]
    train_data = build_pyg_data(train_feat, node_df, node_features)
    test_data = build_pyg_data(test_feat, node_df, node_features)

    print("Nodes:", train_data.num_nodes)
    print("Edges:", train_data.num_edges)
    in_dim = len(node_features)

    # GraphSAGE
    print("\nTraining GraphSAGE")
    sage = GraphSAGE(in_dim, HIDDEN).to(DEVICE)
    sage_model = EdgeModel(sage, HIDDEN).to(DEVICE)
    train(sage_model, train_data)
    evaluate(sage_model, test_data)

    # GAT
    print("\nTraining GAT")
    gat = GAT(in_dim, HIDDEN).to(DEVICE)
    gat_model = EdgeModel(gat, HIDDEN).to(DEVICE)
    train(gat_model, train_data)
    evaluate(gat_model, test_data)


if __name__ == "__main__":
    main()