import pandas as pd
import networkx as nx
from data_loader import load_data, temporal_split

def build_graph(df):
    G = nx.DiGraph()
    nodes = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
    G.add_nodes_from(nodes)

    # edges
    edges = list(zip(df["nameOrig"], df["nameDest"]))
    G.add_edges_from(edges)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    return G


def compute_node_features(G):
    pagerank = nx.pagerank(G)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    rows = []
    for node in G.nodes():
        ind = in_deg.get(node, 0)
        outd = out_deg.get(node, 0)
        rows.append({
            "account": node,
            "in_degree": ind,
            "out_degree": outd,
            "total_degree": ind + outd,
            "pagerank": pagerank.get(node, 0),
            "is_merchant": int(node.startswith("M"))
        })

    node_df = pd.DataFrame(rows)
    print("Node feature shape:", node_df.shape)
    return node_df


def attach_node_features(df, node_df):
    orig_features = node_df.add_prefix("orig_").rename(columns={"orig_account": "nameOrig"})
    dest_features = node_df.add_prefix("dest_").rename(columns={"dest_account": "nameDest"})
    df = df.merge(orig_features, on="nameOrig", how="left")
    df = df.merge(dest_features, on="nameDest", how="left")
    print("Edge dataset shape:", df.shape)
    return df


def print_graph_stats(G):
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Density:", nx.density(G))

    components = list(nx.weakly_connected_components(G))
    largest = max(len(c) for c in components)
    print("Connected components:", len(components))
    print("Largest component:", largest)


if __name__ == "__main__":
    df = load_data("data/raw/paysim.csv")
    train_df, _ = temporal_split(df)
    train_df = train_df[train_df["type"].isin(["TRANSFER", "CASH_OUT"])]

    G = build_graph(train_df)
    node_df = compute_node_features(G)
    edge_df = attach_node_features(train_df, node_df)

    print_graph_stats(G)
    node_df.to_parquet("data/processed/node_features.parquet", index=False)
    edge_df.to_parquet("data/processed/graph_edges.parquet", index=False)