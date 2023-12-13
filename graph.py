import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Credit Card Fraud Detection dataset
data = pd.read_csv("creditcard.csv")

# Standardize the 'Amount' column using StandardScaler
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Apply PCA to reduce dimensionality (excluding 'Time' and 'Class' columns)
pca = PCA(n_components=2)
data_pca = pd.DataFrame(pca.fit_transform(
    data.drop(['Time', 'Class'], axis=1)), columns=['PCA1', 'PCA2'])

# Create a graph
G = nx.Graph()

# Add nodes with features
for idx, row in data.iterrows():
    G.add_node(idx, features=row.drop(['Time', 'Class']))

# Add edges (connecting nodes with similar features, you can define your own criteria)
for i in range(len(data)):
    for j in range(i + 1, len(data)):
        similarity = sum(data.iloc[i].drop(
            ['Time', 'Class']) == data.iloc[j].drop(['Time', 'Class']))
        if similarity >= 28:  # You can adjust the threshold based on your criteria
            G.add_edge(i, j)

# Visualize the graph (for a small subset of the data)
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, node_size=10, node_color='b', with_labels=False)
plt.title("Graph Representation of Credit Card Transactions")
plt.show()
