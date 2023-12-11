import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Assuming you have a dataframe 'df' with the Credit Card Fraud Detection dataset

# Preprocessing
features = df.drop(['Class'], axis=1).values
labels = df['Class'].values
edge_index = ...  # Construct the edge index based on transaction relationships

# Train-test split
x_train, x_test, y_train, y_test, edge_train, edge_test = train_test_split(
    features, labels, edge_index, test_size=0.2, random_state=42
)

# Convert data to PyTorch Geometric format
train_data = Data(x=torch.tensor(x_train, dtype=torch.float32),
                  edge_index=torch.tensor(edge_train, dtype=torch.long).t().contiguous(),
                  y=torch.tensor(y_train, dtype=torch.float32))
test_data = Data(x=torch.tensor(x_test, dtype=torch.float32),
                 edge_index=torch.tensor(edge_test, dtype=torch.long).t().contiguous(),
                 y=torch.tensor(y_test, dtype=torch.float32))

# Create DataLoader
train_loader = DataLoader([train_data], batch_size=64, shuffle=True)
test_loader = DataLoader([test_data], batch_size=64, shuffle=False)

# Define a simple Graph Convolutional Network (GCN) model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(x_train.shape[1], 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Training
model.train()
for epoch in range(50):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y.unsqueeze(1))
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
y_pred = []
with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        y_pred.extend(torch.sigmoid(output).cpu().numpy())

# Convert predictions to binary (fraud or not fraud)
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
