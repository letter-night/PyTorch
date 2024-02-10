import torch, requests
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchmetrics import Accuracy 

# Download helper functions from repository (if not already downloaded) ------------------------------------------------- #
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

def accuracy_fn(y_true, y_pred):
	correct = torch.eq(y_true, y_pred).sum().item()
	acc = (correct / len(y_pred)) * 100
	return acc

# -------------------------------------------------------------------------------------------------------------------------- #
# print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# 1. Create multi-class classification data
from sklearn.datasets import make_blobs 

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1) Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES, # X features
                            centers=NUM_CLASSES, # y labels
                            cluster_std=1.5, # give the clusters a little shake up (by default, 1.0)
                            random_state=RANDOM_SEED)

# 2) Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long) # <- "long" type!!!!

# 3) Split into train & test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=.2, random_state=RANDOM_SEED)

# 4) Plot data
# plt.figure(figsize=(9,6))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

# 2. Building a multi-class classification model (with/without nonlinearity)
# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """
        Initializes all required hyperparameters for a multi-class classification model.
        
        Args:
        input_features (int): Number of input features to the model.
        output_features (int): Number of output features of the model (how many classses there are).
        hidden_units (int): Number of hidden units between layers, default=8.
        """
        super().__init__()
        
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
		)
    
    def forward(self, x):
            return self.linear_layer_stack(x)

# Create an instance of BlobModel and send it to the target device
model_4 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,
                    hidden_units=8).to(device)

# print(model_4)

# 3. Create loss function & optimizer 
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=.1)

# 4. Check with some sample data - Getting prediction probs for a multi-class
# Perform a single forward pass on the data
# print(model_4(X_blob_train.to(device))[:5])

# Make prediction logits with model
# y_logits = model_4(X_blob_test.to(device))

# Perform softmax calculation on logits across dimension 1 to get prediction probs
# y_pred_probs = torch.softmax(y_logits, dim=1)
# print(y_logits[:5])
# print(y_pred_probs[:5])
# print(torch.argmax(y_pred_probs, axis=1))

# 5. Create a training & testing loop
epochs = 100

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()
        
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_func(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        
        test_loss = loss_func(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)
        
    if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
# 6. Making & evaluating predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = y_pred_probs.argmax(dim=1)

print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracyL {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_4, X_blob_train, y_blob_train)

plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()

# 7. Evaluation metrics
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)

print(torchmetrics_accuracy(y_preds, y_blob_test))
