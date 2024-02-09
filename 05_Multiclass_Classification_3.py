import torch, requests
from torch import nn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchmetrics import Accuracy

# print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# torch.cuda.manual_seed(RANDOM_SEED)

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

# torchmetrics Accuracy function
acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)

# -------------------------------------------------------------------------------------------------------------------------- #

# Create a spiral dataset
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

# Create train & test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_SEED)
  
# print(len(X_train), len(X_test), len(y_train), len(y_test))

# Build a model
class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))
    
model_1 = SpiralModel().to(device)
# print(model_1)

# Setup data to be device agnostic
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Pring out untrained model outputs
# print("Logits:")
# print(model_1(X_train[:10]))

# print("Pred probs:")
# print(torch.softmax(model_1(X_train[:10]), dim=1))

# print("Pred labeles:")
# print(torch.softmax(model_1(X_train[:10]), dim=1).argmax(dim=1))

# Setup loss function & optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=.02)

# Build a training loop for the model
epochs = 1000

# Loop over data
for epoch in range(epochs):
    
    model_1.train()
    
    y_logits = model_1(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_func(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_func(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test)
        
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f}, Acc: {acc:.2f} | Test loss: {test_loss:.2f}, Test acc: {test_acc:.2f}")

# Plot decision boundaries for training & test sets
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_1, X_test, y_test)
plt.show()
