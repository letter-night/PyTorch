import torch, requests
from torch import nn
import pandas as pd
import numpy as np 
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

# 1. Make a binary classification dataset with Scikit-Learn's make_moons() function
from sklearn.datasets import make_moons

NUM_SAMPLES = 1000
RANDOM_SEED = 42

X, y = make_moons(n_samples=NUM_SAMPLES,
                  noise=0.07,
                  random_state=RANDOM_SEED)

# print(X[:10], y[:10])

# Turn it into a DataFrame
# data_df = pd.DataFrame({"X0":X[:,0], "X1":X[:,1], "y":y})
# print(data_df.head())

# Visualize the data on a plot
# plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Turn data into tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# Split the data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_SEED)

# print(len(X_train), len(X_test), len(y_train), len(y_test))

# 2. Build a model
class MoonModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        
        self.layer_1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_0 = MoonModel(in_features=2, out_features=1, hidden_units=10).to(device)

# print(model_0)
# print(model_0.state_dict())

# 3. loss function & optimizer
loss_func = nn.BCEWithLogitsLoss()
# loss_func = nn.BCELoss() requires sigmoid layer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=.1)

# 4. Create a training & testing loop to fit the model to the data

# Check with some sample data
# print("Logits:")
# print(model_0(X_train.to(device)[:10]).squeeze())

# print("Pred probs:")
# print(torch.sigmoid(model_0(X_train.to(device)[:10]).squeeze()))

# print("Pred labels:")
# print(torch.round(torch.sigmoid(model_0(X_train.to(device)[:10]).squeeze())))

# Accuracy function
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

# torch.manual_seed(RANDOM_SEED)

epochs = 2000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    
    model_0.train()
    
    y_logits = model_0(X_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(y_pred_probs)
    
    loss = loss_func(y_logits, y_train)
    acc = acc_fn(y_pred, y_train.int()) # the accuracy function needs to compare pred labels (not logits) with actual labels
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_func(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test.int())
        
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f}, Acc: {acc:.2f} | Test loss: {test_loss:.2f}, Test acc: {test_acc:.2f}")

# 5. Make predictions with trained model and plot them using the 'plot_decision_boundary()' function
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

