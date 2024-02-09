import torch, requests
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path 

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

# 1. Create data : make and plot data
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

# plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Convert to tensors and split into train & test sets
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# print(X_train[:4], y_train[:4])

# 2. Build a model 
class CircleModel(nn.Module):
	def __init__(self):
		super().__init__()

		self.layer_1 = nn.Linear(in_features=2, out_features=10)
		self.layer_2 = nn.Linear(in_features=10, out_features=10)
		self.layer_3 = nn.Linear(in_features=10, out_features=1)
		self.relu = nn.ReLU()
		# self.sigmoid = nn.Sigmoid() # This would mean you don't need to use it on the predictions
	
	def forward(self, x):
		x = self.layer_1(x)
		x = self.relu(x)
		x = self.layer_2(x)
		x = self.relu(x)
		return self.layer_3(x)

model_3 = CircleModel().to(device)
# print(model_3)

# Setup loss function & optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=.1)

# 3. Train a model
epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):

	model_3.train()

	y_logits = model_3(X_train).squeeze()
	y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> prediction probabilities -> prediction labels

	loss = loss_func(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
	acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	model_3.eval()
	with torch.inference_mode():
		test_logits = model_3(X_test).squeeze()
		test_pred = torch.round(torch.sigmoid(test_logits))

		test_loss = loss_func(test_logits, y_test)
		test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
	
	if epoch % 100 == 0:
		print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# 4. Evaluate trained model
# Make predictions
model_3.eval()
with torch.inference_mode():
	y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

# print(y_preds[:10], y_test[:10]) # want preds in same format as truth labels

# Plot decision boundaries for training and test sets
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_3, X_test, y_test)
plt.show()