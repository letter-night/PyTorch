import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Using device: {device}")

weight = 0.7
bias = 0.3

start=0
end=1
step=.02

X = torch.arange(start, end, step).unsqueeze(1)
y = weight*X + bias

# print(X[:10])
# print()
# print(y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test,
					 predictions=None):
	plt.figure(figsize=(6,4))

	plt.scatter(train_data, train_labels, c='b', s=4, label='Train')
	plt.scatter(test_data, test_labels, c='g', s=4, label='Test')
	if predictions is not None:
		plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
	
	plt.legend(prop={'size':14})
	plt.show()

# plot_predictions()

class LinearRegressionModelV2(nn.Module):
	def __init__(self):
		super().__init__()

		self.linear_layer = nn.Linear(in_features=1, out_features=1)
	
	def forward(self, x):
		return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
# print(model_1)
# print()
# print(model_1.state_dict())

model_1.to(device)

loss_func = nn.L1Loss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=.01)

epochs = 1000

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
	model_1.train()

	y_pred = model_1(X_train)
	loss = loss_func(y_pred, y_train)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	model_1.eval()
	with torch.inference_mode():
		test_pred = model_1(X_test)
		test_loss = loss_func(test_pred, y_test)
	
	# if epoch % 100 == 0:
	# 	print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

# from pprint import pprint
# print(f"The model learned the following values for weights and bias:")
# pprint(model_1.state_dict())
# print("\nAnd the original values for weights and bias are:")
# print(f"weights: {weight}, bias: {bias}")

model_1.eval()

with torch.inference_mode():
	y_preds = model_1(X_test)

# plot_predictions(predictions=y_preds.cpu())
		
from pathlib import Path

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = '01_pytorch_workflow_model_1.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModelV2()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"\nModel on device:\n{next(loaded_model_1.parameters()).device}")

loaded_model_1.eval()
with torch.inference_mode():
	loaded_model_1_preds = loaded_model_1(X_test)

print(y_preds == loaded_model_1_preds)

