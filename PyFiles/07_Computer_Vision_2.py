# Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader 

import torchvision
from torchvision import datasets, transforms

from torchmetrics import Accuracy, ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

import requests, random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

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

from helper_functions import accuracy_fn

# -------------------------------------------------------------------------------------------------------------------------- #
from timeit import default_timer as timer 

def print_train_time(start:float, end:float, device:torch.device=None):
	"""Prints difference between start and end time"""
	total_time = end - start
	print(f"Train time on {device}: {total_time:.3f} seconds")
	return round(total_time, 4)

# -------------------------------------------------------------------------------------------------------------------------- #
# Training & testing(evaluation) loops function
def train_step(model: torch.nn.Module,
			   data_loader:torch.utils.data.DataLoader,
			   loss_func:torch.nn.Module,
			   optimizer:torch.optim.Optimizer,
			   accuracy_fn,
			   device:torch.device=device):
	"""Performs a training with model trying to learn on data_loader"""

	train_loss, train_acc = 0, 0
	model.train()

	for batch, (X, y) in enumerate(data_loader):
		X, y = X.to(device), y.to(device)

		y_pred = model(X)
		loss = loss_func(y_pred, y)
		train_loss += loss
		acc = accuracy_fn(y, y_pred.argmax(dim=1))
		train_acc += acc

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	train_loss /= len(data_loader)
	train_acc /= len(data_loader)

	print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
			  data_loader: torch.utils.data.DataLoader,
			  loss_func: torch.nn.Module,
			  accuracy_fn,
			  device: torch.device=device):
	"""Performs a testing loop step on model going over data_loader."""
	test_loss, test_acc = 0, 0
	model.eval()

	with torch.inference_mode():
		for X, y in data_loader:
			X, y = X.to(device), y.to(device)

			test_pred = model(X)

			loss = loss_func(test_pred, y)
			test_loss += loss
			acc = accuracy_fn(y, test_pred.argmax(dim=1))
			test_acc += acc
		
		test_loss /= len(data_loader)
		test_acc /= len(data_loader)
	
	print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")

# -------------------------------------------------------------------------------------------------------------------------- #
# Evaluation function
def eval_model(model:torch.nn.Module,
			   data_loader:torch.utils.data.DataLoader,
			   loss_func:torch.nn.Module,
			   accuracy_fn,
			   device:torch.device=device):
	"""Evaluates a given model ona given dataset.

		Args:
			model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
            data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
			loss_func (torch.nn.Module): The loss function of model.
			accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
			device (str, optional): Target device to compute on. Default to device.

		Returns:
			(dict): Results of model making predictions on data_loader.
	"""
	loss, acc = 0, 0

	model.to(device)
	model.eval()
	with torch.inference_mode():
		for X, y in tqdm(data_loader):
			X, y = X.to(device), y.to(device)

			y_pred = model(X)

			loss += loss_func(y_pred, y)
			acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
		
		loss /= len(data_loader)
		acc /= len(data_loader)
	
	return {"model_name":model.__class__.__name__,
		 "model_loss":loss.item(),
		 "model_acc":acc}

# -------------------------------------------------------------------------------------------------------------------------- #
# Make prediction function
def make_predictions(model:torch.nn.Module, data:list, device:torch.device=device):
	pred_probs = []
	model.eval()
	with torch.inference_mode():
		for sample in data:
			sample = torch.unsqueeze(sample, dim=0).to(device)

			pred_logit = model(sample)
			pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)
			pred_probs.append(pred_prob.cpu())
	
	return torch.stack(pred_probs)

# -------------------------------------------------------------------------------------------------------------------------- #
# 1. Getting a dataset (MNIST)
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())

class_names = train_data.classes

# print(train_data, test_data)
# print(len(train_data), len(test_data))

# img = train_data[0][0]
# label = train_data[0][1]
# print(f"Image:\n{img}")
# print(f"Label:\n{label}")

# # Check out the shapes of data
# print(f"Image shape: {img.shape} -> [color_channels, height, width]")
# print(f"Label: {label} -> no shape, due to being integer")

# print(class_names)

# # Visualize some samples
# for i in range(4):
# 	img = train_data[i][0].squeeze()
# 	label = train_data[i][1]
# 	plt.figure(figsize=(3,3))
# 	plt.imshow(img, cmap='gray')
# 	plt.title(label)
# 	plt.axis(False)
# 	plt.show()

# Turn datasets into dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# for sample in next(iter(train_loader)):
# 	print(sample.shape)

# print(len(train_loader), len(test_loader))

# -------------------------------------------------------------------------------------------------------------------------- #
# 2. Create a model
class MNIST_Model(torch.nn.Module):
	"""Model capable of predicting on MNIST dataset.
	"""
	def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
		super().__init__()
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(in_channels=input_shape,
			 out_channels=hidden_units,
			 kernel_size=3,
			 stride=1,
			 padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=hidden_units,
			 out_channels=hidden_units,
			 kernel_size=3,
			 stride=1,
			 padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)

		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(in_channels=hidden_units,
			 out_channels=hidden_units,
			 kernel_size=3,
			 stride=1,
			 padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=hidden_units,
			 out_channels=hidden_units,
			 kernel_size=3,
			 stride=1,
			 padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=hidden_units*7*7,
			 out_features=output_shape)
		)
	
	def forward(self, x):
		# print(f"Input shape: {x.shape}")
		x = self.conv_block_1(x)
		# print(f"Output shape of conv block 1: {x.shape}")
		x = self.conv_block_2(x)
		# print(f"Output shape of conv block 2: {x.shape}")
		x = self.classifier(x)
		# print(f"Output shape of classifier: {x.shape}")
		return x
	
model = MNIST_Model(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
# print(model)
# print(model.state_dict())

# # Try a dummy forward pass to see what shapes data is
# dummy_x = torch.rand(size=(1,28,28)).unsqueeze(dim=0).to(device)
# model(dummy_x)

# dummy_y = torch.rand(size=(1,10,7,7))
# print(dummy_y.shape)

# flatten_layer = nn.Flatten()
# print(flatten_layer(dummy_y).shape)

# -------------------------------------------------------------------------------------------------------------------------- #
# 3. Train the model built for 5 epochs on CPU & GPU and see how long it takes on each.

EPOCHS = 5

# Train on CPU
# model_cpu = MNIST_Model(input_shape=1, hidden_units=10, output_shape=len(class_names)).to("cpu")

# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model_cpu.parameters(), lr=0.1)

# start_time = timer()

# for epoch in tqdm(range(EPOCHS)):
# 	print(f"Epoch: {epoch}\n--------------------")
# 	train_step(model=model_cpu,
# 			data_loader=train_loader,
# 			loss_func=loss_func,
# 			optimizer=optimizer,
# 			accuracy_fn=accuracy_fn,
# 			device="cpu")

# 	test_step(model=model_cpu,
# 		   data_loader=test_loader,
# 		   loss_func=loss_func,
# 		   accuracy_fn=accuracy_fn,
# 		   device="cpu")

# end_time = timer()

# total_train_time_model_cpu = print_train_time(start=start_time, end=end_time, device="cpu")

# Train on GPU
model_gpu = MNIST_Model(input_shape=1, hidden_units=10, output_shape=len(class_names)).to("cuda")

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_gpu.parameters(), lr=0.1)

start_time = timer()

for epoch in tqdm(range(EPOCHS)):
	print(f"Epoch: {epoch}\n--------------------")
	train_step(model=model_gpu,
			data_loader=train_loader,
			loss_func=loss_func,
			optimizer=optimizer,
			accuracy_fn=accuracy_fn,
			device="cuda")

	test_step(model=model_gpu,
		   data_loader=test_loader,
		   loss_func=loss_func,
		   accuracy_fn=accuracy_fn,
		   device="cuda")

end_time = timer()

total_train_time_model_gpu = print_train_time(start=start_time, end=end_time, device="cuda")

# -------------------------------------------------------------------------------------------------------------------------- #
# 4. Make predictions using trained model and 
# visualize some of them comparing the prediction to the target label.

# plt.imshow(test_data[0][0].squeeze(), cmap='gray')
# plt.show()

# # Logits -> Prediction probabilities -> Prediction labels
# pred_logits = model_gpu(test_data[0][0].unsqueeze(dim=0).to(device))
# pred_probs = torch.softmax(pred_logits, dim=1)
# pred_label = torch.argmax(pred_probs, dim=1)

# print(pred_logits)
# print(pred_probs)
# print(pred_label)

# num_to_plot = 4
# for i in range(num_to_plot):
# 	img = test_data[i][0]
# 	label = test_data[i][1]

# 	pred_logits = model_gpu(img.unsqueeze(0).to(device))
# 	pred_probs = torch.softmax(pred_logits, dim=1)
# 	pred_label = torch.argmax(pred_probs, axis=1)

# 	plt.figure()
# 	plt.imshow(img.squeeze(), cmap='gray')
# 	plt.title(f"Truth: {label} | Pred: {pred_label.cpu().item()}")
# 	plt.axis(False)
# 	plt.show()

# -------------------------------------------------------------------------------------------------------------------------- #
# 5. Confusion Matrix comparing model's predictions to the truth labels.

pred_probs = make_predictions(model=model_gpu, data=[img for img, _ in test_data], device=device)
pred_labels = pred_probs.argmax(dim=1)

print(test_data.targets[:10], pred_labels[:10])

confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=pred_labels, target=test_data.targets)

fig, ax = plot_confusion_matrix(
	conf_mat=confmat_tensor.numpy(),
	class_names=class_names,
	figsize=(9,6)
)
plt.show()
