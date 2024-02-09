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
# 1. Getting a dataset
train_data = datasets.FashionMNIST(
	root='data',
	train=True,
	download=True,
	transform=torchvision.transforms.ToTensor(),
	target_transform=None
)

test_data = datasets.FashionMNIST(
	root='data',
	train=False,
	download=True,
	transform=torchvision.transforms.ToTensor(),
	target_transform=None
)

class_names = train_data.classes
class_to_idx = train_data.class_to_idx

# print(train_data, test_data)
# print(len(train_data), len(test_data))
# print(class_names)
# print(class_to_idx)
# print(train_data.targets)

# image, label = train_data[0]
# print(image.shape)

# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# fig = plt.figure(figsize=(6,6))
# rows, cols = 4, 4
# for i in range(1, rows*cols+1):
# 	random_idx = torch.randint(0, len(train_data), size=[1]).item()
# 	img, label = train_data[random_idx]
# 	fig.add_subplot(rows, cols, i)
# 	plt.imshow(img.squeeze(), cmap='gray')
# 	plt.title(class_names[label])
# 	plt.axis(False)

# plt.show()
# -------------------------------------------------------------------------------------------------------------------------- #

# 2. Prepare DataLoader
BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# print(f"Length of train_loader: {len(train_loader)} batches of {BATCH_SIZE}...")
# print(f"Length of test_loader: {len(test_loader)} batches of {BATCH_SIZE}...")

# -------------------------------------------------------------------------------------------------------------------------- #
# 3. Build a model
class FashionMNISTModelV2(nn.Module):
	"""
	Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
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
			 padding=1), # options = "valid" (no padding) or  "same" (output has same shape as input) or int for specific number
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
		# print(f"Input shape: {x.shape}\n")
		x = self.conv_block_1(x)
		# print(f"Output shape of conv_block_1: {x.shape}\n")
		x = self.conv_block_2(x)
		# print(f"Output shape of conv_block_2: {x.shape}\n")
		x = self.classifier(x)
		# print(f"Output shape of classifier: {x.shape}")
		return x 

# -------------------------------------------------------------------------------------------------------------------------- #
model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
# print(model_2)
# rand_image_tensor = torch.randn(size=(1,28,28))
# model_2(rand_image_tensor.unsqueeze(0).to(device))

# -------------------------------------------------------------------------------------------------------------------------- #
# Setup loss function & optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=.01)

# Training & testing
start_time = timer()

EPOCHS = 3
for epoch in tqdm(range(EPOCHS)):
	print(f"Epoch: {epoch}\n--------------------")
	train_step(model=model_2,
			data_loader=train_loader,
			loss_func=loss_func,
			optimizer=optimizer,
			accuracy_fn=accuracy_fn,
			device=device)

	test_step(model=model_2,
		   data_loader=test_loader,
		   loss_func=loss_func,
		   accuracy_fn=accuracy_fn,
		   device=device)

end_time = timer()

total_train_time_model_2 = print_train_time(start=start_time, end=end_time, device=device)

# -------------------------------------------------------------------------------------------------------------------------- #
# Evaluation
model_2_results = eval_model(
	model=model_2,
	data_loader=test_loader,
	loss_func=loss_func,
	accuracy_fn=accuracy_fn
)

# print(model_2_results)

# -------------------------------------------------------------------------------------------------------------------------- #
# Compare results across models
model_0_results = {"model_name":"FashionMNISTModelV0", "model_loss":0.476639, "model_acc":83.426518}
model_1_results = {"model_name":"FashionMNISTModelV1", "model_loss":0.674039, "model_acc":77.306310}
total_train_time_model_0 = 29.5142
total_train_time_model_1 = 30.5721

compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results["traing_time"] = [total_train_time_model_0, 
								  total_train_time_model_1,
								  total_train_time_model_2]

# print(compare_results)
# compare_results.set_index('model_name')["model_acc"].plot(kind="barh")
# plt.xlabel('accuracy (%)')
# plt.ylabel('model')
# plt.show()

# -------------------------------------------------------------------------------------------------------------------------- #
# make predictions and visualization
# test_samples = []
# test_labels = []

# for sample, label in random.sample(list(test_data), k=9):
# 	test_samples.append(sample)
# 	test_labels.append(label)

# print(f'Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})')

# pred_probs = make_predictions(model=model_2, data=test_samples)
# pred_classes = pred_probs.argmax(axis=1)

# plt.figure(figsize=(6,6))
# nrows=3
# ncols=3

# for i, sample in enumerate(test_samples):
# 	plt.subplot(nrows, ncols, i+1)

# 	plt.imshow(sample.squeeze(), cmap='gray')

# 	pred_label = class_names[pred_classes[i]]
# 	truth_label = class_names[test_labels[i]]

# 	title_text = f"Pred {pred_label} | Truth: {truth_label}"

# 	if pred_label == truth_label:
# 		plt.title(title_text, fontsize=9, c='g')
# 	else:
# 		plt.title(title_text, fontsize=9, c='r')
# 	plt.axis(False)
# plt.show()

# -------------------------------------------------------------------------------------------------------------------------- #
# Confusion Matrix

y_preds = []
model_2.eval()
with torch.inference_mode():
	for X, y in tqdm(test_loader, desc='Making predictions...'):
		X, y = X.to(device), y.to(device)

		y_logit = model_2(X)
		y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
		y_preds.append(y_pred.cpu())

y_preds_tensor = torch.cat(y_preds)
# print(y_preds_tensor[:10])
# print(len(y_preds_tensor))

# Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_preds_tensor, target=test_data.targets)

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
	conf_mat=confmat_tensor.numpy(),
	class_names=class_names,
	figsize=(9,4)
)
plt.show()

# -------------------------------------------------------------------------------------------------------------------------- #
# Save & load best performing model

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'computer_vision_model_2.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)

loaded_model = FashionMNISTModelV2(input_shape=1, output_shape=len(class_names), hidden_units=10)
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

# Evaluate loaded model
loaded_model_results = eval_model(
	model=loaded_model,
	data_loader=test_loader,
	loss_func=loss_func,
	accuracy_fn=accuracy_fn
)

print(loaded_model_results)

# Check to see if results are close to each other 
print(torch.isclose(torch.tensor(model_2_results["model_loss"]),
			  torch.tensor(loaded_model_results["model_loss"]), 
			  atol=1e-08, # absolute tolerance
			  rtol=0.0001)) # relative tolerance

