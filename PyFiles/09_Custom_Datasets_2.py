# 0. Importing libraries and setting up device-agnostic code
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset 

import torchvision
from torchvision import datasets, transforms

import torchinfo
from torchinfo import summary
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image 
import requests, zipfile, os, random
from typing import Tuple, Dict, List 
from timeit import default_timer as timer 


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# 1. Get data ------------------------------------------------------------------------------------
# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doen't exist, download it and prepare it...
if image_path.is_dir():
	print(f"{image_path} directory exist.")
else:
	print(f"Did not find {image_path} directory, creating...")
	image_path.mkdir(parents=True, exist_ok=True)

	# Download pizza, steak, sushi data (images from GitHub)
	with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
		request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
		print(f"Downloading pizza, steak, sushi data...")
		f.write(request.content)

	# Unzip data
		with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
			print(f"Unzipping pizza, steak, sushi data to {image_path}")
			zip_ref.extractall(image_path)
	
# 2. Data preparation ----------------------------------------------------------------------------
def walk_through_dir(dir_path):
	"""Walks through dir_path returing file counts of its contents."""
	for dirpath, dirnames, filenames in os.walk(dir_path):
		print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# print(walk_through_dir(image_path))

# ---------------------------------------------------------------------------------------------
# Setup train & testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# print(train_dir)
# print(test_dir)

# Visualize an image
# 1) Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))
# print(image_path_list[:4])

# 2) Get random image path
# random_image_path = random.choice(image_path_list)

# # 3) Get image class from path name
# image_class = random_image_path.parent.stem

# # 4) Open image
# img = Image.open(random_image_path)

# # 5) Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# plt.imshow(img)
# plt.show()

# 3. Transforming Data ----------------------------------------------------------------------------
data_transform = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.RandomHorizontalFlip(p=.5),
	transforms.ToTensor()
])

# def plot_transformed_images(image_paths, transform, n=3, seed=42):
# 	"""Plots a series of random images from image_paths."""
# 	random.seed(seed)
# 	random_image_paths = random.sample(image_paths, k=n)
# 	for image_path in random_image_paths:
# 		with Image.open(image_path) as f:
# 			fig, ax = plt.subplots(nrows=1, ncols=2)
# 			ax[0].imshow(f)
# 			ax[0].set_title(f'Original\nsize:{f.size}')
# 			ax[0].axis('off')

# 			# Transform and plot image
# 			transformed_image = transform(f).permute(1,2,0)
# 			ax[1].imshow(transformed_image)
# 			ax[1].set_title(f'Transformed\nsize:{transformed_image.shape}')
# 			ax[1].axis('off')

# 			fig.suptitle(f'Class: {image_path.parent.stem}', fontsize=14)
# 			plt.show()

# plot_transformed_images(image_path_list, transform=data_transform, n=3)
			
# 4. Loading Image data using "ImageFolder" ------------------------------------------------------
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)

class_names = train_data.classes
class_dict = train_data.class_to_idx

# print(class_names)
# print(class_dict)
# print(len(train_data), len(test_data))

BATCH_SIZE = 1
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# img, label = next(iter(train_loader))

# print(f"Image shape: {img.shape}")
# print(f"Label shape: {label.shape}")

# 5. Model 0 : TinyVGG without data augmentation ------------------------------------------------
class TinyVGG(nn.Module):
	def __init__(self, input_shape, hidden_units, output_shape):
		super().__init__()
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
		)
	
	def forward(self, x):
		return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)

# print(summary(model=model_0, input_size=[1,3,64,64]))

# Pass dummy data through model
# dummy_x = torch.rand(size=[1,3,64,64])
# print(model_0(dummy_x.to(device)))

# ---------------------------------------------------------------------------------------------
# Train & test loop functions
def train_step(model, dataloader, loss_func, optimizer):
	model.train()

	train_loss, train_acc = 0, 0

	for X, y in dataloader:
		X, y = X.to(device), y.to(device)

		y_pred = model(X)
		loss = loss_func(y_pred, y)
		train_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		y_pred_label = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
		train_acc += ( (y_pred_label == y).sum().item() / len(y_pred) )
	
	train_loss /= len(dataloader)
	train_acc /= len(dataloader)
	
	return train_loss, train_acc

def test_step(model, dataloader, loss_func):
	model.eval()

	test_loss, test_acc = 0, 0

	with torch.inference_mode():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)

			test_pred = model(X)
			loss = loss_func(test_pred, y)
			test_loss += loss.item()

			test_pred_label = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
			test_acc += ( (test_pred_label==y).sum().item() / len(test_pred) )
	
	test_loss /= len(dataloader)
	test_acc /= len(dataloader)
	
	return test_loss, test_acc 

def train(model, train_loader, test_loader, optimizer, loss_func, epochs=5):
	
	results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}

	for epoch in tqdm(range(epochs)):
		train_loss, train_acc = train_step(model=model, dataloader=train_loader, loss_func=loss_func, optimizer=optimizer)
		test_loss, test_acc = test_step(model=model, dataloader=test_loader, loss_func=loss_func)

		print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["test_loss"].append(test_loss)
		results["test_acc"].append(test_acc)

	return results 

# ---------------------------------------------------------------------------------------------
# Train & Evaluate Model 0~2
# model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_0.parameters(), lr=.01)

# # Train for 5 epochs
# model_0_results = train(model=model_0, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_func=loss_func)

# # Train for 20 epochs
# model_1 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
# optimizer = torch.optim.Adam(model_1.parameters(), lr=.01)

# model_1_results = train(model=model_1, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_func=loss_func, epochs=20)

# # Train for 50 epochs
# model_2 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
# optimizer = torch.optim.Adam(model_2.parameters(), lr=.01)

# model_2_results = train(model=model_2, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_func=loss_func, epochs=50)

# ---------------------------------------------------------------------------------------------
# Train & Evaluate Model 3
# Double the number of hidden units and train it for 20 epochs
# model_3 = TinyVGG(input_shape=3, hidden_units=20, output_shape=len(class_names)).to(device)
# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_3.parameters(), lr=.01)

# model_3.results = train(model=model_3, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_func=loss_func, epochs=20)

# ---------------------------------------------------------------------------------------------
# Train & Evaluate Model 4
# Double the data  and train it for 20 epochs
image_path = data_path / "pizza_steak_sushi_20_percent"

if image_path.is_dir():
	print(f"{image_path} directory exist.")
else:
	print(f"Did not find {image_path} directory, creating one...")
	image_path.mkdir(parents=True, exist_ok=True)

	with open(data_path / "pizza_steak_sushi_20_percent.zip", "wb") as f:
		request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
		print('Downloading pizza, steak, sushi 20% data...')
		f.write(request.content)
	
	with zipfile.ZipFile(data_path / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
		print("Unzipping data...")
		zip_ref.extractall(image_path)

# print(walk_through_dir(image_path))

train_data_20_path = image_path / "train"
test_data_20_path = image_path / "test"

simple_transform = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.ToTensor()
])

train_data_20 = datasets.ImageFolder(train_data_20_path, transform=simple_transform)
test_data_20 = datasets.ImageFolder(test_data_20_path, transform=simple_transform)

train_loader_20 = DataLoader(train_data_20, batch_size=32, shuffle=True)
test_loader_20 = DataLoader(test_data_20, batch_size=32, shuffle=False)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_4 = TinyVGG(input_shape=3, hidden_units=20, output_shape=len(class_names)).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_4.parameters(), lr=.001)

model_4_results = train(model=model_4, train_loader=train_loader_20, test_loader=test_loader_20, optimizer=optimizer, loss_func=loss_func, epochs=10)

# ---------------------------------------------------------------------------------------------
# Make a prediction on a custom image

# Get a custom image
custom_image = "pizza_dad.jpeg"
with open("pizza_dad.jpeg", "wb") as f:
	request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
	f.write(request.content)

# Load the image
img = torchvision.io.read_image(custom_image)
# print(img)

# View the image
# plt.figure(figsize=(9,6))
# plt.imshow(img.permute(1,2,0))
# plt.axis('off')
# plt.show()

# Make a prediction on the image
model_4.eval()
with torch.inference_mode():

	img = img / 255.

	resize = transforms.Resize((64,64))
	img = resize(img)

	batch = img.unsqueeze(0).to(device)

	y_pred_logit = model_4(batch)
	pred_label = torch.argmax(torch.softmax(y_pred_logit, dim=1), dim=1)

	plt.imshow(img.permute(1,2,0))
	plt.title(f"Pred label: {class_names[pred_label]}")
	plt.axis('off')
	plt.show()

