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
data_path = Path("data/") 
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
	print(f"{image_path} directory exist.")
else:
	print(f"Did not find {image_path} directory, creating one...")
	image_path.mkdir(parents=True, exist_ok=True)

	# Download pizza, steak, sushi data
	with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
		request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
		print("Downloading pizza, steak, sushi data...")
		f.write(request.content)
	
	with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
		print("Unzipping pizza, steak, sushi data...")
		zip_ref.extractall(image_path)

# 2. Data preparation ----------------------------------------------------------------------------
def walk_through_dir(dir_path):
	"""
	Walks through dir_path returning its contents.
	Args:
		dir_path (str or pathlib.Path): target directory
	
	Returns:
		A print out of:
			number of subdirectories in dir_path
			number of images (files) in each subdirectory
			name of each subdirectory
	"""
	for dirpath, dirnames, filenames in os.walk(dir_path):
		print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}.")

# walk_through_dir(image_path)

# ---------------------------------------------------------------------------------------------
# Setup train & testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"
# print(train_dir, test_dir)

# Visualize an image
# 1) Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# # 2) Get random image path
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

data_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.TrivialAugmentWide(num_magnitude_bins=31),
	transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
	"""Plots a series of random images from image_paths.
	
	Will open n image paths from image_paths, transform them
	with transform and plot them side by side.
	
	Args:
		image_paths (list): list of target image paths.
		transform (PyTorch Transforms): Transforms to apply to images.
		n (int, optional): Number of images to plot. Defaults=3.
		seed (int, optional): Random seed for the random generator. Defaults=42
	"""
	random.seed(seed)
	random_image_paths = random.sample(image_paths, k=n)
	for image_path in random_image_paths:
		with Image.open(image_path) as f:
			fig, ax = plt.subplots(1,2)
			ax[0].imshow(f)
			ax[0].set_title(f"Original\nSize: {f.size}")
			ax[0].axis('off')

			# Transform and plot image
			transformed_image = transform(f).permute(1,2,0)
			ax[1].imshow(transformed_image)
			ax[1].set_title(f'Transformed\nSize: {transformed_image.shape}')
			ax[1].axis('off')

			fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=14)
			plt.show()
			
# plot_transformed_images(image_path_list, transform=data_transform, n=3)
			
# 4-1. (Option1) Loading Image data using "ImageFolder" ----------------------------------------
train_data = datasets.ImageFolder(root=train_dir,
								  transform=data_transform,
								  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
								 transform=data_transform)

# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# Get class names as a list / as a dict
class_names = train_data.classes
class_dict = train_data.class_to_idx

# print(class_names)
# print(class_dict)

# print(len(train_data), len(test_data))

img, label = train_data[0][0], train_data[0][1]
# print(f"Image tensor:\n{img}")
# print(f"Image shape: {img.shape}")
# print(f"Image datatype: {img.dtype}")
# print(f"Image label: {label}")
# print(f"Label datatype: {type(label)}")

# plt.figure(figsize=(9,6))
# plt.imshow(img.permute(1,2,0))
# plt.axis(False)
# plt.title(class_names[label], fontsize=14)
# plt.show()

# Turn train & test Datasets into DataLoaders
train_loader = DataLoader(train_data, batch_size=1, num_workers=os.cpu_count(), shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, num_workers=os.cpu_count(), shuffle=False)

# 4-1. (Option2) Loading Image data with a "Custom Dataset" -------------------------------------

# function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
	"""Finds the class folder names in a target directory.
	
	Assumes target directory is in standard image classification format.
	
	Args:
		directory (str): target directory to load classnames from.
	
	Returns:
		Tuple[List[str], Dict[str, int]: (list_of_class_names, dict(class_name:idx))]
	
	Example:
		find_classes("food_images/train")
		>>> (["class_1", "class_2"], {"class_1":0, ...})
	"""
	# 1) Get the class names by scanning the target directory
	classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

	# 2) Raise an error if class names not found
	if not classes:
		raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
	
	# 3) Create a dictionary of index labels 
	class_to_idx = {cls_name : i for i, cls_name in enumerate(classes)}
	return classes, class_to_idx

# print(find_classes(train_dir))
# ------------------------------------------------------------------------------------------------
# Create a custom Dataset to replicate ImageFolder

class ImageFolderCustom(Dataset): # inherits from torch.utils.data.Dataset

	def __init__(self, targ_dir:str, transform=None) -> None:

		# Create class attributes
		self.paths = list(Path(targ_dir).glob("*/*.jpg"))
		self.transform=transform
		self.classes, self.class_to_idx = find_classes(targ_dir)
	
	# Make function to load images
	def load_image(self, index:int) -> Image.Image:
		"Open an image via a path and returns it."
		image_path = self.paths[index]
		return Image.open(image_path)
	
	# Overwrite the __len__() method
	def __len__(self) -> int:
		"Return the total number of samples."
		return len(self.paths)
	
	# Overwrite the __getitem__() method
	def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
		"Returns one sample of data: data and label (X, y)."
		img = self.load_image(index)
		class_name = self.paths[index].parent.name
		class_idx = self.class_to_idx[class_name]

		if self.transform:
			return self.transform(img), class_idx
		else:
			return img, class_idx

# ------------------------------------------------------------------------------------------------

# transform (augment)
train_transforms = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.RandomHorizontalFlip(p=.5),
	transforms.ToTensor()
])

test_transforms = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.ToTensor()
])

# Turn into Dataset 
train_data_custom = ImageFolderCustom(targ_dir=train_dir,
									  transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir,
									 transform=test_transforms)

# print(len(train_data_custom), len(test_data_custom))
# print(train_data_custom.classes, train_data_custom.class_to_idx)

# Turn train & test Datasets into DataLoaders
train_loader_custom = DataLoader(train_data_custom, batch_size=1, num_workers=os.cpu_count(), shuffle=True)
test_loader_custom = DataLoader(test_data_custom, batch_size=1, num_workers=os.cpu_count(), shuffle=False)

# ------------------------------------------------------------------------------------------------
# Function to display random images
def display_random_images(dataset:torch.utils.data.dataset.Dataset,
						  classes: List[str]=None,
						  n:int=10,
						  display_shape:bool=True,
						  seed:int=None):
	
	# Adjust display if n too high
	if n > 10:
		n = 10
		display_shape=False
		print(f"For display purpose, n shouldn't be larger than 10, setting to 10 and removing shape display.")
	
	# Set random seed
	if seed:
		random.seed(seed)
	
	# Get random sample indices
	random_sample_idx = random.sample(range(len(dataset)), k=n)

	# Setup plot
	plt.figure(figsize=(16,9))

	# Loop through samples and display it
	for i, targ_idx in enumerate(random_sample_idx):
		targ_image, targ_label = dataset[targ_idx][0], dataset[targ_idx][1]

		# Adjust image tensor shape for plotting
		targ_image_adjust = targ_image.permute(1,2,0)

		# Plot
		plt.subplot(1,n,i+1)
		plt.imshow(targ_image_adjust)
		plt.axis('off')
		if classes:
			title = f"class: {classes[targ_label]}"
			if display_shape:
				title = title + f"\nshape: {targ_image_adjust.shape}"
		
		plt.title(title)
	plt.show()

# display_random_images(train_data, n=5, classes=class_names, seed=None)
# display_random_images(train_data_custom, n=20, classes=class_names, seed=None)

# 5. Model 0 : TinyVGG without data augmentation ------------------------------------------------

# Creating transforms and loading data for Model 0
simple_transform = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.ToTensor()
])

train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

BATCH_SIZE=32
NUM_WORERS = os.cpu_count()

train_loader_simple = DataLoader(train_data_simple,
								 batch_size=BATCH_SIZE,
								 shuffle=True,)
								#  num_workers=NUM_WORERS)
test_loader_simple = DataLoader(test_data_simple, 
								batch_size=BATCH_SIZE,
								shuffle=False,)
								# num_workers=NUM_WORERS)

# Create TinyVGG model class
class TinyVGG(nn.Module):
	"""
	Model architecture copying TinyVGG from:
	https://poloclub.github.io/cnn-explainer/
	"""
	def __init__(self, input_shape:int, hidden_units:int, output_shape:int) -> None:
		super().__init__()
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(in_channels=input_shape,
			 out_channels=hidden_units,
			 kernel_size=3,
			 stride=1,
			 padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
			nn.ReLU(),
			nn.Conv2d(in_channels=hidden_units,
			 out_channels=hidden_units,
			 kernel_size=3,
			 stride=1,
			 padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2) # default stride value is same as kernel_size
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
			nn.Linear(in_features=hidden_units*16*16,
			 out_features=output_shape)
		)
	
	def forward(self, x):
		# print(x.shape)
		# x = self.conv_block_1(x)
		# print(x.shape)
		# x = self.conv_block_2(x)
		# print(x.shape)
		# x = self.classifier(x)
		# print(x.shape)
		# return x
		return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)

# print(model_0)

# Try a forward pass on a single image (to test the model)
# img_batch, label_batch = next(iter(train_loader_simple))

# img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
# print(f"Single image shape: {img_single.shape}\n")

# model_0.eval()
# with torch.inference_mode():
# 	pred = model_0(img_single.to(device))

# print(f"Output logits:\n{pred}\n")
# print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
# print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
# print(f"Actual label:\n{label_single}")

# Use torchinfo to get an idea of the shapes going through the model
# print(summary(model_0, input_size=[1,3,64,64]))

# ---------------------------------------------------------------------------------------------
# Train & test loop functions
def train_step(model: torch.nn.Module,
			   dataloader:torch.utils.data.DataLoader,
			   loss_func:torch.nn.Module,
			   optimizer:torch.optim.Optimizer):
	
	model.train()

	train_loss, train_acc = 0, 0

	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		y_pred = model(X)
		loss = loss_func(y_pred, y)
		train_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		y_pred_class = torch.argmax(torch.softmax(y_pred, axis=1), axis=1)
		train_acc += ( (y_pred_class == y).sum().item() / len(y_pred) )
	
	train_loss /= len(dataloader)
	train_acc /= len(dataloader)
	return train_loss, train_acc

def test_step(model:torch.nn.Module,
			  dataloader:torch.utils.data.DataLoader,
			  loss_func:torch.nn.Module):
	
	model.eval()

	test_loss, test_acc = 0, 0

	with torch.inference_mode():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)

			test_pred_logits = model(X)
			loss = loss_func(test_pred_logits, y)
			test_loss += loss.item()

			test_pred_labels = torch.softmax(test_pred_logits, dim=1).argmax(dim=1)
			test_acc += ( (test_pred_labels==y).sum().item() / len(test_pred_labels) )

	test_loss /= len(dataloader)
	test_acc /= len(dataloader)
	return test_loss, test_acc

def train(model:torch.nn.Module,
		  train_loader:torch.utils.data.DataLoader,
		  test_loader:torch.utils.data.DataLoader,
		  optimizer:torch.optim.Optimizer,
		  loss_func:torch.nn.Module=nn.CrossEntropyLoss(),
		  epochs:int=5):
	
	results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}

	for epoch in tqdm(range(epochs)):
		train_loss, train_acc = train_step(model=model,
									 dataloader=train_loader,
									 loss_func=loss_func,
									 optimizer=optimizer)
		
		test_loss, test_acc = test_step(model=model,
								  dataloader=test_loader,
								  loss_func=loss_func)
		
		print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["test_loss"].append(test_loss)
		results["test_acc"].append(test_acc)
	
	return results

# ---------------------------------------------------------------------------------------------
# Train & Evaluate Model 0

NUM_EPOCHS = 5

model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data_simple.classes)).to(device)

loss_func=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=.01)

start_time = timer()

model_0_results = train(model=model_0,
						train_loader=train_loader_simple,
						test_loader=test_loader_simple,
						optimizer=optimizer,
						loss_func=loss_func,
						epochs=NUM_EPOCHS)

end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")

# ---------------------------------------------------------------------------------------------
# Plot the loss curves
def plot_loss_curves(results: Dict[str, List[float]]):
	"""
	Ploats training curves of a reuslts dictionary.

	Args:
		results (dict): dictionary containing list of values, e.g.
		{"train_loss":[...],
		"train_acc":[...],
		"test_loss":[...].
		"test_acc":[...]}
	"""

	loss = results["train_loss"]
	test_loss = results["test_loss"]

	accuracy = results["train_acc"]
	test_accuracy = results["test_acc"]

	epochs = range(len(results["train_loss"]))

	plt.figure(figsize=(9,4))

	plt.subplot(1,2,1)
	plt.plot(epochs, loss, label='Train')
	plt.plot(epochs, test_loss, label='Test')
	plt.title('Loss')
	plt.xlabel('Epochs')
	plt.legend()

	plt.subplot(1,2,2)
	plt.plot(epochs, accuracy, label='Train')
	plt.plot(epochs, test_accuracy, label='Test')
	plt.title('Accuracy')
	plt.xlabel('Epochs')
	plt.legend()

	plt.show()

# plot_loss_curves(model_0_results)

# ---------------------------------------------------------------------------------------------
# Model 1: TinyVGG with Data Augmentation

# Create transform with data augmentation
train_transform_trivial_augment = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.TrivialAugmentWide(num_magnitude_bins=31),
	transforms.ToTensor()
])
test_transform_trivial_augment = transforms.Compose([
	transforms.Resize((64,64)),
	transforms.ToTensor()
])

# Create Datasets & DataLoaders
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_augmented = datasets.ImageFolder(test_dir, transform=test_transform_trivial_augment)

train_loader_augmented = DataLoader(train_data_augmented,
									batch_size=BATCH_SIZE, shuffle=True) # num_workers=os.cpu_counts()
test_loader_augmented = DataLoader(test_data_augmented, batch_size=BATCH_SIZE, shuffle=False)

# Construct and train Model 1
model_1 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data_augmented.classes)).to(device)

NUM_EPOCHS = 5
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=.01)

start_time = timer()

model_1_results = train(model=model_1,
						train_loader=train_loader_augmented,
						test_loader=test_loader_augmented,
						optimizer=optimizer,
						loss_func=loss_func,
						epochs=NUM_EPOCHS)

end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")

# plot_loss_curves(model_1_results)

# ---------------------------------------------------------------------------------------------
# Compare model results
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

# plt.figure(figsize=(14,9))

# epochs = range(len(model_0_df))

# plt.subplot(2,2,1)
# plt.plot(epochs, model_0_df["train_loss"], label='Model 0')
# plt.plot(epochs, model_1_df["train_loss"], label='Model 1')
# plt.title('Train Loss')
# plt.xlabel('Epochs')
# plt.legend()

# plt.subplot(2,2,2)
# plt.plot(epochs, model_0_df["test_loss"], label='Model 0')
# plt.plot(epochs, model_1_df["test_loss"], label='Model 1')
# plt.title('Test Loss')
# plt.xlabel('Epochs')
# plt.legend()

# plt.subplot(2,2,3)
# plt.plot(epochs, model_0_df["train_acc"], label='Model 0')
# plt.plot(epochs, model_1_df["train_acc"], label='Model 1')
# plt.title('Train Accuracy')
# plt.xlabel('Epochs')
# plt.legend()

# plt.subplot(2,2,4)
# plt.plot(epochs, model_0_df["test_acc"], label='Model 0')
# plt.plot(epochs, model_1_df["test_acc"], label='Model 1')
# plt.title('Test Accuracy')
# plt.xlabel('Epochs')
# plt.legend()

# plt.show()

# ---------------------------------------------------------------------------------------------
# Make a prediction on a custom image
# Load an image and preprocess it in a way that matches the type of data model was trained on.
# We'll have to convert own custom image to a tensor and make sure it's in the right datatype.

custom_image_path = data_path / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
	with open(custom_image_path, "wb") as f:
		# When downloading from GitHub, need to use the "raw" file link
		request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
		print(f"Downloading {custom_image_path}...")
		f.write(request.content)
else:
	print(f"{custom_image_path} already exists, skipping download.")

# ---------------------------------------------------------------------------------------------
def pred_and_plot_image(model:torch.nn.Module,
						image_path:str,
						class_names:List[str]=None,
						transform=None,
						device:torch.device=device):
	"""Make a prediction on a target image and plots the image with its prediction."""

	# 1. Load in image and convert the tensor values to float32
	target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

	# 2. Divide the image pixel values by 255 to get them between [0,1]
	target_image /= 255.

	# 3. Transform if necessary
	if transform:
		target_image = transform(target_image)
	
	# 4. Make sure the model is on the target device
	model.to(device)

	# 5. Turn on model evaluation mode and inference mode
	model.eval()
	with torch.inference_mode():
		# Add an extra dimension to the image
		target_image = target_image.unsqueeze(dim=0)

		# Make a prediction on image with an extra dimension and send it to the target device
		pred = model(target_image.to(device))

	# 6. Convert logits -> prediction probabilities 
	pred_probs = torch.softmax(pred, dim=1)

	# 7. Convert prediction probabilities -> prediction labels
	pred_label = torch.argmax(pred_probs, dim=1)

	# 8. Plot the image alongside the prediction and prediction probability
	plt.imshow(target_image.squeeze().permute(1,2,0))
	if class_names:
		title = f"Pred: {class_names[pred_label.cpu()]} | Prob: {pred_probs.max().cpu():.3f}"
	else:
		title = f"Pred: {pred_label} | Prob: {pred_probs.max().cpu():.3f}"
	plt.title(title)
	plt.axis(False)
	plt.show()

# ---------------------------------------------------------------------------------------------
custom_image_transform = transforms.Compose([
	transforms.Resize((64,64))
])

pred_and_plot_image(model=model_1,
					image_path=custom_image_path,
					class_names=class_names,
					transform=custom_image_transform,
					device=device)
# ---------------------------------------------------------------------------------------------
