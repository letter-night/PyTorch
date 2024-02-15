# 0. Getting setup
import torch
import torchvision
from torch import nn 
from torchvision import transforms
from torchinfo import summary

import get_data, data_setup, engine, utils, model_builder

from pathlib import Path
import matplotlib.pyplot as plt
import random

from PIL import Image

# print(torch.__version__) # 1.13.1+cu117
# print(torchvision.__version__) # 0.14.1+cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# ----------------------------------------------------------------------------------------
# 1. Get data
# Download 20 percent training data 
data_20_percent_path = get_data.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
	destination="pizza_steak_sushi_20_percent"
)

# ----------------------------------------------------------------------------------------
# 2. Create Datasets and DataLoaders

# Setup directories
train_dir_20_percent = data_20_percent_path / "train"
test_dir_20_percent = data_20_percent_path / "test"

# ----------------------------------------------------------------------------------------
# 3. Create feature extractor models 

class_names = ["pizza", "steak", "sushi"]

# Create an EffNetV2_s feature extractor
effnetv2_s = model_builder.create_effnetv2_s(class_names=class_names, device=device)

# ----------------------------------------------------------------------------------------
# 4. Create experiments and setup training code
# [Experiment 2. Compare data augmentation]

# Create a data augmentation transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_aug_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.TrivialAugmentWide(),
	transforms.ToTensor(),
	normalize
])

# Create a non-data aug transform
no_data_aug_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	normalize
])

# Create dataloaders
BATCH_SIZE=16

# Create train dataloader *with* data augmentation
train_dataloader_20_percent_with_aug, test_dataloader_20_percent, class_names = data_setup.create_dataloaders(
	train_dir=train_dir_20_percent,
	test_dir=test_dir_20_percent,
	train_transform=data_aug_transform,
	test_transform=no_data_aug_transform,
	batch_size=BATCH_SIZE
)

# Create train dataloader *without* data augmentation
train_dataloader_20_percent_without_aug, test_dataloader_20_percent, class_names = data_setup.create_dataloaders(
	train_dir=train_dir_20_percent,
	test_dir=test_dir_20_percent,
	train_transform=no_data_aug_transform,
	test_transform=no_data_aug_transform,
	batch_size=BATCH_SIZE
)

# ----------------------------------------------------------------------------------------
# Create a helper function for viewing different DataLoader images
# Visualize different samples from both dataloaders (aug vs. no aug)
def view_dataloader_images(dataloader, n=10):
	if n > 10:
		print(f"Having n higher than 10 will create messy plots, lowering to 10.")
		n = 10
	imgs, labels = next(iter(dataloader))
	plt.figure(figsize=(9,4))
	for i in range(n):
		# Min Max scale the image for display purposes
		targ_image = imgs[i]
		sample_min, sample_max = targ_image.min(), targ_image.max()
		sample_scaled = (targ_image - sample_min) / (sample_max - sample_min)

		# Plot images with appropriate axes information
		plt.subplot(1,10,i+1)
		plt.imshow(sample_scaled.permute(1,2,0))
		plt.title(class_names[labels[i]])
		plt.axis(False)
	plt.tight_layout()
	plt.show()

# Check out samples with data augmentation
# view_dataloader_images(train_dataloader_20_percent_with_aug)

# Check out samples without data augmentation
# view_dataloader_images(train_dataloader_20_percent_without_aug)

# ----------------------------------------------------------------------------------------
# Run data aug vs. no data aug experiments
num_epochs=[5,10]
train_dataloaders = {"data_20_percent_with_aug":train_dataloader_20_percent_with_aug,
					 "data_20_percent_without_aug":train_dataloader_20_percent_without_aug}
models = ["effnetv2_s"]

# set_seeds(seed=42)

experiment_number = 0

for dataloader_name, train_dataloader in train_dataloaders.items():
	
	for epochs in num_epochs:

		for model_name in models:

			experiment_number += 1
			print(f"[INFO] Experiment Number: {experiment_number}")
			print(f"[INFO] Model name: {model_name}")
			print(f"[INFO] DataLoader: {dataloader_name}")
			print(f"[INFO] Number of epochs: {epochs}")

			if model_name == "effnetv2_s":
				model = model_builder.create_effnetv2_s(class_names=class_names, device=device)
			else:
				pass
			
			loss_fn = nn.CrossEntropyLoss()
			optimizer = torch.optim.Adam(params=model.parameters(), lr=.001)

			engine.train(
				model=model,
				train_dataloader=train_dataloader,
				test_dataloader=test_dataloader_20_percent,
				optimizer=optimizer,
				loss_fn=loss_fn,
				epochs=epochs,
				device=device,
				writer=engine.create_writer(experiment_name=dataloader_name,
								model_name=model_name,
								extra=f"{epochs}_epochs")
			)

			save_filepath = f"{model_name}_{dataloader_name}_{epochs}_epochs.pth"
			utils.save_model(model=model, target_dir="models", model_name=save_filepath)
			print("-"*49+"\n")

"""
Terminal: tensorboard --logdir=runs
or
Ctrl+Shift+P -> python tensorboard
"""