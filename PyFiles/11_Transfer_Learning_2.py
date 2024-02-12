import torch
import torchvision
from torch import nn 
from torchvision import transforms, datasets 
from torchinfo import summary 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from pathlib import Path

import data_setup, engine

# print(torch.__version__) # 1.13.1+cu117
# print(torchvision.__version__) # 0.14.1+cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device) # cuda

# ----------------------------------------------------------------------------------------------
# 1. Make predictions on the test dataset and plot a confusion matrix

# 1-1. Get data
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# 1-2. Prepare data
## Create a transforms pipeline
simple_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

## Create DataLoader's as well as a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir=train_dir,
	test_dir=test_dir,
	transform=simple_transform,
	batch_size=32
)

# ----------------------------------------------------------------------------------------------
# 1-3. Get and prepare a pretrained model
## Setup the model with pretrained weights and send it to the target device
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model_0 = torchvision.models.efficientnet_b0(weights=weights).to(device)

## Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model_0.features.parameters():
	param.requires_grad=False

## Set the random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

## Recreate the classifier layer and send it to the target device
model_0.classifier = torch.nn.Sequential(
	torch.nn.Dropout(p=0.2, inplace=True),
	torch.nn.Linear(in_features=1280, out_features=output_shape, bias=True)
).to(device)

# ----------------------------------------------------------------------------------------------
# 1-4. Train model
## Define loss func and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=.001)

## Set the random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Start the timer
from timeit import default_timer as timer 
start_time = timer()

## Setup training and save the results
model_0_results = engine.train(model=model_0,
							   train_dataloader=train_dataloader,
							   test_dataloader=test_dataloader,
							   optimizer=optimizer,
							   loss_fn=loss_fn,
							   epochs=5,
							   device=device)

## End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds.")

# ----------------------------------------------------------------------------------------------
# 1-5. Make predictions on the entire test dataset with the model
# print(len(test_dataloader))

from tqdm.auto import tqdm

test_preds = []
model_0.eval()
with torch.inference_mode():
	for X, y in tqdm(test_dataloader):
		X, y = X.to(device), y.to(device)

		pred_logits = model_0(X)
		pred_probs = torch.softmax(pred_logits, dim=1)
		pred_labels = torch.argmax(pred_probs, dim=1)
		test_preds.append(pred_labels)

test_preds = torch.cat(test_preds).cpu()
# print(test_preds)

## 1-6. Make confusion matrix with the test preds and the truth labels
test_truth = torch.cat([y for _, y in test_dataloader])
# print(test_truth)

# from torchmetrics import ConfusionMatrix
# from mlxtend.plotting import plot_confusion_matrix

# confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
# confmat_tensor = confmat(preds=test_preds, target=test_truth)

# fig, ax = plot_confusion_matrix(
# 	conf_mat=confmat_tensor.numpy(),
# 	class_names=class_names,
# 	figsize=(9,6)
# )
# plt.show()

# ----------------------------------------------------------------------------------------------
# 2. Get the "most wrong" of the predictions on the test dataset and plot the 5 "most wrong" images.

## Get all test data paths
from pathlib import Path
test_data_paths= list(Path(test_dir).glob("*/*.jpg"))
test_labels = [path.parent.stem for path in test_data_paths]

## Create a function to return a list of dictionaries with sample, label, prediction, pred_prob
def pred_and_store(test_paths, model, transform, class_names, device):
	test_pred_list = []
	for path in tqdm(test_paths):

		pred_dict = {}

		pred_dict["image_path"] = path

		class_name = path.parent.stem
		pred_dict["class_name"] = class_name

		from PIL import Image
		img = Image.open(path)
		transformed_image = transform(img).unsqueeze(0)
		model.eval()
		with torch.inference_mode():
			pred_logit = model(transformed_image.to(device))
			pred_prob = torch.softmax(pred_logit, dim=1)
			pred_label = torch.argmax(pred_prob, dim=1)
			pred_class = class_names[pred_label.cpu()]

			pred_dict["pred_prob"] = pred_prob.max().cpu().item()
			pred_dict["pred_class"] = pred_class
		
		pred_dict["correct"] = (class_name==pred_class)

		test_pred_list.append(pred_dict)
	
	return test_pred_list
# ----------------------------------------------------------------------------------------------
test_pred_dicts = pred_and_store(test_paths=test_data_paths,
								 model=model_0,
								 transform=simple_transform,
								 class_names=class_names,
								 device=device)

# print(test_pred_dicts[:5])
# ----------------------------------------------------------------------------------------------

## Turn the test_pred_dicts into a DataFrame
test_pred_df = pd.DataFrame(test_pred_dicts)

## Sort DataFrame by correct then by pred_prob
top_5_most_wrong = test_pred_df.sort_values(by=["correct", "pred_prob"], ascending=[True, False]).head()
# print(top_5_most_wrong)

## Plot the top 5 most wrong images
# for row in top_5_most_wrong.iterrows():
# 	row = row[1]
# 	image_path = row[0]
# 	true_label = row[1]
# 	pred_prob = row[2]
# 	pred_class = row[3]

# 	img = torchvision.io.read_image(str(image_path))
# 	plt.figure()
# 	plt.imshow(img.permute(1,2,0))
# 	plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.3f}")
# 	plt.axis(False)
# 	plt.show()

# ----------------------------------------------------------------------------------------------
# 3. Predict on your own image

## Make a funciton to pred and plot images
def pred_and_plot(image_path, model, transform, class_names, device=device):
	from PIL import Image
	image = Image.open(image_path)

	transformed_image = transform(image)

	model.eval()
	with torch.inference_mode():
		pred_logit = model(transformed_image.unsqueeze(0).to(device))
		pred_label = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1)
	
	plt.figure()
	plt.imshow(image)
	plt.title(f"Pred: {class_names[pred_label]}")
	plt.axis(False)
	plt.show()

# ----------------------------------------------------------------------------------------------
# pred_and_plot(image_path="pizza.jpg", model=model_0, transform=simple_transform,
# 			  class_names=class_names)

# pred_and_plot(image_path="steak.jpg", model=model_0, transform=simple_transform,
# 			  class_names=class_names)

# pred_and_plot(image_path="apple.jpg", model=model_0, transform=simple_transform,
# 			  class_names=class_names)

# ----------------------------------------------------------------------------------------------
# 4. Train the model for longer (10 epochs), what happens to the performance?

## Recreate a new model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model_1 = torchvision.models.efficientnet_b0(weights=weights).to(device)

## Freeze the base layers
for param in model_1.features.parameters():
	param.requires_grad=False

## Change the classification head
model_1.classifier = torch.nn.Sequential(
	nn.Dropout(p=.2, inplace=True),
	nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
).to(device)

# print(summary(model_1, input_size=[32,3,224,224], col_names=["input_size", "output_size", "trainable"]))

## Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Start the timer
start_time = timer()

## Create loss & optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=.001)

## Setup training and save the results
model_1_results = engine.train(model=model_1,
							   train_dataloader=train_dataloader,
							   test_dataloader=test_dataloader,
							   optimizer=optimizer,
							   loss_fn=loss_fn,
							   epochs=10,
							   device=device)

## End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds.")

## Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it
try:
	from helper_functions import plot_loss_curves
except:
	print(f"[INFO] Couldn't find helper_functions.py, downloading...")
	with open("helper_functions.py", "wb") as f:
		import requests
		request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
		f.write(request.content)
	from helper_functions import plot_loss_curves

## Plot the loss curves of our model
# plot_loss_curves(model_1_results)
# plt.show()

# ----------------------------------------------------------------------------------------------
# 5. Train the model with more data, say 20% of the images from Food101 of Pizza, Steak, and Sushi images.

## Get 20% data
import os, requests, zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi_20_percent"
image_data_zip_path = "pizza_steak_sushi_20_percent.zip"

if image_path.is_dir():
	print(f"{image_path} directory exists.")
else:
	print(f"Did not find {image_path} directory, creating one...")
	image_path.mkdir(parents=True, exist_ok=True)

	with open(data_path / image_data_zip_path, "wb") as f:
		request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
		print("Downloading pizza, steak, sushi data...")
		f.write(request.content)
	
	with zipfile.ZipFile(data_path / image_data_zip_path, "r") as zip_ref:
		print("Unzipping pizza, steak,sushi 20% data...")
		zip_ref.extractall(image_path)
	
	os.remove(data_path / image_data_zip_path)

train_dir_20_percent = image_path / "train"
test_dir_20_percent = image_path / "test"

## Create DataLoaders
simple_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataloader_20, test_dataloader_20, class_names = data_setup.create_dataloaders(
	train_dir=train_dir_20_percent,
	test_dir=test_dir_20_percent,
	transform=simple_transform,
	batch_size=32)

print(len(train_dataloader_20), len(test_dataloader_20))

## Get a pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model_2 = torchvision.models.efficientnet_b0(weights=weights).to(device)

for param in model_2.features.parameters():
	param.requires_grad=False

model_2.classifier = nn.Sequential(
	nn.Dropout(p=.2, inplace=True),
	nn.Linear(in_features=1280, out_features=3, bias=True)
).to(device)

## Train a model with 20% of the data
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(), lr=.001)

start_time = timer()

model_2_results = engine.train(
	model=model_2,
	train_dataloader=train_dataloader_20,
	test_dataloader=test_dataloader_20,
	optimizer=optimizer,
	loss_fn=loss_fn,
	epochs=5,
	device=device
)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds.")

# print(max(model_0_results["test_acc"]), min(model_0_results["test_loss"]))
# print(max(model_2_results["test_acc"]), min(model_2_results["test_loss"]))

# ----------------------------------------------------------------------------------------------
# 6. Try a different model from torchvision.models

from torchvision import transforms, models

effnet_b2_transform = transforms.Compose([
	transforms.Resize((288,288)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# effnet_b2 takes images of size 288, 288 - https://github.com/pytorch/vision/blob/d2bfd639e46e1c5dc3c177f889dc7750c8d137c7/references/classification/train.py#L92-L93

train_dataloader_effnet_b2, test_dataloader_effnet_b2, class_names = data_setup.create_dataloaders(
	train_dir=train_dir,
	test_dir=test_dir,
	transform=effnet_b2_transform,
	batch_size=32
)

# Create a effnet_b2 new model - https://pytorch.org/vision/stable/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
model_3 = torchvision.models.efficientnet_b2(weights=weights).to(device)

for param in model_3.parameters():
	param.requires_grad=False

model_3.classifier = nn.Sequential(
	nn.Dropout(p=.3, inplace=True),
	nn.Linear(in_features=1408, out_features=len(class_names), bias=True)
).to(device)

print(summary(model_3, input_size=[32,3,288,288], col_names=["input_size","output_size","trainable"]))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_3.parameters(), lr=.001)

start_time = timer()

model_3_results = engine.train(
	model=model_3,
	train_dataloader=train_dataloader_effnet_b2,
	test_dataloader=test_dataloader_effnet_b2,
	optimizer=optimizer,
	loss_fn=loss_fn,
	epochs=5,
	device=device
)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds.")

# ----------------------------------------------------------------------------------------------
"""Which one did the best?
Experiments:
- model_0 : 10% data, effnet_b0, 5 epochs
- model_1 : 10% data, effnet_b0, 10 epochs (double training time)
- model_2 : 20% data, effnet_b0, 5 epochs (double data)
- model_3 : 10% data, effnet_b2, 5 epochs (double model parameters)
"""

# Check effnet_b0 results with 10% of data for 5 epochs
print(max(model_0_results["test_acc"]), min(model_0_results["test_loss"]))

# Check effnet_b0 results with 10% of data for 10 epochs (double training time)
print(max(model_1_results["test_acc"]), min(model_1_results["test_loss"]))

# Check effnet_b0 results with 20% of data for 5 epochs (double data)
print(max(model_2_results["test_acc"]), min(model_2_results["test_loss"]))

# Check effnet_b2 results with 10% of data for 5 epochs (double model parameters)
print(max(model_3_results["test_acc"]), min(model_3_results["test_loss"]))

plot_loss_curves(model_3_results)
