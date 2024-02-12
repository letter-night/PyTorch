import torch
import torchvision
from torch import nn 
from torchvision import transforms
from torchinfo import summary 

import matplotlib.pyplot as plt
from pathlib import Path

import data_setup, engine

# print(torch.__version__) # 1.13.1+cu117
# print(torchvision.__version__) # 0.14.1+cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device) # cuda

# 1. Get data
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# --------------------------------------------------------------------------------
# 2. Create Datasets and DataLoaders
"""
It's important that custom data going into the model is prepared in the same way
as the original training data that went into the model.
"""
## manual creation 
## 1) transform for torchvision.models
"""
All pre-trained models expect input images normalized in the same way.
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
where H and W are expected to be at least 224.
The images have to be loaded in to a range of [0, 1] and then
normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
"""

manual_transforms = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

## 2) create dataloaders and class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir = train_dir,
	test_dir = test_dir,
	transform=manual_transforms,
	batch_size=32
)
# print(train_dataloader, test_dataloader, class_names)

## auto creation ------------------------------------------------------------------
# Get a set of pretrained model weights
# .DEFAULT = best available weights from pretraining on ImageNet
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
# print(auto_transforms)

# create dataloaders and class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir=train_dir,
	test_dir=test_dir,
	transform=auto_transforms,
	batch_size=32
)

# print(train_dataloader, test_dataloader, class_names)

# ----------------------------------------------------------------------------------
# 3. Getting a pretrained model
## 1) Setup the model with pretrained weights and send it to the target device
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# print(model)

## 2) Getting a summary of our model
# summary1 = summary(model=model,
# 				  input_size=(32,3,224,224),
# 				  col_names=["input_size", "output_size", "num_params", "trainable"],
# 				  col_width=20,
# 				  row_settings=["var_names"])

# print(summary1)

## 3) Freezing the base model and changing the output layer to suit our needs
# Freeze all base layers in the "features" section of the model (the feature exractor) by setting requires_grad=False
for param in model.features.parameters():
	param.requires_grad=False 

# Set the manaul seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names 
output_shape = len(class_names)

# Recreate the classifier layer and send it to the target device
model.classifier = torch.nn.Sequential(
	torch.nn.Dropout(p=0.2, inplace=True),
	torch.nn.Linear(in_features=1280, out_features=output_shape, bias=True)
).to(device)

# Do a summary "after" freezing the features and changing the output classifier layer
# summary2 = summary(model=model,
# 				  input_size=(32,3,224,224),
# 				  col_names=["input_size", "output_size", "num_params", "trainable"],
# 				  col_width=20,
# 				  row_settings=["var_names"])
# print(summary2)

# ----------------------------------------------------------------------------------
# 4. Train model

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Setup training and save the results
results = engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
					   optimizer=optimizer, loss_fn=loss_fn, epochs=5, device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds.")

# ----------------------------------------------------------------------------------
# 5. Evalaute model by plotting loss curves
# Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it
try:
	from helper_functions import plot_loss_curves
except:
	print(f"[INFO] Couldn't find helper_functions.py, downloading...")
	with open("helper_functions.py", "wb") as f:
		import requests
		request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
		f.write(request.content)
	from helper_functions import plot_loss_curves

# Plot the loss curves of our model
# plot_loss_curves(results)
# plt.show()

# ----------------------------------------------------------------------------------
# 6. Make predictions on images from the test set
"""
For our model to make predictions on an image, the image has to be in same format 
as the images our model was trained on.
- Same shape, same datatype, same device, same transformations.
"""
from typing import List, Tuple
from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model:torch.nn.Module,
						image_path:str,
						class_names:List[str],
						image_size:Tuple[int,int]=(224,224),
						transform:torchvision.transforms=None,
						device:torch.device=device):
	
	# 2. Open image
	img = Image.open(image_path)

	# 3. Create transformation for image (if one doesn't exist)
	if transform is not None:
		image_transform=transform
	else:
		image_transform=transforms.Compose([
			transforms.Resize(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
		])
	
	### Predict on image ###
	# 4. Make sure the model is on the target device
	model.to(device)

	# 5. Turn on model evaluation mode and inference mode
	model.eval()
	with torch.inference_mode():
		# 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
		transformed_image = image_transform(img).unsqueeze(dim=0)

		# 7. Make a prediction on image with an extra dimension and send it to the target device
		target_image_pred = model(transformed_image.to(device))

		# 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
		target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

		# 9. Convert prediction probabilities -> prediction labels
		target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

		# 10. Plot image with predicted label and probability
		plt.figure()
		plt.imshow(img)
		plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
		plt.axis(False)
		plt.show()

# ----------------------------------------------------------------------------------
# Get a random list of image paths from test set
import random
# num_images_to_plot = 3
# test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
# test_image_path_sample = random.sample(population=test_image_path_list, k=num_images_to_plot)

# # Make predictions on and plot the images
# for image_path in test_image_path_sample:
# 	pred_and_plot_image(model=model,
# 					 image_path=image_path,
# 					 class_names=class_names,
# 					 #transform=weights.transforms(),  # optionally pass in a specified transform from our pretrained model weights
# 					 image_size=(224,224))

# ----------------------------------------------------------------------------------
## Making predictions on a custom image
# Download custom image
import requests

# Setup custom image path
custom_image_path = Path("data/04-pizza-dad.jpeg")

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
	with open(custom_image_path, "wb") as f:
		request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
		print(f"Downloading {custom_image_path}...")
		f.write(request.content)
else:
	print(f"{custom_image_path} already exist, skipping download.")

# Predict on custom image
pred_and_plot_image(model=model, image_path=custom_image_path, class_names=class_names)
