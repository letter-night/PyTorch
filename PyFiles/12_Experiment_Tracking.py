# 0. Getting setup
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn 
from torchvision import transforms
from torchinfo import summary

import get_data, data_setup, engine, utils 

# print(torch.__version__) # 1.13.1+cu117
# print(torchvision.__version__) # 0.14.1+cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# ----------------------------------------------------------------------------------------
# 1. Get data
# Download 10 percent and 20 percent training data 
data_10_percent_path = get_data.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
	destination="pizza_steak_sushi"
)

data_20_percent_path = get_data.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
	destination="pizza_steak_sushi_20_percent"
)

# ----------------------------------------------------------------------------------------
# 2. Create Datasets and DataLoaders

## 2-1. Create DataLoaders using manually created transforms

# Setup directories
train_dir_10_percent = data_10_percent_path / "train"
train_dir_20_percent = data_20_percent_path / "train"
test_dir = data_10_percent_path / "test"

# Create a transform to normalize data distribution to be inline with ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Compose transforms into a pipeline
simple_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	normalize
])

BATCH_SIZE = 32

# Create 10% training and test DataLoaders
train_dataloader_10_percent, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir=train_dir_10_percent,
	test_dir=test_dir,
	transform=simple_transform,
	batch_size=BATCH_SIZE
)

# Create 20% training and test DataLoaders
train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir=train_dir_20_percent,
	test_dir=test_dir,
	transform=simple_transform,
	batch_size=BATCH_SIZE
)

# Find the number of samples/batches per dataloader (using the same test_dataloader for both experiments)
# print(f"Number of batches of size {BATCH_SIZE} in 10 percent training data: {len(train_dataloader_10_percent)}")
# print(f"Number of batches of size {BATCH_SIZE} in 20 percent training data: {len(train_dataloader_20_percent)}")
# print(f"Number of batches of size {BATCH_SIZE} in testing data: {len(test_dataloader)} (all experiments will use the same test set)")
# print(f"Number of classes: {len(class_names)}, class names: {class_names}")

## 2-2. Create DataLoaders using automatically created transforms

# Setup directories
# train_dir = image_path / "train"
# test_dir = image_path / "test"

# # Setup pretrained weights 
# weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# # Get transforms from weights (these are the transforms that were used to obtain the weights)
# automatic_transforms = weights.transforms()
# # print(f"Automatically created transforms: {automatic_transforms}")

# # Create dataloaders
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
# 	train_dir=train_dir,
# 	test_dir=test_dir,
# 	transform=automatic_transforms,
# 	batch_size=32
# )

# print(train_dataloader, test_dataloader, class_names)

# ----------------------------------------------------------------------------------------
# 3. Create feature extractor models 
import torch
from torch import nn 

# Get num out features
OUT_FEATURES = len(class_names)

# Create an EffNetB0 feature extractor
def create_effnetb0():
	# 1. Get the base model with pretrained weights and send to target device
	weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
	model = torchvision.models.efficientnet_b0(weights=weights).to(device)

	# 2. Freeze the base model layers
	for param in model.features.parameters():
		param.requires_grad=False
	
	# 3. Set the seeds
	# set_seeds()
	
	# 4. Change the classifier head
	model.classifier = nn.Sequential(
		nn.Dropout(p=.2),
		nn.Linear(in_features=1280, out_features=OUT_FEATURES)
	).to(device)

	# 5. Give the model a name
	model.name = "effnetb0"
	print(f"[INFO] Created new {model.name} model.")

	return model

# Create an EffNetB2 feature extractor
def create_effnetb2():
	# 1. Get the base model with pretrained weights and send to target device
	weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
	model = torchvision.models.efficientnet_b2(weights=weights).to(device)

	# 2. Freeze the base model layers
	for param in model.features.parameters():
		param.requires_grad=False

	# 3. Set the seeds
	# set_seeds()
		
	# 4. Change the classifier head
	model.classifier = nn.Sequential(
		nn.Dropout(p=.3),
		nn.Linear(in_features=1408, out_features=OUT_FEATURES)
	).to(device)

	# 5. Give the model a name
	model.name = "effnetb2"
	print(f"[INFO] Created new {model.name} model.")

	return model

# ----------------------------------------------------------------------------------------
# effnetb0 = create_effnetb0()

# print(summary(
# 	model=effnetb0,
# 	input_size=(32,3,224,224),
# 	col_names=["input_size", "output_size", "num_params", "trainable"],
# 	col_width=20,
# 	row_settings=["var_names"]
# ))

# effnetb2 = create_effnetb2()

# print(summary(
# 	model=effnetb2,
# 	input_size=(32,3,224,224),
# 	col_names=["input_size", "output_size", "num_params", "trainable"],
# 	col_width=20,
# 	row_settings=["var_names"]
# ))

# ----------------------------------------------------------------------------------------
# 4. Create experiments and setup training code

# Create epochs list
num_epochs = [5,10]

# Create models list (need to create a new model for each experiment)
models = ["effnetb0", "effnetb2"]

# Create dataloaders dictionary for various dataloaders
train_dataloaders = {"data_10_percent":train_dataloader_10_percent,
					 "data_20_percent":train_dataloader_20_percent}

# 1) Set the random seeds
# set_seeds(seed=42)

# 2) Keep track of experiment numbers
experiment_number = 0

# 3) Loop through each DataLoader
for dataloader_name, train_dataloader in train_dataloaders.items():

	# 4) Loop through each number of epochs
	for epochs in num_epochs:

		# 5) Loop through each model name and create a new model based on the name
		for model_name in models:

			# 6) Create information print outs
			experiment_number += 1
			print(f"[INFO] Experiment number: {experiment_number}")
			print(f"[INFO] Model: {model_name}")
			print(f"[INFO] DataLoader: {dataloader_name}")
			print(f"[INFO] Number of epochs: {epochs}")

			# 7) Select the model
			# creates a new model each time 
			if model_name=="effnetb0":
				model = create_effnetb0()
			else:
				model = create_effnetb2()
			
			# 8) Create a new loss func and optimizer for every model
			loss_fn = nn.CrossEntropyLoss()
			optimizer = torch.optim.Adam(params=model.parameters(), lr=.001)

			# 9) Train target moel with target dataloaders and track experiments
			engine.train(model=model,
		 			train_dataloader=train_dataloader,
					test_dataloader=test_dataloader,
					optimizer=optimizer,
					loss_fn=loss_fn,
					epochs=epochs,
					device=device,
					writer=engine.create_writer(experiment_name=dataloader_name,
								 model_name=model_name,
								 extra=f"{epochs}_epochs"))

			# 10) Save the model to file so we can get back the best model
			save_filepath = f"{model_name}_{dataloader_name}_{epochs}_epochs.pth"
			utils.save_model(model=model, target_dir="models", model_name=save_filepath)
			print("-"*49 + "\n")

"""
Terminal: tensorboard --logdir=runs
or
Ctrl+Shift+P -> python tensorboard
"""

# ----------------------------------------------------------------------------------------
# 5. Load in the best model and make predictions with it

# Setup the best model filepath
best_model_path = "models/effnetb0_data_20_percent_10_epochs.pth"

# Instantiate a new instance of EffNetB2 (to load the saved state_dict() to)
best_model = create_effnetb0()

# Load the saved best model state_dict()
best_model.load_state_dict(torch.load(best_model_path))

# Import function to make predictions on images and plot them
from helper_functions import pred_and_plot_image

# get a random list of 3 images from 20% test set
from pathlib import Path
import random
num_images_to_plot=3
test_image_path_list = list(Path(data_20_percent_path / "test").glob("*/*.jpg"))
test_image_path_sample = random.sample(population=test_image_path_list,
									   k=num_images_to_plot)

from typing import Tuple, List
from PIL import Image

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

# Iterate through random test image paths, make predictions on them and plot them
for image_path in test_image_path_sample:
	pred_and_plot_image(model=best_model,
					 image_path=image_path,
					 class_names=class_names,
					 image_size=(224,224))
	plt.show()

# Predict on a custom image with the best model
# Download custom image
import requests

# Setup custom image path
custom_image_path = Path('data/04-pizza-dad.jpeg')

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# Predict on custom image
pred_and_plot_image(model=best_model,
                    image_path=custom_image_path,
                    class_names=class_names)
plt.show()