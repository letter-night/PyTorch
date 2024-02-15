# 0. Getting setup
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn 
from torchvision import transforms
from torchinfo import summary

import get_data, data_setup, engine, utils 

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# print(torch.__version__) # 1.13.1+cu117
# print(torchvision.__version__) # 0.14.1+cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# ----------------------------------------------------------------------------------------
# 1. Get data
# Download Food101 Dataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

simple_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	normalize
])

train_data = torchvision.datasets.Food101(root="data", split="train", transform=simple_transform, download=True)
test_data = torchvision.datasets.Food101(root="data", split="test", transform=simple_transform, download=True)

# print(len(train_data), len(test_data))

# ----------------------------------------------------------------------------------------
# 2. Create Datasets and DataLoaders
BATCH_SIZE=32 # use a big batch size to get through all the images (100,000+ in Food101)

train_dataloader_big = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)  # avoid copies of the data into and out of memory, where possible (for speed ups)
test_dataloader_big = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, pin_memory=True)

# ----------------------------------------------------------------------------------------
# 3. Create model
effnetv2_s_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
foodvision_big_model = torchvision.models.efficientnet_v2_s(weights=effnetv2_s_weights).to(device)

for param in foodvision_big_model.features.parameters():
	param.requires_grad=False

foodvision_big_model.classifier = nn.Sequential(
	nn.Dropout(p=0.2),
	nn.Linear(in_features=1280, out_features=101)
).to(device)

# print(summary(model=foodvision_big_model, input_size=(1,3,224,224)))

# ----------------------------------------------------------------------------------------
# 4. Train
# [Experiment 3. Scale up the dataset using the entire Food101]
epochs=5

foodvision_big_results = engine.train(
	model=foodvision_big_model,
	train_dataloader=train_dataloader_big,
	test_dataloader=test_dataloader_big,
	optimizer=torch.optim.Adam(params=foodvision_big_model.parameters(), lr=.001),
	loss_fn=nn.CrossEntropyLoss(),
	epochs=epochs,
	device=device,
	writer=engine.create_writer(experiment_name="food101", model_name="foodvision_big", extra=f"{epochs}_epochs")
)

"""
Terminal: tensorboard --logdir=runs
or
Ctrl+Shift+P -> python tensorboard
"""