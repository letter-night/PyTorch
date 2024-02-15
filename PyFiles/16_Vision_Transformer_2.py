import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn 
from torchvision import transforms 

from torchinfo import summary

from utils import set_seeds, plot_loss_curves

import get_data, data_setup, engine, utils

from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------------------------
# 1. Get Data
# Download images from GitHub
image_path = get_data.download_data(
	source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
	destination="pizza_steak_sushi"
)

data_20_percent_path = get_data.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir_20_percent = data_20_percent_path / "train"

class_names = ["pizza", "steak", "sushi"]

# ------------------------------------------------------------------------------------------------
# 2. Train a pretrained ViT feature extractor model on 20% of the pizza, steak and sushi data

# Download pretrained ViT weights and model
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=vit_weights).to(device)

# Freeze all layers in pretrained ViT model
for param in pretrained_vit.parameters():
	param.requires_grad=False

# Update the  pretrained ViT head
embedding_dim = 768
set_seeds()

pretrained_vit.heads = nn.Sequential(
	nn.LayerNorm(normalized_shape=embedding_dim),
	nn.Linear(in_features=embedding_dim, out_features=len(class_names))
).to(device)

# print(summary(model=pretrained_vit, 
# 			  input_size=(1,3,224,224),
# 			  col_names=["input_size", "output_size", "num_params", "trainable"],
# 			  col_width=20,
# 			  row_settings=["var_names"]))

# Preprocess the data
vit_transforms = vit_weights.transforms() # get transforms from vit_weights
train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir = train_dir_20_percent,
	test_dir=test_dir,
	train_transform=vit_transforms,
	test_transform=vit_transforms,
	batch_size=8
)

optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

set_seeds()
pretrained_vit_results = engine.train(
	model=pretrained_vit,
	train_dataloader=train_dataloader_20_percent,
	test_dataloader=test_dataloader,
	optimizer=optimizer,
	loss_fn=loss_fn,
	epochs=5,
	device=device,
	writer=engine.create_writer(experiment_name="vit", model_name="default_weights")
)

plot_loss_curves(pretrained_vit_results)

# ------------------------------------------------------------------------------------------------
# 3. Using the "ViT_B_16_Weights.IMGENET1K_SWAG_E2E_V1" pretrained weights

# Download pretrained ViT weights and model
vit_weights_swag = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
pretrained_vit_swag = torchvision.models.vit_b_16(weights=vit_weights_swag).to(device)

# Freeze all layers in pretrained ViT model
for param in pretrained_vit_swag.parameters():
	param.requires_grad=False

# Update the pretrained ViT head
embedding_dim = 768
set_seeds()
pretrained_vit_swag.heads = nn.Sequential(
	nn.LayerNorm(normalized_shape=embedding_dim),
	nn.Linear(in_features=embedding_dim, out_features=len(class_names))
).to(device)

print(summary(model=pretrained_vit_swag,
			  input_size=(1,3,384,384),
			  col_names=["input_size", "output_size", "num_params", "trainable"],
			  col_width=20,
			  row_settings=["var_names"]))

vit_transforms_swag = vit_weights_swag.transforms()
print(vit_transforms_swag)

train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir=train_dir_20_percent,
	test_dir=test_dir,
	train_transform=vit_transforms_swag,
	test_transform=vit_transforms_swag,
	batch_size=8
)

optimizer = torch.optim.Adam(params=pretrained_vit_swag.parameters(),
							 lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

set_seeds()
pretrained_vit_swag_results = engine.train(model=pretrained_vit_swag,
										   train_dataloader=train_dataloader_20_percent,
										   test_dataloader=test_dataloader,
										   optimizer=optimizer,
										   loss_fn=loss_fn,
										   device=device,
										   epochs=5,
										   writer=engine.create_writer(experiment_name="vit", model_name="swag_weights"))

plot_loss_curves(pretrained_vit_swag_results)

save_filepath = "swag_weights.pth"
utils.save_model(model=pretrained_vit_swag, target_dir="models", model_name=save_filepath)

# ------------------------------------------------------------------------------------------------
# 4. Get the "most wrong" examples from the test dataset

vit_weights_swag = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
loaded_model = torchvision.models.vit_b_16(weights=vit_weights_swag).to(device)

for param in loaded_model.parameters():
	param.requires_grad=False

embedding_dim=768

loaded_model.heads = nn.Sequential(
	nn.LayerNorm(normalized_shape=embedding_dim),
	nn.Linear(in_features=embedding_dim, out_features=len(class_names))
).to(device)

loaded_model.load_state_dict(torch.load("models/swag_weights.pth"))

vit_transforms_swag = vit_weights_swag.transforms()

test_data_paths = list(Path(test_dir).glob("*/*.jpg"))

test_pred_df = utils.pred_and_store(test_paths=test_data_paths, transform=vit_transforms_swag, 
					 model=loaded_model, class_names=class_names, device=device)

top_5_most_wrong = test_pred_df.sort_values(by=["correct", "pred_prob"], ascending=[True, False]).head()

print(top_5_most_wrong)

print(test_pred_df.correct.value_counts())

for row in top_5_most_wrong.iterrows():
	row = row[1]
	image_path = row[0]
	true_label = row[1]
	pred_prob = row[2]
	pred_class = row[3]

	img = torchvision.io.read_image(str(image_path))
	plt.figure()
	plt.imshow(img.permute(1,2,0))
	plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.3f}")
	plt.show()

# ------------------------------------------------------------------------------------------------
"""
Terminal: tensorboard --logdir=runs
or
Ctrl+Shift+P -> python tensorboard
"""	

"""A few Missing Things
- Learing rate warmup
- Learing rate decay
- Gradient clipping
"""