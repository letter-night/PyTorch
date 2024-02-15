import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn 
from torchvision import transforms 

from torchinfo import summary

from helper_functions import set_seeds, plot_loss_curves

import get_data, data_setup, engine, utils 
from custom_vit import PatchEmbedding, MultiheadSelfAttentionBlock, MLPBlock
from custom_vit import TransformerEncoderBlock, ViT

from numba import cuda

device = cuda.get_current_device()
device.reset()

device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------------------------------
# 1. Get Data
# Download images from GitHub
image_path = get_data.download_data(
	source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
	destination="pizza_steak_sushi"
)

# print(image_path)

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"

# ------------------------------------------------------------------------------------------------
# 2. Create Datasets and DataLoaders

IMG_SIZE = 224

# Create transform pipeline 
simple_transforms = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ToTensor()
])

BATCH_SIZE = 1

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
	train_dir=train_dir,
	test_dir=test_dir,
	train_transform=simple_transforms,
	test_transform=simple_transforms,
	batch_size=BATCH_SIZE
)

# ------------------------------------------------------------------------------------------------
# Visualize a single image
image_batch, label_batch = next(iter(train_dataloader))
image, label  = image_batch[0], label_batch[0]

# print(image.shape, label)

# plt.imshow(image.permute(1,2,0))
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# 1) Set patch size
patch_size = 16

# 2) Print shape of original image tensor and get the image dimensions
print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

# 3) Get image tensor and add batch dimension
x = image.unsqueeze(0)
print(f"Input image with batch dimension shape: {x.shape}")

# 4) Create patch embedding layer
patch_embedding_layer = PatchEmbedding(in_channels=3, patch_size=patch_size, embedding_dim=768)

# 5) Pass image through patch embedding layer
patch_embedding = patch_embedding_layer(x)
print(f"Patch embedding shape: {patch_embedding.shape}")

# 6) Create class token embedding
batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension), requires_grad=True)
print(f"Class token embedding shape: {class_token.shape}")

# 7) Prepend class token embedding to patch embedding
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

# 8) Create position embedding
num_of_patches = int((height*width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, num_of_patches+1, embedding_dimension), requires_grad=True)

# 9) Add potision embedding to patch embedding with class token
patch_and_position_embedding = patch_embedding_class_token + position_embedding
print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")

# 10) Create an instance of MSABlock
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768,
															 num_heads=12)

# 11) Pass patch and position image embedding through MSABlock
patched_image_through_msa_block = multihead_self_attention_block(patch_and_position_embedding)
print(f"Output shape MSA block: {patched_image_through_msa_block.shape}")

# 12) Create an instance of MLPBlock
mlp_block = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.1)

# 13) Pass output of MSABlock through MLPBlock
patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
print(f"Output shape of MLP block: {patched_image_through_mlp_block.shape}")

# ------------------------------------------------------------------------------------------------
# Create an instance of TransformerEncoderBlock
transformer_encoder_block = TransformerEncoderBlock()

# Print an input and output summary of our Transformer Encoder
# print(summary(model=transformer_encoder_block,
# 		input_size=(1,197,768), # (batch_size, num_patches, embedding_dimension)
# 		col_names=["input_size", "output_size", "num_params", "trainable"],
# 		col_width=20,
# 		row_settings=["var_names"]))

# ------------------------------------------------------------------------------------------------
# Create a Transformer Encoder with torch.nn.TransformerEncoderLayer()
torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
															 nhead=12,
															 dim_feedforward=3072,
															 dropout=0.1,
															 activation="gelu",
															 batch_first=True,
															 norm_first=True)

# print(summary(model=torch_transformer_encoder_layer,
# 			  input_size=(1,197,768),
# 			  col_names=["input_size", "output_size", "num_params", "trainable"],
# 			  col_width=20,
# 			  row_settings=["var_names"]))

# ------------------------------------------------------------------------------------------------
# 3. Create a Model
# random_image_tensor = torch.randn(1,3,224,224).to(device)

# vit = ViT(num_classes=len(class_names)).to(device)

# print(vit(random_image_tensor))

# print(summary(model=vit, input_size=(32,3,224,224),
# 			  col_names=["input_size", "output_size", "num_params", "trainable"],
# 			  col_width=20,
# 			  row_settings=["var_names"]))

# ------------------------------------------------------------------------------------------------
# 4. Training our ViT model

# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
# optimizer = torch.optim.Adam(params=vit.parameters(), 
# 							 lr=3e-3,
# 							 betas=(0.9, 0.999),
# 							 weight_decay=0.3)

# loss_fn = torch.nn.CrossEntropyLoss()

# # Set the seeds
# set_seeds()

# torch.cuda.empty_cache()

# # Train the model and save the training results to a dictionary
# results = engine.train(model=vit,
# 					   train_dataloader=train_dataloader,
# 					   test_dataloader=test_dataloader,
# 					   optimizer=optimizer,
# 					   loss_fn=loss_fn,
# 					   epochs=10,
# 					   device=device,
# 					   writer=None)

# ------------------------------------------------------------------------------------------------
# 5. Plot the loss curves of our ViT model
# plot_loss_curves(results)
# plt.show()

# ------------------------------------------------------------------------------------------------
# 6. Using a pretrained ViT

# 1) Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# 2) Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3) Freeze the base parameters
for param in pretrained_vit.parameters():
	param.requires_grad=False

# 4) Change the classifier head (set the seeds to ensure same initialization with linear head)
set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# print(summary(model=pretrained_vit,
# 			  input_size=(32,3,224,224),
# 			  col_names=["input_size", "output_size", "num_params", "trainable"],
# 			  col_width=20,
# 			  row_settings=["var_names"]))

# 5) Preparing data for the pretrained ViT model
# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()
# print(pretrained_vit_transforms)

train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(
	train_dir=train_dir,
	test_dir=test_dir,
	train_transform=pretrained_vit_transforms,
	test_transform=pretrained_vit_transforms,
	batch_size=8
)

optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# 6) Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit,
									  train_dataloader=train_dataloader_pretrained,
									  test_dataloader=test_dataloader_pretrained,
									  optimizer=optimizer,
									  loss_fn=loss_fn,
									  epochs=10,
									  device=device,
									  writer=None)

# Plot the loss curves
plot_loss_curves(pretrained_vit_results)
plt.show()

# 7) Save the model
utils.save_model(model=pretrained_vit,
				 target_dir="models",
				 model_name="pretrained_vit_feature_extractor_pizza_steak_sushi.pth")

# 8) Make predictions on a custom image
import requests 

custom_image_path = image_path / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
	with open(custom_image_path, "wb") as f:
		request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
		print(f"Downloading {custom_image_path}....")
		f.write(request.content)
else:
	print(f"{custom_image_path} already exists, skipping download.")

utils.pred_and_plot_image(model=pretrained_vit,
					image_path = custom_image_path,
					class_names=class_names,
					transform=pretrained_vit_transforms)
plt.show()
