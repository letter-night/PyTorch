"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
import torchvision
from torch import nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units."""

    def __init__(self, input_shape:int, hidden_units:int, output_shape:int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# -------------------------------------------------------------------------------------------
# Create an EffNetB0 faeture extractor
def create_effnetb0(class_names, device="cuda" if torch.cuda.is_available() else "cpu"):
	# 1. Get the base model with pretrained weights and send to target device
	weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
	model = torchvision.models.efficientnet_b0(weights=weights).to(device)

	# 2. Freeze the base model layers
	for param in model.features.parameters():
		param.requires_grad=False
	
	# 3. Set the seeds
	# utils.set_seeds()

	# 4. Change the classifier head
	model.classifier = nn.Sequential(
		nn.Dropout(p=.2), 
		nn.Linear(in_features=1280, out_features=len(class_names))
	).to(device)

	# 5. Give the odel a name
	model.name = "effnetb0"
	print(f"[INFO] Created new {model.name} model.")

	return model

# -------------------------------------------------------------------------------------------
# Create an EffNetB2 feature extractor
def create_effnetb2(class_names, device="cuda" if torch.cuda.is_available() else "cpu"):
	weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
	model = torchvision.models.efficientnet_b2(weights=weights).to(device)

	for param in model.features.parameters():
		param.requires_grad=False
	
	# utils.set_seeds()

	model.classifier = nn.Sequential(
		nn.Dropout(p=.3),
		nn.Linear(in_features=1408, out_features=len(class_names))
	).to(device)

	model.name = "effnetb2"
	print(f"[INFO] Created new {model.name} model.")

	return model

# -------------------------------------------------------------------------------------------
# Create an EffNetV2_s feature extractor
def create_effnetv2_s(class_names, device="cuda" if torch.cuda.is_available() else "cpu"):
	weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
	model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
	dropout=0.2
	in_features=1280
	out_features=len(class_names)
	
    # Freeze the base model layers
	for param in model.features.parameters():
		param.requires_grad=False
	
    # Set the seeds
	# utils.set_seeds()
	
    # Update the classifier head
	model.classifier = nn.Sequential(
		nn.Dropout(p=dropout, inplace=True),
		nn.Linear(in_features=in_features, out_features=out_features)
    ).to(device)
	
    # Set the model name
	model.name="effnetv2_s"
	print(f"[INFO] Creating {model.name} faeture extractor model...")
	return model

# -------------------------------------------------------------------------------------------