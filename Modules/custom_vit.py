import torch
import torchvision

from torch import nn 

# ------------------------------------------------------------------------------------------------
# 1. Patch embedding layer

# 1) Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
	"""Turns a 2D input image into a 1D sequence learnable embedding vector.
	
	Args:
		in_channels (int): Number of color channels for the input images. Defaults to 3.
		patch_size (int): Size of patches to convert input image into. Default to 16.
		embedding_dim (int): Size of embedding to turn image into. Default to 768.
	"""
	# 2) Initialize the class with appropriate variables
	def __init__(self, 
			  in_channels:int=3,
			  patch_size:int=16,
			  embedding_dim:int=768):
		super().__init__()

		self.patch_size = patch_size

		# 3) Create a layer to turn an image into patches
		self.patcher = nn.Conv2d(in_channels=in_channels,
						   out_channels=embedding_dim,
						   kernel_size=patch_size,
						   stride=patch_size,
						   padding=0)

		# 4) Create a layer to flatten the patch feature maps into a single dimension
		self.flatten = nn.Flatten(start_dim=2, end_dim=3)
	
	# 5) Define the forward method
	def forward(self, x):
		# Create assertion to check that inputs are the correct shape
		image_resolution = x.shape[-1]
		assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {patch_size}"

		# Perform the forward pass
		x_patched = self.patcher(x)
		x_flattened = self.flatten(x_patched)
		# 6. Make sure the output shape has the right order
		return x_flattened.permute(0,2,1) # adjust so the embedding is on the final dimension [batch_size, P^2*C, N] -> [batch_size, N, P^2*C]

# ------------------------------------------------------------------------------------------------
# 2. Multi-head Self-Attention (MSA) layer

# 1) Create a class thah inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
	"""Creates a multi-head self-attention block ("MSA block" for short).
	"""
	# 2) Initialize the class with hyperparameters
	def __init__(self, 
			  embedding_dim:int=768,
			  num_heads:int=12,
			  attn_dropout:float=0):
		super().__init__()

		# 3) Create the Norm layer (LN)
		self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

		# 4) Create the Multi-head Attention (MSA) layer
		self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
											  num_heads=num_heads,
											  dropout=attn_dropout,
											  batch_first=True)
	
	# 5) Create a forward() method to pass the data through the layers
	def forward(self, x):
		x = self.layer_norm(x)
		attn_output, _ = self.multihead_attn(query=x,
									   key=x,
									   value=x,
									   need_weights=False)
		
		return attn_output

# ------------------------------------------------------------------------------------------------
# 3. Multilayer Perceptron (MLP)

# 1) Create a class that inherits from nn.Module
class MLPBlock(nn.Module):
	"""Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
	# 2) Initialize the class with hyperparameters
	def __init__(self, 
			  embedding_dim:int=768,
			  mlp_size:int=3072,
			  dropout:float=0.1):
		super().__init__()

		# 3) Create the Norm layer (LN)
		self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

		# 4) Create the Multilayer perceptron (MLP) layer(s)
		self.mlp = nn.Sequential(
			nn.Linear(in_features=embedding_dim, out_features=mlp_size),
			nn.GELU(),
			nn.Dropout(p=dropout),
			nn.Linear(in_features=mlp_size, out_features=embedding_dim),
			nn.Dropout(p=dropout)
		)
	
	# 5) Create a forward() method to pass the data through the layers
	def forward(self, x):
		x = self.layer_norm(x)
		x = self.mlp(x)
		return x

# ------------------------------------------------------------------------------------------------
# 4. Transformer Encoder Block

# 1) Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
	"""Creates a Transformer Encoder block."""
	# 2) Initialize the class with hyperparameters
	def __init__(self, 
			  embedding_dim:int=768,
			  num_heads:int=12,
			  mlp_size:int=3072,
			  mlp_dropout:float=0.1,
			  attn_dropout:float=0):
		super().__init__()

		# 3) Create MSA block
		self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
											   num_heads=num_heads,
											   attn_dropout=attn_dropout)

		# 4) Create MLP block
		self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
							mlp_size=mlp_size,
							dropout=mlp_dropout)
	
	# 5) Create a forward() method
	def forward(self, x):

		# 6) Create residual connection for MSA block (add the input to the output)
		x = self.msa_block(x) + x

		# 7) Create residual connection for MLP block (add the input to the output)
		x = self.mlp_block(x) + x

		return x

# ------------------------------------------------------------------------------------------------
# 5. Vision Transformer

# 1) Create a ViT Class that inherits from nn.Module
class custom_ViT(nn.Module):
	"""Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
	#2) Initialize the class with hyperparameters
	def __init__(self, 
			  img_size:int=224,
			  in_channels:int=3,
			  patch_size:int=16,
			  num_transformer_layers:int=12,
			  embedding_dim:int=768,
			  mlp_size:int=3072,
			  num_heads:int=12,
			  attn_dropout:float=0,
			  mlp_dropout:float=0.1,
			  embedding_dropout:float=0.1,
			  num_classes:int=1000):
		super().__init__()

		# 3) Make the image size is divisible by the patch size
		assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image_size: {img_size}, patch_size: {patch_size}."

		# 4) Calculate number of patches (height * width / patch**2)
		self.num_patches = (img_size * img_size) // patch_size**2

		# 5) Create learnable class embedding (needs to go at front of sequence of patch embeddings)
		self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)

		# 6) Create learnable position embedding
		self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim), requires_grad=True)

		# 7) Create embedding dropout value
		self.embedding_dropout = nn.Dropout(p=embedding_dropout)

		# 8) Create patch embedding layer
		self.patch_embedding = PatchEmbedding(in_channels=in_channels,
										patch_size=patch_size,
										embedding_dim=embedding_dim)

		# 9) Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
		# Note: The "*" means "all"
		self.transformer_encoder = nn.Sequential(
			*[TransformerEncoderBlock(embedding_dim=embedding_dim,
							 num_heads=num_heads,
							 mlp_size=mlp_size,
							 attn_dropout=attn_dropout,
							 mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)]
		)

		# 10) Create classifier head
		self.classifier = nn.Sequential(
			nn.LayerNorm(normalized_shape=embedding_dim),
			nn.Linear(in_features=embedding_dim, out_features=num_classes)
		)
	
	# 11) Create a forward() method
	def forward(self, x):
		
		# 12) Get batch size
		batch_size = x.shape[0]

		# 13) Create class token embedding and expand it to match the batch size 
		class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension 

		# 14) Create patch embedding
		x = self.patch_embedding(x)

		# 15) Concat class embedding and patch embedding
		x = torch.cat((class_token, x), dim=1)

		# 16) Add posiiton embedding to patch embedding
		x = self.position_embedding + x

		# 17) Run embedding dropout
		x = self.embedding_dropout(x)

		# 18) Pass patch, position and class embedding through transformer encoder layers
		x = self.transformer_encoder(x)

		# 19) Put 0 index logit through classifier
		x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

		return x
	
# ------------------------------------------------------------------------------------------------
# ViT architecture with in-built PyTorch transformer layers
class ViT(nn.Module):
	def __init__(self, 
			  img_size=224,
			  num_channels=3,
			  patch_size=16,
			  embedding_dim=768,
			  dropout=0.1,
			  mlp_size=3072,
			  num_transformer_layers=12,
			  num_heads=12,
			  num_classes=1000):
		super().__init__()

		assert img_size % patch_size == 0, "Image size must be divisible by patch size."

		# 1. Create patch embedding
		self.patch_embedding = PatchEmbedding(in_channels=num_channels,
										patch_size=patch_size,
										embedding_dim=embedding_dim)
		
		# 2. Create class token
		self.class_token = nn.Parameter(torch.randn(1,1,embedding_dim), requires_grad=True)

		# 3. Create positional embedding
		num_patches = (img_size * img_size) // patch_size**2
		self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))

		# 4. Create patch + position embedding dropout
		self.embedding_dropout = nn.Dropout(p=dropout)

		# 5. Create stack Transformer Encoder layers (stacked single layers)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
																								d_model=embedding_dim,
																								nhead=num_heads,
																								dim_feedforward=mlp_size,
																								activation="gelu",
																								batch_first=True,
																								norm_first=True),
			num_layers=num_transformer_layers)
		
		# 6. Create MLP head
		self.mlp_head = nn.Sequential(
			nn.LayerNorm(normalized_shape=embedding_dim),
			nn.Linear(in_features=embedding_dim, out_features=num_classes)
		)
	
	def forward(self, x):
		# Get some dimensions from x
		batch_size = x.shape[0]

		# Create the patch embedding
		x = self.patch_embedding(x)
		# print(x.shape)

		# First, expand the class token across the batch size
		class_token = self.class_token.expand(batch_size, -1, -1)

		# Prepend the class token to the patch embedding
		x = torch.cat((class_token, x), dim=1)
		# print(x.shape)

		# Add the positional embedding to patch embedding with class token
		x = self.positional_embedding + x
		# print(x.shape)

		# Dropout on patch + positional embedding
		x = self.embedding_dropout(x)

		# Pass embedding through Transformer Encoder stack
		x = self.transformer_encoder(x)

		# Pass 0th index of x through MLP head
		x = self.mlp_head(x[:, 0])

		return x