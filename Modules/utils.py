"""
Utility functions for PyTorch model training and saving.
"""
import torch

import os
from pathlib import Path 
import matplotlib.pyplot as plt
from typing import List

# -------------------------------------------------------------------------------------------
def set_seeds(seed:int=42):
	# Set the seed for geenral torch operations
	torch.manual_seed(seed)
	# Set the seed for CUDA torch operations (one that happen on the GPU)
	torch.cuda.manual_seed(seed)

# -------------------------------------------------------------------------------------------
# Walk through an directory and find out how many files are in each subdirectory.
def walk_through_dir(dir_path):
	"""
	Walks through dir_path returning its contents.
	Args:
	dir_path (str): target directory
	
	Returns:
	A print out of:
		number of subdirectories in dir_path
		number of images (files) in each subdirectory
		name of each subdirectory
	"""
	for dirpath, dirnames, filenames in os.walk(dir_path):
		print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# -------------------------------------------------------------------------------------------
def save_model(model:torch.nn.Module,
			   target_dir:str,
			   model_name:str):
	"""Saves a PyTorch model to a target directory.
	
	Args:
		model: A target PyTorch model to save.
		target_dir: A directory for saving the model to.
		model_name: A filename for the saved model. Should include either ".pth" or "pt" as the file extension.
	
	Example usage:
		save_model(model=model_9, target_dir="models", model_name="06_tinyvgg_model.pth")
	"""

	# Create target directory
	target_dir_path = Path(target_dir)
	target_dir_path.mkdir(parents=True, exist_ok=True)

	# Create model save path
	assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
	model_save_path = target_dir_path / model_name

	# Save the model state_dict()
	print(f"[INFO] Saving model to: {model_save_path}")
	torch.save(obj=model.state_dict(), f=model_save_path)

# -------------------------------------------------------------------------------------------
# Plot loss curves of a model
def plot_loss_curves(results):
	"""Plots training curves of a results dictionary.
	
	Args:
		results (dict): dictionary containing list of values, e.g.
		{"train_loss" : [...], "train_acc" : [...], "test_loss" : [...], "test_acc" : [...]}
	"""

	loss = results["train_loss"]
	test_loss = results["test_loss"]

	acc = results["train_acc"]
	test_acc = results["test_acc"]

	epochs = range(len(results["train_loss"]))

	plt.figure(figsize=(11,4))

	# Plot loss
	plt.subplot(1,2,1)
	plt.plot(epochs, loss, label='Train')
	plt.plot(epochs, test_loss, label='Test')
	plt.title('Loss')
	plt.xlabel('Epochs')
	plt.legend()

	# Plot accuracy
	plt.subplot(1,2,2)
	plt.plot(epochs, acc, label='Train')
	plt.plot(epochs, test_acc, label='Test')
	plt.title('Accuracy')
	plt.xlabel('Epochs')
	plt.legend()

	plt.show()

# -------------------------------------------------------------------------------------------
# Pred and plot image
def pred_and_plot(
		model:torch.nn.Module,
		image_path:str,
		class_names:List[str],
		transform,
		device:torch.device="cuda" if torch.cuda.is_available() else "cpu",
):
	"""Makes a prediction on a target image with a trained model and plots the image.
	
	Args:
		model (torch.nn.Module): trained PyTorch image classification model.
		image_path (str): filepath to target image.
		class_names (List[str]): different class names for target image. 
		transform (_type_): transform of target image.
		device (torch.device, optional): target device to compute on. Default to "cuda" if torch.cuda.is_available() else "cpu".
	
	Returns:
		Matplotlib plot of target image and model prediction as title.
	"""

	from PIL import Image
	image = Image.open(image_path)
	transformed_image = transform(image)

	model.to(device)
	model.eval()
	with torch.inference_mode():
		pred_logit = model(transformed_image.unsqueeze(0).to(device))
		pred_label  = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1)
	
	plt.figure()
	plt.imshow(image)
	plt.title(f"Pred: {class_names[pred_label]}")
	plt.axis(False)
	plt.show()

# -------------------------------------------------------------------------------------------
# Make confusion matrix with the test preds and the truth labels
def plot_confusion_mat(model:torch.nn.Module, 
						  dataloader:torch.utils.data.DataLoader, 
						  class_names:List[str],
						  device:torch.device="cuda" if torch.cuda.is_available() else "cpu",):
	"""Makes a confusion matrix  with the test preds and the truth labels."""
	from torchmetrics import ConfusionMatrix
	from mlxtend.plotting import plot_confusion_matrix

	test_preds = []
	model.to(device)
	model.eval()
	with torch.inference_mode():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)

			pred_labels = torch.argmax(torch.softmax(model(X), dim=1), dim=1)
			test_preds.append(pred_labels)
	
	test_preds = torch.cat(test_preds).cpu()
	test_truth = torch.cat([y for _, y in dataloader])

	confmat = ConfusionMatrix(num_classes = len(class_names), task="multiclass")
	confmat_tensor = confmat(preds=test_preds, target=test_truth)

	fig, ax = plot_confusion_matrix(
		conf_mat = confmat_tensor.numpy(),
		class_names=class_names,
		figsize=(6,4)
	)
	plt.show()

# -------------------------------------------------------------------------------------------
# Returns a list of dictionaries with sample, label, prediction, pred_prob
def pred_and_store(test_paths, model, transform, class_names, device):
	"""Returns a DataFrame with sample, label, prediction, pred_prob"""

	from tqdm.auto import tqdm 
	import pandas as pd

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
			pred_prob = torch.softmax(model(transformed_image.to(device)), dim=1)
			pred_class = class_names[torch.argmax(pred_prob, dim=1).cpu()]

			pred_dict["pred_prob"] = pred_prob.max().cpu().item()
			pred_dict["pred_class"] = pred_class
		
		pred_dict["correct"] = (class_name==pred_class)

		test_pred_list.append(pred_dict)
		test_pred_df = pd.DataFrame(test_pred_list)
	
	return test_pred_df

# -------------------------------------------------------------------------------------------
