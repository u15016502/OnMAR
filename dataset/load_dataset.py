import random 
import torchvision
import numpy as np
import torch 
from torchvision import transforms

RANDOM_ORDER = []

def load_image_dataset(dataset, batch_size):
	
	if dataset == "mnist":
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Grayscale(num_output_channels=3),
			torchvision.transforms.ToTensor(),
		])
		data = torchvision.datasets.MNIST('./data', download=True, transform=transform)
	if dataset == "fashion-mnist":
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Grayscale(num_output_channels=3),
			torchvision.transforms.ToTensor(),
		])
		data = torchvision.datasets.FashionMNIST('./data', download=True, transform=transform)
	if dataset == "cifar10":
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
		])
		data = torchvision.datasets.CIFAR10('./data', download=True, transform=transform)
	if dataset == "cifar100":
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
		])
		data = torchvision.datasets.CIFAR100('./data', download=True, transform=transform)
		
	data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1)

	return data_loader

# def load_video_dataset(dataset, frame_sampling):
# 	if dataset == "ucf101":
# 	if dataset == "hmdb51":
# 	if dataset == "lmtd"