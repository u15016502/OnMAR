import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import torch.optim as optim
import math 

class NeuralNetwork(nn.Module):
	def __init__(self, stack):
		super(NeuralNetwork, self).__init__()
		self.stack = nn.ModuleList()
		
		for s in stack:
			self.stack.append(s)
		
	
	def forward(self, x):
		for i, s in enumerate(self.stack):
			x = self.stack[i](x)

		return x

def load_state(model, state):
	weights = []
	biases = []

	for s in state:
		for key in s.keys():
			if '.weight' in key:
				weights.append(s[key])
			if '.bias' in key:
				biases.append(s[key])

	with torch.no_grad():
		for name, param in model.named_parameters():
			if 'weight' in name:
				try:
					weights_set = False
					weights_chosen = None

					for i, w in enumerate(weights):
						cw = param.flatten().shape[0]
						w_ = w.flatten().shape[0]
						if cw == w_:
							param.copy_(torch.reshape(w, param.shape))
							weights_set = True
							weights_chosen = i
							# print('weights set')
							break
					
					if weights_set == False:
						for i, w in enumerate(weights):
							cw = param.flatten().shape[0]
							w_ = w.flatten().shape[0]
							if cw % w_ == 0:
								w_cat = torch.cat([w] * (cw//w_))				
								param.copy_(torch.reshape(w_cat, param.shape))
								weights_chosen = i
								# print('weights set')
								break

					if weights_chosen != None:
						weights.pop(weights_chosen)
				
				except Exception as e:
					# print(e)
					pass

			if 'bias' in name:
				try:
					bias_set = False
					bias_chosen = None 

					for i, b in enumerate(biases):
						if param.shape[0] == b.shape[0]:
							param.copy_(b)
							bias_set = True
							bias_chosen = i
							# print('bias set')
							break

					if bias_set == False:	
						for i, b in enumerate(biases):
							if param.shape[0] < b.shape[0]:
								param.copy_(b[:param.shape[0]])
								bias_chosen = i
								# print('bias set')
								break

					if bias_chosen != None:
						biases.pop(bias_chosen)

				except Exception as e:
					# print(e)
					pass

	return model

def run(chromosome, dataloader, timestep, state=None):

	num_conv_layers = chromosome[-3]
	num_dense_layers = chromosome[-2]

	prev_in_channels = 3
	out_channels = -1
	chromosome_idx = 0
	stack = []

	test_data = None

	for i, data in enumerate(dataloader, 0):
		images, labels = data
		test_data = images
		break 

	# Creating the convolutional layers
	for i in range(num_conv_layers):
		in_channels = prev_in_channels
		out_channels = chromosome[chromosome_idx][0]
		kernel_size = 3

		if test_data.shape[2] < kernel_size:
			kernel_size = test_data.shape[2]
			
		stack.append(
			nn.Conv2d(in_channels, out_channels, kernel_size)
		)
		test_data = stack[-1](test_data)

		prev_in_channels = deepcopy(out_channels)

		if chromosome[chromosome_idx][1] == 1:
			stack.append(
				nn.BatchNorm2d(prev_in_channels)
			)
			test_data = stack[-1](test_data)
		
		if chromosome[chromosome_idx][2] == 1:
			stack.append(
				nn.GELU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 2:
			stack.append(
				nn.ELU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 3:
			stack.append(
				nn.SELU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 4:
			stack.append(
				nn.ReLU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 5:
			stack.append(
				nn.Sigmoid()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 6:
			stack.append(
				nn.Softmax()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 7:
			stack.append(
				nn.Softplus()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 8:
			stack.append(
				nn.Hardswish()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 9:
			stack.append(
				nn.Tanh()
			)
			test_data = stack[-1](test_data)

		if chromosome[chromosome_idx][3] != -1:
			stack.append(
				nn.Dropout(p=chromosome[chromosome_idx][3])
			)
			test_data = stack[-1](test_data)
		
		if chromosome[chromosome_idx][4] != 0:
			if test_data.shape[2] >= abs(math.floor(((test_data.shape[2] + 2 * 0 - 1 * (chromosome[chromosome_idx][4] - 1) - 1) / 2) + 1)):
				stride = 1
			else:
				stride = 2
			
			if abs(math.floor(((test_data.shape[2] + 2 * 0 - 1 * (chromosome[chromosome_idx][4] - 1) - 1) / 2) + 1)) > 0:
				stack.append(
					nn.MaxPool2d(chromosome[chromosome_idx][4], stride=stride)
				)
				try:
					test_data = stack[-1](test_data)
				except:
					stack.pop()

		chromosome_idx += 1

	stack.append(nn.Flatten())
	test_data = stack[-1](test_data)

	# Creating the dense layers
	for i in range(num_dense_layers):
		in_channels = test_data.shape[1]

		if i == num_dense_layers - 1:
			out_channels = len(dataloader.dataset.classes) 
		else:
			out_channels = chromosome[chromosome_idx][0]

		stack.append(
			nn.Linear(in_channels, out_channels, 3)
		)
		test_data = stack[-1](test_data)

		prev_in_channels = deepcopy(out_channels)

		if chromosome[chromosome_idx][1] == 1:
			stack.append(
				nn.BatchNorm1d(prev_in_channels)
			)
			test_data = stack[-1](test_data)
		
		if chromosome[chromosome_idx][2] == 1:
			stack.append(
				nn.GELU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 2:
			stack.append(
				nn.ELU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 3:
			stack.append(
				nn.SELU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 4:
			stack.append(
				nn.ReLU()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 5:
			stack.append(
				nn.Sigmoid()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 6:
			stack.append(
				nn.Softmax()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 7:
			stack.append(
				nn.Softplus()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 8:
			stack.append(
				nn.Hardswish()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][2] == 9:
			stack.append(
				nn.Tanh()
			)
			test_data = stack[-1](test_data)
		if chromosome[chromosome_idx][3] != -1:
			stack.append(
				nn.Dropout(p=chromosome[chromosome_idx][3])
			)
			test_data = stack[-1](test_data)

		chromosome_idx += 1
	
	model = NeuralNetwork(stack)
	criterion = nn.CrossEntropyLoss()
	
	if chromosome[-1] == 1:
		optimiser = optim.Adam(model.parameters())
	elif chromosome[-1] == 2:
		optimiser = optim.Adamax(model.parameters())
	elif chromosome[-1] == 3:
		optimiser = optim.RMSprop(model.parameters())
	elif chromosome[-1] == 4:
		optimiser = optim.Adagrad(model.parameters())
	elif chromosome[-1] == 5:
		optimiser = optim.Adadelta(model.parameters())
	elif chromosome[-1] == 6:
		optimiser = optim.SGD(model.parameters(), lr=0.001)
	elif chromosome[-1] == 7:
		optimiser = optim.NAdam(model.parameters())
	else:
		optimiser = optim.RAdam(model.parameters())

	if timestep > 0:
		model = load_state(model, state)

	# Training
	for epoch in range(1): 
		running_loss = 0.0
		correct = 0
		total = 0
		for i, data in enumerate(dataloader, 0):
			inputs, labels = data

			optimiser.zero_grad()

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimiser.step()
			running_loss = loss.item()
			
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			# print('[%d, %5d] loss: %.4f acc: %.4f' %
			# 		(epoch + 1 + timestep, i + 1, running_loss, (100 * correct / total)))

					# (epoch + 1, i + 1, running_loss / 2000))
			# running_loss = 0.0

			if i > 30:
				break
		
	# Testing
	with torch.no_grad():
		for i, data in enumerate(dataloader, 0):
			images, labels = data
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			if i > 50:
				break

	fitness = (100 * correct / total)
	state = model.state_dict()

	return fitness, running_loss, state
	