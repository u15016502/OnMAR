from genetic_algorithm.individual import individual
from copy import deepcopy
import random

def crossover_nn(inds):
	first_ind = inds[0]
	second_ind = inds[1]

	num_conv1 = first_ind.chromosome[-3]
	num_dense1 = first_ind.chromosome[-2]

	num_conv2 = second_ind.chromosome[-3]
	num_dense2 = second_ind.chromosome[-2]

	ind1 = individual(application=inds[0].application)
	ind2 = individual(application=inds[1].application)

	ind1.chromosome = deepcopy(second_ind.chromosome[:num_conv2] + first_ind.chromosome[num_conv1: num_conv1 + num_dense1] + [second_ind.chromosome[-3],first_ind.chromosome[-2],second_ind.chromosome[-1]])
	ind2.chromosome = deepcopy(first_ind.chromosome[:num_conv1] + second_ind.chromosome[num_conv2: num_conv2 + num_dense2] + [first_ind.chromosome[-3],second_ind.chromosome[-2],first_ind.chromosome[-1]])

	ind1.state = deepcopy(inds[0].state) + deepcopy(inds[1].state)
	ind2.state = deepcopy(inds[0].state) + deepcopy(inds[1].state)

	return random.choice([ind1, ind2])

def mutation_nn(inds, ind):
	first_ind = inds[0]
	second_ind = ind

	num_conv1 = first_ind.chromosome[-3]
	num_dense1 = first_ind.chromosome[-2]

	ind1 = individual(application=inds[0].application)

	conv_layers = [[
		c[0], #random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048]),	# convolutional filters
		random.choice([0, 1]),										# batch normalisation
		random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),					# activation function
		random.choice([-1, random.uniform(0.0, 1.0)]),				# dropout
		c[-1] # random.choice([0, 2, 4, 6, 8])								# max pooling 
	] if random.choice([0,1]) == 1 else c for c in first_ind.chromosome[:num_conv1]]

	dense_layers = [[
		d[0], # random.choice([16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),	# dense nodes
		random.choice([0, 1]),											# batch normalisation
		random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),						# activation function
		random.choice([-1, random.uniform(0.0, 1.0)])					# dropout
	] if random.choice([0,1]) == 1 else d for d in first_ind.chromosome[num_conv1: num_conv1 + num_dense1]]

	optimiser = random.choice([1, 2, 3, 4, 5, 6, 7, 8])

	# add_conv = random.choice([0, 1])
	# add_dense = random.choice([0, 1])
	# remove_conv = random.choice([0, 1])
	# remove_dense = random.choice([0, 1])

	# if add_conv == 1 and num_conv1 <= 6:
	# 	num_conv1 += 1
	# 	conv_layers = conv_layers + [conv_layers[-1]]
	# else:
	# 	if remove_conv == 1 and num_conv1 >= 2:
	# 		num_conv1 -= 1
	# 		conv_layers.pop(0)

	# if add_dense == 1 and num_dense1 <= 6:
	# 	num_dense1 += 1
	# 	dense_layers = dense_layers + [dense_layers[-1]]
	# else:
	# 	if remove_dense == 1 and num_dense1 >= 2:
	# 		num_dense1 -= 1
	# 		dense_layers.pop(0)

	ind1.chromosome = deepcopy(conv_layers + dense_layers + [num_conv1, num_dense1, optimiser])
	ind1.state = deepcopy(inds[0].state)

	return ind1
