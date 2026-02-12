# Python program to demonstrate
# command line arguments


import argparse
import math 
from metalearner.metalearner import metalearner
from genetic_algorithm.individual import individual
from dataset.load_dataset import load_image_dataset
from genetic_algorithm.ops import crossover, mutation
import time
from random import seed
import numpy as np
from copy import deepcopy

if __name__ == '__main__':

	run_seed = math.floor(time.time())
	seed(run_seed)
	np.random.seed(int(run_seed))
	print(run_seed)

	# Initialize parser
	parser = argparse.ArgumentParser()

	# Adding optional argument
	parser.add_argument("-a", "--application", choices=["video_configuration", "neural_network_configuration", "clustering_composition"])
	parser.add_argument("-d", "--dataset", choices=["mnist", "fashion-mnist", "cifar10", "cifar100", "mosquito", "fruitsgb", "melanoma", "ucf101", "hmdb51", "lmtd"])
	parser.add_argument("-m", "--meta_learner", choices=["xgb","knn","rf"])
	parser.add_argument("-g", "--num_generations", type=int)
	parser.add_argument("-x", "--crossover_rate", type=float)
	parser.add_argument("-z", "--mutation_rate", type=float)
	parser.add_argument("-p", "--population_size", type=int)

	# Read arguments from command line
	args = parser.parse_args()

	if args.application == "video_configuration":
		timesteps = 100
		batch_size = 10
	elif args.application == "neural_network_configuration":
		timesteps = 80
		batch_size = 4
	elif args.application == "clustering_composition":
		timesteps = 20
		batch_size = 128
	else: 
		exit(-1)

	ml = metalearner(args.meta_learner)

	data_loader = load_image_dataset(args.dataset, batch_size)

	for t in range(timesteps):
		if t == 0:
			pop = [individual(args.application) for i in range(args.population_size)]
		else:
			pop = [individual(args.application) for i in range(args.population_size - 1)] + [best_ind]

		for g in range(args.num_generations):
			print('TIMESTEP=' + str(t) + '/' + str(timesteps) + ' GENERATION=' + str(g) + '/' + str(args.num_generations))

			if t == 0:
				state = None
			else:
				state = deepcopy(best_ind.state)

			fitnesses = []
			
			for ind in pop:
				try:
					ind.evaluate(data_loader, t, state)
				except Exception as e:
					print(e)
					data_loader = load_image_dataset(args.dataset, batch_size)
					ind.evaluate(data_loader, t, state)

				fitnesses.append(ind.fitness)
				
			best_ind = max(pop)

			crossover_pop = []

			for i in range(math.floor(len(pop) * args.crossover_rate)):
				ind = crossover(pop)
				crossover_pop.append(ind)

			mutation_pop = [mutation(pop) for i in range(math.floor(len(pop) * args.mutation_rate))]
			
			pop = crossover_pop + mutation_pop

			print('BEST=' + str(best_ind.fitness))
			