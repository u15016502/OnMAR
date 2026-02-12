import random
from copy import deepcopy
from genetic_algorithm.individual import individual
from applications.neural_network_configuration.utils import crossover_nn, mutation_nn

def select(pop, num_inds=2, tournament_size=3): 
	inds = []

	for i in range(num_inds):
		selected = []

		for t in range(tournament_size):
			random_idx = random.choice(list(range(len(pop))))
			selected.append(pop[random_idx])

		inds.append(max(selected))
	
	return inds 

def crossover(pop):
	inds = select(pop)

	if inds[0].application == "neural_network_configuration":
		return crossover_nn(inds)

	crossover_idx = random.choice(list(range(min([len(inds[0].chromosome), len(inds[1].chromosome)]))))

	ind1 = individual(application=inds[0].application)
	ind2 = individual(application=inds[1].application)

	ind1.chromosome = deepcopy(inds[0].chromosome[:crossover_idx]) + deepcopy(inds[1].chromosome[crossover_idx:])
	ind2.chromosome = deepcopy(inds[1].chromosome[:crossover_idx]) + deepcopy(inds[0].chromosome[crossover_idx:])

	return random.choice([ind1, ind2])

def mutation(pop):
	inds = select(pop, 1)

	ind = individual(application=inds[0].application)

	if inds[0].application == "neural_network_configuration":
		return mutation_nn(inds, ind)

	crossover_idx1 = random.choice(list(range(min([len(inds[0].chromosome), len(ind.chromosome)]))))
	crossover_idx2 = deepcopy(crossover_idx1)

	coinflip = random.choice([0,1])

	if coinflip == 0:
		inds[0].chromosome = deepcopy(inds[0].chromosome[:crossover_idx1]) + deepcopy(ind.chromosome[crossover_idx2:])
	else:
		inds[0].chromosome = deepcopy(ind.chromosome[:crossover_idx2]) + deepcopy(inds[0].chromosome[crossover_idx1:])

	return inds[0]