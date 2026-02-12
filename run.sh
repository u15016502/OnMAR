# python main.py \
# 	--application "neural_network_configuration" \
# 	--dataset "mnist" \
# 	--meta_learner "xgb" \
# 	--num_generations 1 \
# 	--crossover_rate 0.5 \
# 	--mutation_rate 0.5 \
# 	--population_size 10 > static1.out

# python main.py \
# 	--application "neural_network_configuration" \
# 	--dataset "mnist" \
# 	--meta_learner "xgb" \
# 	--num_generations 3 \
# 	--crossover_rate 0.8 \
# 	--mutation_rate 0.2 \
# 	--population_size 4 > dynamic_mnist_3.out

python main.py \
	--application "neural_network_configuration" \
	--dataset "cifar10" \
	--meta_learner "xgb" \
	--num_generations 3 \
	--crossover_rate 0.8 \
	--mutation_rate 0.2 \
	--population_size 4 > dynamic_cifar_1.out

python main.py \
	--application "neural_network_configuration" \
	--dataset "cifar10" \
	--meta_learner "xgb" \
	--num_generations 3 \
	--crossover_rate 0.8 \
	--mutation_rate 0.2 \
	--population_size 4 > dynamic_cifar_2.out

# python main.py \
# 	--application "clustering_composition" \
# 	--dataset "mnist" \
# 	--meta_learner "xgb" \
# 	--num_generations 1 \
# 	--crossover_rate 0.8 \
# 	--mutation_rate 0.2 \
# 	--population_size 5 > clustering_static_1.out
