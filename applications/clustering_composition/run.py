
def run(chromosome, trainX, trainY, testX, testY, timestep):

		if op == 0:
			self.get_cluster_creation(data, clusters, n_clusters, iteration)
		if op == 1:
			return self.get_cluster_addition(data, clusters, n_clusters, iteration)
		if op == 2:
			return self.get_cluster_removal(data, clusters, n_clusters, iteration)
		if op == 3:
			return self.get_cluster_merging(data, clusters, n_clusters, iteration)
		if op == 4:
			return self.get_cluster_splitting(data, clusters, n_clusters, iteration)