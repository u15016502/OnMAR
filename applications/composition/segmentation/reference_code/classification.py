from kneed import KneeLocator
import sklearn.cluster as sklc
from sklearn.mixture import GaussianMixture
from utils import correct_or_return_input
from sklearn.neighbors import NearestNeighbors
import numpy as np

class classification():

	def __init__(self):
		return

	def classify(self, images, labels, classification_type, classification_args):
# https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
		if classification_type == 'C1': # kmeans
			images = correct_or_return_input(images, labels, (0,))
			num_clusters = classification_args['num_clusters']
			kmeans = sklc.KMeans(n_clusters=num_clusters, init='k-means++').fit(images)
			return kmeans.labels_

		if classification_type == 'C2': # affinity propagation
			images = correct_or_return_input(images, labels, (0,))
			affprop = sklc.AffinityPropagation().fit(images)
			return affprop.labels_

		if classification_type == 'C3': # mean shift
			images = correct_or_return_input(images, labels, (0,))
			ms = sklc.MeanShift(n_jobs=-1).fit(images)
			return ms.labels_

		if classification_type == 'C4': # spectral clustering
			images = correct_or_return_input(images, labels, (0,))
			spec = sklc.SpectralClustering(n_clusters=classification_args['num_clusters'], n_jobs=-1, assign_labels=classification_args['assign_labels']).fit(images)
			return spec.labels_

		if classification_type == 'C5': # agglomerative clustering
			images = correct_or_return_input(images, labels, (0,))
			aggc = sklc.AgglomerativeClustering(linkage=classification_args['linkage']).fit(images)
			return aggc.labels_

		if classification_type == 'C6': # dbscan
			min_samples = images.shape[1] * 2

			if min_samples > len(images):
				min_samples = len(images)
			
			images = correct_or_return_input(images, labels, (0,))

			neighbors = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
			neighbors_fit = neighbors.fit(images)
			distances, indices = neighbors_fit.kneighbors(images)

			distances = np.sort(distances, axis=0)
			distances = distances[:,1]

			num_clusters = range(1, len(distances)+1)

			kn = KneeLocator(num_clusters, distances, curve='convex', direction='increasing')

			if kn.knee == len(distances):
				eps = distances[len(distances) - 1]
			else:
				eps = distances[kn.knee]
		
			dbs = sklc.DBSCAN(eps=eps, min_samples=min_samples, metric=classification_args['metric']).fit(images)
			return dbs.labels_

		if classification_type == 'C7': # optics
			images = correct_or_return_input(images, labels, (0,))
			min_samples = images.shape[1] * 2

			if min_samples > len(images):
				min_samples = len(images)

			optics = sklc.OPTICS(min_samples=min_samples, max_eps=classification_args['max_eps'], metric=classification_args['metric'], n_jobs=-1).fit(images)
			return optics.labels_

		if classification_type == 'C8': # gaussian mixtures
			images = correct_or_return_input(images, labels, (0,))
			if images.shape[1] > 50000:
				return [1 for im in images]

			gaum = GaussianMixture().fit(images)

			labels = gaum.predict(images)
			return labels

		if classification_type == 'C9': # birch
			images = correct_or_return_input(images, labels, (0,))
			bir = sklc.Birch(threshold=classification_args['threshold'], n_clusters=classification_args['n_clusters']).fit(images)
			return bir.labels_

		if classification_type == 'C10': # bisecting k-means
			images = correct_or_return_input(images, labels, (0,))
			if classification_args['n_clusters'] > len(images):
				n_clusters = len(images)
			else:
				n_clusters = classification_args['n_clusters']
			bkm = sklc.BisectingKMeans(n_clusters=n_clusters, init='k-means++', bisecting_strategy=classification_args['bisecting_strategy']).fit(images)
			return bkm.labels_
