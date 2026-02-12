from copy import deepcopy
from skimage.transform import resize
import random
import math
import matplotlib.pyplot as plt
import sklearn.cluster as sklc
from sklearn.feature_extraction import image as skfei
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.segmentation import slic, chan_vese, felzenszwalb, flood_fill, quickshift, watershed
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import NearestNeighbors
from .filters import correct_or_return_input
from skimage.transform import rescale, resize
import concurrent.futures
from .utils import spec_cluster
from kneed import KneeLocator
import cv2
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from skimage import data
from skimage import color
from skimage import morphology

class segmentation():

	def __init__(self):
		return

		# https://scikit-image.org/docs/stable/api/skimage.segmentation.html
	def segment(self, images, labels, segment_type, args):

		images = np.array([im * (255.0/im.max()) if im.max() > 0 else im for im in images])

		if segment_type == 'S1': # SLIC superpixels
			if len(images[0].shape) == 2:
				channel_axis = None
			else:
				channel_axis = -1

			return np.array([slic(im, n_segments=args['n_segments'],  compactness=args['compactness'], sigma=args['sigma'], channel_axis=channel_axis) for im in images])

		if segment_type == 'S2': # Chan vese
			return np.array([resize(chan_vese(rescale(im, 0.5), mu=args['mu']), im.shape) for im in correct_or_return_input(images, labels, (0,0), 'gray')])

		if segment_type == 'S3': # Felsenszwalb
			if len(images[0].shape) == 2:
				channel_axis = None
			else:
				channel_axis = -1

			return np.array([felzenszwalb(im, scale=args['scale'], sigma=args['sigma'], channel_axis=channel_axis) for im in images])

		if segment_type == 'S4': # Flood fill
			images = correct_or_return_input(images, labels, (0,0))
			try:
				return np.array([flood_fill(im, seed_point=(args['seed'],args['seed']), new_value=255, tolerance=args['tolerance']) for im in images])
			except:
				return np.array([flood_fill(im, seed_point=(math.floor(im.shape[0] / 2),math.floor(im.shape[1] / 2)), new_value=255, tolerance=args['tolerance']) for im in images])

		if segment_type == 'S5': # Quickshift
			return np.array([quickshift(im, max_dist=args['max_dist'], ratio=args['ratio'], kernel_size=args['kernel_size'], sigma=args['sigma']) for im in correct_or_return_input(images, labels, (0,0,0), 'RGB')])

		if segment_type == 'S6': # Watershed
			return np.array([resize(watershed(rescale(im, 0.75), compactness=args['compactness']), im.shape) for im in images])

		if segment_type == 'S7': # K-Medoids
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.2)
				kmedoids = KMedoids(n_clusters=args['n_clusters'], metric=args['metric'], method='alternate')
				kmedoids = kmedoids.fit(im_[0:5].flatten().reshape((-1,1)))
				labels_ = kmedoids.predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S8': # kmeans
			segs = []
			images_ = correct_or_return_input(images, labels, (0,0))
			images = correct_or_return_input(images, labels, (0,0,0))

			for idx, im in enumerate(images_):
				min_samples = images_[0].shape[1] * 2
				
				if min_samples > len(im):
					min_samples = len(im)

				neighbors = NearestNeighbors(n_neighbors=min_samples)
				neighbors_fit = neighbors.fit(im)
				distances, indices = neighbors_fit.kneighbors(im)
				distances = np.sort(distances, axis=0)
				distances = distances[:,1]
				num_clusters = range(1, len(distances)+1)
				kn = KneeLocator(num_clusters, distances, curve='concave', direction='increasing', interp_method='polynomial', online=True)
				if len(kn.all_knees_y) == 0:
					num_clusters = 2
				elif kn.all_knees_y[0] > 50:
					num_clusters = random.choice([2,3,4,5,6,8,10,13,15,20])
				else:
					num_clusters = math.ceil(np.average(np.array(kn.all_knees_y))) + 2

				img=cv2.cvtColor(np.array(images[idx]).astype(np.uint8),cv2.COLOR_BGR2RGB)
				vectorized = img.reshape((-1,3))
				vectorized = np.float32(vectorized)
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
				attempts=5

				ret,label,center=cv2.kmeans(vectorized,num_clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
				center = np.uint8(center)
				res = center[label.flatten()]
				result_image = res.reshape((img.shape))

				segs.append(result_image)

			return np.array(segs)

		if segment_type == 'S9': # affinity propagation
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.1)
				affprop = sklc.AffinityPropagation(damping=args['damping'])
				affprop = affprop.fit(im_[0:5].flatten().reshape((-1,1)))
				labels_ = affprop.predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S10': # mean shift
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.1)
				mn = sklc.MeanShift()
				mn = mn.fit(im_[0:5].flatten().reshape((-1,1)))
				labels_ = mn.predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S11': # spectral clustering
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.1)
				spec = sklc.SpectralClustering(n_clusters=args['num_clusters'], assign_labels='cluster_qr', n_init=2)
				labels_ = spec.fit_predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S12': # agglomerative clustering
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				aggc = sklc.AgglomerativeClustering(linkage=args['linkage']).fit(im)
				segs.append(resize(aggc.labels_, im.shape))

			return np.array(segs)

		if segment_type == 'S13': # dbscan
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				min_samples = im.shape[1] * 2

				if min_samples > len(im):
					min_samples = len(im)

				neighbors = NearestNeighbors(n_neighbors=min_samples)
				neighbors_fit = neighbors.fit(im)
				distances, indices = neighbors_fit.kneighbors(im)

				distances = np.sort(distances, axis=0)
				distances = distances[:,1]

				num_clusters = range(1, len(distances)+1)

				kn = KneeLocator(num_clusters, distances, curve='convex', direction='increasing')

				if kn.knee == len(distances):
					eps = distances[len(distances) - 1]
				else:
					if kn.knee == None:
						eps = 0.5
					else:
						eps = distances[kn.knee]

				im_ = rescale(im, 0.1)
				dbs = sklc.DBSCAN(eps=eps, min_samples=min_samples, metric=args['metric'])
				# dbs = dbs.fit(im_[0:5].flatten().reshape((-1,1)))
				labels_ = dbs.fit_predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S14': # optics
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.1)
				min_samples = im_.shape[1] * 2
				if min_samples > len(im_):
					min_samples = len(im_)

				optics = sklc.OPTICS(min_samples=min_samples, max_eps=args['max_eps'], metric=args['metric'], leaf_size=10)
				# optics = optics.fit(im_[0:5].flatten().reshape((-1,1)))
				labels_ = optics.fit_predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S15': # gaussian mixtures
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.1)
				gm = GaussianMixture()
				gm = gm.fit(im_[0:5])
				labels_ = gm.predict(im_)
				resized = resize(labels_, im.shape)
				segs.append(resized)

			return np.array(segs)

		if segment_type == 'S16': # birch
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				im_ = rescale(im, 0.1)
				bir = sklc.Birch(threshold=args['threshold'], n_clusters=args['n_clusters'])
				bir = bir.fit(im_[0:5].flatten().reshape((-1,1)))
				labels_ = bir.predict(im_.flatten().reshape((-1,1)))
				reshaped = np.reshape(labels_, im_.shape)
				resized = resize(reshaped, im.shape)
				segs.append(resized)
				
			return np.array(segs)

		if segment_type == 'S17': # bisecting k-means
			segs = []
			images = correct_or_return_input(images, labels, (0,0))

			for idx, im in enumerate(images):
				min_samples = images[0].shape[1] * 2

				if min_samples > len(im):
					min_samples = len(im)

				neighbors = NearestNeighbors(n_neighbors=min_samples)
				neighbors_fit = neighbors.fit(im)
				distances, indices = neighbors_fit.kneighbors(im)
				distances = np.sort(distances, axis=0)
				distances = distances[:,1]
				num_clusters = range(1, len(distances)+1)
				kn = KneeLocator(num_clusters, distances, curve='concave', direction='increasing', interp_method='polynomial', online=True)
				if len(kn.all_knees_y) == 0:
					num_clusters = 2
				elif kn.all_knees_y[0] > 50:
					num_clusters = random.choice([2,3,4,5,6,8,10,13,15,20])
				else:
					num_clusters = math.ceil(np.average(np.array(kn.all_knees_y))) + 2

				if im.shape[0] < num_clusters:
					num_clusters = random.choice([2,3,4,5,6,8,10,13,15,20])

				try:
					bkm = sklc.BisectingKMeans(n_clusters=num_clusters, init='k-means++', bisecting_strategy=args['bisecting_strategy']).fit(im)
					segs.append(resize(bkm.labels_, im.shape))
				except:
					try:
						bkm = sklc.BisectingKMeans(n_clusters=2, bisecting_strategy=args['bisecting_strategy']).fit(im)
						segs.append(resize(bkm.labels_, im.shape))
					except:
						segs.append(im)

			return np.array(segs)

		if segment_type == 'S18': # Mask SLIC
			images = correct_or_return_input(images, labels, (0,0))
			segs = []

			for im in images:
				mask = lum = deepcopy(im)
				mask = morphology.remove_small_holes(morphology.remove_small_objects(lum < 0.5, 500), 500)
				if len(np.unique(mask)) == 1:
					segs.append(slic(im, n_segments=args['n_segments'],  compactness=args['compactness'], sigma=args['sigma'], channel_axis=None))
				else:
					segs.append(slic(im, n_segments=args['n_segments'],  compactness=args['compactness'], sigma=args['sigma'], channel_axis=None, mask=mask))
			
			return np.array(segs)
		# def create_gaborfilter():
		# 	# This function is designed to produce a set of GaborFilters 
		# 	# an even distribution of theta values equally distributed amongst pi rad / 180 degree
			
		# 	filters = []
		# 	num_filters = 16
		# 	ksize = 35  # The local area to evaluate
		# 	sigma = 3.0  # Larger Values produce more edges
		# 	lambd = 10.0
		# 	gamma = 0.5
		# 	psi = 0  # Offset value - lower generates cleaner results
		# 	for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
		# 		kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
		# 		kern /= 1.0 * kern.sum()  # Brightness normalization
		# 		filters.append(kern)
		# 	return filters

		# def apply_filter(img, filters):
		# 	# This general function is designed to apply filters to our image
			
		# 	# First create a numpy array the same size as our input image
		# 	newimage = np.zeros_like(img)
			
		# 	# Starting with a blank image, we loop through the images and apply our Gabor Filter
		# 	# On each iteration, we take the highest value (super impose), until we have the max value across all filters
		# 	# The final image is returned
		# 	depth = -1 # remain depth same as original image
			
		# 	for kern in filters:  # Loop through the kernels in our GaborFilter
		# 		image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
				
		# 		# Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
		# 		np.maximum(newimage, image_filter, newimage)
		# 	return newimage