from PIL import Image
import time
from scipy.spatial import distance
import math
from math import hypot
import scipy
from functools import reduce
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, recall_score, precision_score, f1_score, jaccard_score
import numpy as np
import tensorflow as tf
import keras
from skimage.color import rgb2hsv, hsv2rgb
import cv2
from sklearn.feature_extraction import image as skfei
from scipy.ndimage import gaussian_filter
from sklearn.cluster import spectral_clustering
from skimage.transform import rescale, resize

# ====
# ====
# ==== FILTERS 
# ====
# ====

def correct_or_return_input(images, labels, expected_shape, color_space=None):
	if 1 == 1:
				
		if len(images[0].shape) == 1 and len(expected_shape) == 2:
			# images = images.reshape((224,224))
			if color_space == None or color_space == 'gray':
				return images

		elif len(images[0].shape) == 1 and len(expected_shape) == 3:
			temp_images = []
			
			for idx, label in enumerate(labels):
				temp_image = np.array([images[idx]])
				temp_image = temp_image.reshape((label.shape[0],label.shape[1]))
				temp_image = np.expand_dims(temp_image, axis=-1)
				temp_image = np.repeat(temp_image, 3, axis=-1)
				temp_image = temp_image.astype('float32') / 255
				temp_image = tf.image.resize(temp_image, [label.shape[0], label.shape[1]])
				temp_images.append(temp_image[0])

			images = temp_images

			if color_space == None:
				return images

		elif len(images[0].shape) == 2 and len(expected_shape) == 1:
			images = np.array([im.flatten() for im in images])
			if color_space == None:
				return images

		elif len(images[0].shape) == 2 and len(expected_shape) == 2 and color_space == 'binary':
			images = np.array([cv2.threshold(im.astype(np.uint8), 70, 255, 0)[1] for im in images])
			if color_space == None or color_space == 'binary':
				return images

		elif len(images[0].shape) == 2 and len(expected_shape) == 3:
			temp_images = []
			
			for idx, label in enumerate(labels):
				temp_image = np.array([images[idx]])
				temp_image = np.expand_dims(temp_image, axis=-1)
				temp_image = np.repeat(temp_image, 3, axis=-1)
				temp_image = temp_image.astype('float32') / 255
				temp_image = tf.image.resize(temp_image, [label.shape[0], label.shape[1]])
				temp_images.append(temp_image[0])

			images = temp_images

			if color_space == None:
				return images

		elif len(images[0].shape) == 3 and len(expected_shape) == 1 or len(images[0].shape) == 4 and len(expected_shape) == 1:
			images = np.array([im.flatten() for im in images])
			if color_space == None:
				return images

		elif len(images[0].shape) == 3 and len(expected_shape) == 2 and color_space == 'binary':
			images = np.array([cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY) for im in images])
			images = np.array([cv2.threshold(im.astype(np.uint8), 70, 255, 0)[1] for im in images])
			if color_space == None or color_space == 'binary':
				return images

		elif len(images[0].shape) == 3 and len(expected_shape) == 2:
			try:
				images = np.array([cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY) for im in images])
			except:
				images = np.array([cv2.cvtColor(im.astype('float32'), cv2.COLOR_BGR2GRAY) for im in images])

			if color_space == None or color_space == 'gray':
				return images

		if color_space == 'HSV':
			if images[0].shape[2] == 4:
				return np.array([cmyk_to_hsv(im) for im in images])
			else:
				return np.array([rgb_to_hsv(im) for im in images])
		if color_space == 'CMYK':
			return np.array([hsv_to_cmyk(im) for im in images])

		if color_space == 'RGB':
			if images[0].shape[2] == 4:
				return np.array([cmyk_to_rgb(im) for im in images])
			else:
				return np.array([hsv_to_rgb(im) for im in images])

		return images
	# except:
	# 	print('Cannot convert images with shape' + str(images[0].shape))
	# 	print('To representation ' + str(expected_shape) + ' ' + color_space)
	# 	exit()

def rgb_to_cmyk(im):
	bgrdash = im.astype(np.float)/255.

	K = 1 - np.max(bgrdash, axis=2)
	C = (1-bgrdash[...,2] - K)/(1-K)
	M = (1-bgrdash[...,1] - K)/(1-K)
	Y = (1-bgrdash[...,0] - K)/(1-K)

	return (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

def cmyk_to_rgb(im) :
	c = im[:,:,0]
	m = im[:,:,1]
	y = im[:,:,2]
	k = im[:,:,3]

	c=float(c)/100.0
	m=float(m)/100.0
	y=float(y)/100.0
	k=float(k)/100.0
	r = round(255.0 - ((min(1.0, c * (1.0 - k) + k)) * 255.0))
	g = round(255.0 - ((min(1.0, m * (1.0 - k) + k)) * 255.0))
	b = round(255.0 - ((min(1.0, y * (1.0 - k) + k)) * 255.0))
	return (r,g,b)

def rgb_to_hsv(im):
	return rgb2hsv(im)

def hsv_to_rgb(im):
	return hsv2rgb(im)

def cmyk_to_hsv(im):
	im = cmyk_to_rgb(im)
	im = rgb_to_hsv(im)
	return im

def hsv_to_cmyk(im):
	im = hsv_to_rgb(im)
	im = rgb_to_cmyk(im)
	return im

def gaussian1d(sigma, mean, x, ord):
	x = np.array(x)
	x_ = x - mean
	var = sigma**2

	# Gaussian Function
	g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))
	
	if ord == 0:
		g = g1
		return g
	elif ord == 1:
		g = -g1*((x_)/(var))
		return g
	else:
		g = g1*(((x_*x_) - var)/(var**2))
		return g

def gaussian1d(sigma, mean, x, ord):
	x = np.array(x)
	x_ = x - mean
	var = sigma**2

	# Gaussian Function
	g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))
	
	if ord == 0:
		g = g1
		return g
	elif ord == 1:
		g = -g1*((x_)/(var))
		return g
	else:
		g = g1*(((x_*x_) - var)/(var**2))
		return g

def gaussian2d(sup, scales):
	var = scales * scales
	shape = (sup,sup)
	n,m = [(i - 1)/2 for i in shape]
	x,y = np.ogrid[-m:m+1,-n:n+1]
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	return g

def log2d(sup, scales):
	var = scales * scales
	shape = (sup,sup)
	n,m = [(i - 1)/2 for i in shape]
	x,y = np.ogrid[-m:m+1,-n:n+1]
	g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
	h = g*((x*x + y*y) - var)/(var**2)
	return h

def makefilter(scale, phasex, phasey, pts, sup):

	gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
	gy = gaussian1d(scale,   0, pts[1,...], phasey)

	image = gx*gy

	image = np.reshape(image,(sup,sup))
	return image

def make_lm_filters():
	sup	 = 49
	scalex  = np.sqrt(2) * np.array([1,2,3])
	norient = 6
	nrotinv = 12

	nbar  = len(scalex)*norient
	nedge = len(scalex)*norient
	nf	= nbar+nedge+nrotinv
	F	 = np.zeros([sup,sup,nf])
	hsup  = (sup - 1)/2

	x = [np.arange(-hsup,hsup+1)]
	y = [np.arange(-hsup,hsup+1)]

	[x,y] = np.meshgrid(x,y)

	orgpts = [x.flatten(), y.flatten()]
	orgpts = np.array(orgpts)
	
	count = 0
	for scale in range(len(scalex)):
		for orient in range(norient):
			angle = (np.pi * orient)/norient
			c = np.cos(angle)
			s = np.sin(angle)
			rotpts = [[c+0,-s+0],[s+0,c+0]]
			rotpts = np.array(rotpts)
			rotpts = np.dot(rotpts,orgpts)
			F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
			F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
			count = count + 1
			
	count = nbar+nedge
	scales = np.sqrt(2) * np.array([1,2,3,4])
	
	for i in range(len(scales)):
		F[:,:,count]   = gaussian2d(sup, scales[i])
		count = count + 1
		
	for i in range(len(scales)):
		F[:,:,count] = log2d(sup, scales[i])
		count = count + 1
		
	for i in range(len(scales)):
		F[:,:,count] = log2d(sup, 3*scales[i])
		count = count + 1
		
	return F


def apply_lm_filter(im, lm_filters):
	filtered = []
	for i in range(lm_filters.shape[2]):
		filtered.append(scipy.ndimage.correlate(im,lm_filters[:,:,i], mode='constant'))
	filtered = np.asarray(filtered)
	res = np.sum(filtered,axis=0)
	return res



# ====
# ====
# ==== REPRESENTATION 
# ====
# ====

def get_nn_features(nn_type, images):
	feature_extractor = None
	preprocess_input = None
	input_shape = images[0].shape

	if nn_type == 'resnet50':
		feature_extractor = tf.keras.applications.ResNet50(
			weights="imagenet",
			include_top=False,
			pooling="avg",
			input_shape=input_shape
		)
		preprocess_input = tf.keras.applications.resnet.preprocess_input

	if nn_type == 'inceptionv3':
		feature_extractor = tf.keras.applications.InceptionV3(
			weights="imagenet",
			include_top=False,
			pooling="avg",
			input_shape=input_shape
		)
		preprocess_input = tf.keras.applications.inception_v3.preprocess_input

	if nn_type == 'densenet121':
		feature_extractor = tf.keras.applications.DenseNet121(
			weights="imagenet",
			include_top=False,
			pooling="avg",
			input_shape=input_shape
		)
		preprocess_input = tf.keras.applications.densenet.preprocess_input

	if nn_type == 'xception':
		feature_extractor = tf.keras.applications.Xception(
			weights="imagenet",
			include_top=False,
			pooling="avg",
			input_shape=input_shape
		)
		preprocess_input = tf.keras.applications.xception.preprocess_input
		
	if nn_type == 'vgg16':
		feature_extractor = tf.keras.applications.VGG16(
			weights="imagenet",
			include_top=False,
			pooling="avg",
			input_shape=input_shape
		)
		preprocess_input = tf.keras.applications.vgg16.preprocess_input

	inputs = keras.Input(input_shape)
	preprocessed = preprocess_input(inputs)
	outputs = feature_extractor(preprocessed)
	model = keras.Model(inputs, outputs, name="feature_extractor")
	return model.predict(np.array(images), verbose=0)

# ====
# ====
# ==== MORPHOLOGICAL PROCESSING
# ====
# ====

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __repr__(self):
		return '({0},{1})'.format(self.x, self.y)
	def __str__(self):
		return '({0},{1})'.format(self.x, self.y)
	def __lt__(self, other):
		return self.y < other.y if self.x == other.x else self.x < other.x
	def __eq__(self, other):
		return self.x == other.x and self.y == other.y
	def __ne__(self, other):
		return not self == other
	def __hash__(self):
		return hash((self.x, self.y))

	def dist(self, other):
		return hypot(other.x - self.x, other.y - self.y)
	def orientation(self, q, r):
		''' Returns positive number if p-q-r are clockwise, neg if ccw, 0 if collinear'''
		return (q.y - self.y) * (r.x - self.x) - (q.x - self.x) * (r.y - self.y)

def graham_scan(pts): # O(n log n)
	''' Computes the upper and lower hulls for a given set of points and chains them together for a covex hull '''
	u = []
	l = []
	pts = sorted(pts) #O(n log n) time to sort
	for pt in pts:
		# Remove all collinear and clockwise turns from upper hull
		while len(u) > 1 and pt.orientation(u[-2], u[-1]) <= 0:
			u.pop()
		# Remove all collinear and counter clockwise turns lower hull
		while len(l) > 1 and pt.orientation(l[-2], l[-1]) >= 0:
			l.pop()
		u.append(pt)
		l.append(pt)
	# Concat the upper and lower hulls in order
	return u+[l[i] for i in range(len(l)- 2, 0, -1)]

def _next_hull_pt(pt, pts):
	q = pt
	for r in pts:
		orientation = pt.orientation(q, r)
		if (orientation < 0 or orientation == 0 and pt.dist(r) > pt.dist(q)):
			q = r
	return q 

def get_convex_hull(pts): # O(nh) n = # pts, h = # of pts on hull
	''' Algorithm to compute the Convex Hull by computing on point on the hull at a time '''
	pts = np.array(pts)
	# print(pts.shape)
	pts = pts.reshape((math.floor((pts.shape[0] * pts.shape[1])/2),2))
	pts = [Point(p[0],p[1]) for p in pts]
	# print(len(pts))
	hull = [min(pts)] # Get the far left point, O(n)
	for pt in hull:
		# print(pt)
		q = _next_hull_pt(pt, pts)
		if q != hull[0]:
			hull.append(q)
	return hull


class UnionFindOpt:
	"""
	Optimized version of union find that
	implements path compression and
	weight-based merging
	"""
	
	def __init__(self,N):
		self.N = N
		self.parents = []
		self.weights = []
		self._operations = 0
		self._calls = 0
		
		for i in range(N):
			self.parents.append(i)
			self.weights.append(1)
	
	def root(self,i):
		self._calls += 1
		try:
			if i != self.parents[i % len(self.parents)]:
				self._operations += 1
				self.parents[i] = self.root(self.parents[i])	
		except:
			i = len(self.parents) - 1
			if i != self.parents[i]:
				self._operations += 1
				self.parents[i] = self.root(self.parents[i])

		return self.parents[i]
	
	def find(self,i,j):
		self._calls +=1
		return self.root(i) == self.root(j)
	
	
		
	
	def union(self,i,j):
		self._calls +=1
		root_i = self.root(i)
		root_j = self.root(j)
		
		if self.weights[root_i] < self.weights[root_j]:
			self._operations += 1
			self.parents[root_i] = self.parents[root_j]
			self.weights[root_j] += self.weights[root_i]
		else:
			self._operations +=1
			self.parents[root_j] = self.parents[root_i]
			self.weights[root_i] += self.weights[root_j]
			
   

def load_cells_grayscale(filename, n_pixels = 0):
	"""
	Load in a grayscale image of the cells, where 1 is maximum brightness
	and 0 is minimum brightness
	Parameters
	----------
	filename: string
		Path to image holding the cells
	n_pixels: int
		Number of pixels in the image
	
	Returns
	-------
	ndarray(N, N)
		A square grayscale image
	"""
	I = plt.imread(filename)
	cells_gray = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
	# Denoise a bit with a uniform filter
	cells_gray = ndimage.uniform_filter(cells_gray, size=10)
	cells_gray = cells_gray - np.min(cells_gray)
	cells_gray = cells_gray/np.max(cells_gray)
	N = int(np.sqrt(n_pixels))
	if n_pixels > 0:
		# Resize to a square image
		cells_gray = misc.imresize(cells_gray, (N, N))
	return cells_gray


def permute_union(labels):
	"""
	Shuffle around labels by raising them to a prime and
	modding by a large-ish prime, so that cells are easier
	to see against their backround
	Parameters
	----------
	labels: ndarray(M, N)
		An array of labels for the pixels in the image
	Returns
	-------
	labels_shuffled: ndarray(M, N)
		A new image where the labels are different but still
		the same within connected components
	"""
	return (labels**31) % 833


def get_cluster_centers(labels):
	"""
	Parameters
	----------
	labels : list
		table that holds the unionfind
		roots of each pixel
	Returns
	-------
	list_matches : list
		list that holds the average
		pixels of each cluster
	"""
	#initialize an empty list for each pixel in image
	list_labels = [[] for _ in range((len(labels))**2)]
	#loop through pixels in image
	
	for i in range(len(labels)):
		for j in range(len(labels)):
			
			#add coordinates of pixel to the appropriate indexed list
			#root of pixels will be index in list_labels table and convert back to pixel rather than root
			
			list_labels[int(labels[i][j])].append([i, j])
			
	#get the average of each cell (each group of pixels with same root)
	list_matches = []
	for i in range(len(list_labels)):
		if len(list_labels[i]) > 1:
			
			total_x = 0
			total_y = 0
			counter = 0
			
			for j in range(len(list_labels[i])):
				total_x += list_labels[i][j][0]
				total_y += list_labels[i][j][1]
				counter += 1
				
			x = total_x//counter
			y = total_y//counter
			avg_pixel = (x, y)
			list_matches.append(avg_pixel)
			
	return list_matches
	

def get_union(image,thresh):
	"""
	Cluster grayscale pixels together by 
	unioning adjacent pixels that meet the brightness threshold
	Parameters
	----------
	image : list
		table of pixel grayscale values
	thresh : float
		minimum brightness that we're looking for
	Returns
	-------
	labels : list
		table that holds the unionfind 
		roots of each pixel
	"""
	
	#initialize list of pixels and create 2d array of labels based on image size
	#create union find object with number of pixels as size

	matches = UnionFindOpt(len(image)**2)
	labels = np.zeros((len(image), len(image)))
	
	for i in range(len(image) - 1):
		for j in range(len(image[0]) - 1):
			#if a pixel and its adjacent pixel are bright enough, union them together and add their root to the labels list
			if image[i][j] > thresh and image[i + 1][j] > thresh and i < len(image) - 1:
				
				#len(image) = 400, so i * 400 translates proper row of 2d array to 1d for union find
				#adding j is the column offset
				
				matches.union(i*len(image)+j, (i+1)*len(image)+j)
			
			#if a pixel and the other adjacent pixel are bright enough, union them together
			if image[i][j] > thresh and image[i][j + 1] > thresh and j < len(image) - 1:
				matches.union(i*len(image)+j, i*len(image)+j+1)
	
	for i in range(len(image)):
		for j in range(len(image)):
			#add the root of each pixel to the labels list to create 'clusters'
			labels[i][j] = matches.root(j + len(image) * i)
				
	return labels
	

# ====
# ====
# ==== SEGMENTATION 
# ====
# ====

def spec_cluster(tup):
	im = tup[0]
	args = tup[1]
	smoothened_im = gaussian_filter(im, sigma=args['sigma'])
	rescaled_im = rescale(smoothened_im, 0.3, anti_aliasing=False)
	graph = skfei.img_to_graph(rescaled_im)
	beta = args['beta']
	eps = 1e-6
	graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
	n_regions = args['n_regions']

	clust = spectral_clustering(
				graph,
				n_clusters=(n_regions + 3),
				eigen_tol=1e-7,
				assign_labels=args['assign_labels'],
				n_init=1
			)

	clust = clust.reshape(rescaled_im.shape)

	return resize(clust, im.shape)

# ====
# ====
# ==== FITNESS METRICS 
# ====
# ====

def get_fitness_metric(metric, gt, images):

	return ari(gt, images)

def get_overlap(image, image_val, gt, label):
	overlap = 0

	for x in range(0, gt.shape[0]):
		for y in range(0, gt.shape[1]):
			if gt[x][y] == label and image[x][x] == image_val:
				overlap += 1
	
	return overlap

def introspectability(gt, images):
	gt = np.array([[[sum(pixel) for pixel in row] for row in im] for im in gt])

	if 'bool' in str(type(images[0][0][0])):
		if len(images.shape) == 3:
			images = np.array([[[1 if pixel == True else 0 for pixel in row] for row in im] for im in images])
	else:
		if len(images.shape) == 4:
			images = np.array([[[sum(pixel) for pixel in row] for row in im] for im in images])

	total_introspectability = []
	for idx, image in enumerate(images):
		unique_values_in_images = list(np.unique(image))
		unique_values_in_gt = list(np.unique(gt[idx]))
		total_overlap = [[] for val in unique_values_in_images] 

		for i, x in enumerate(unique_values_in_images):
			gt_overlap = []
			for y in unique_values_in_gt:
				gt_overlap.append(get_overlap(image, x, gt[idx], y))
				
			total_overlap[i] = gt_overlap

		reshaped_total_overlap = [[aa[i] for ii, aa in enumerate(total_overlap)] for i, a in enumerate(total_overlap[0])]
		multiplied_total_overlap = [(1/len(unique_values_in_images)) * sum(o) for o in reshaped_total_overlap]

		Nc_2 = 0
		total_distance = 0

		for ix, x in enumerate(unique_values_in_gt):
			for iy, y in enumerate(unique_values_in_gt):
				if iy + 1 < len(unique_values_in_gt):
					total_distance += abs(multiplied_total_overlap[ix] - multiplied_total_overlap[iy + 1])

				if x != y:
					Nc_2 += 1

		total_introspectability.append(Nc_2 * total_distance)
	
	return np.average(np.array(total_introspectability))
	

def ari(gt, images):
	fitness = -1
	gt = np.array(gt)
	images = np.array(images)
	
	if len(list(np.unique(images))) == 1:
		fitness = 0.0
	
	og_shape = (gt.shape[1], gt.shape[2])

	if len(gt[0].shape) > 2:
		gt = np.array([[[sum(pixel) for pixel in row] for row in im] for im in gt])

	if 'bool' in str(type(images[0][0][0])):
		if len(images.shape) == 3:
			images = np.array([[[1 if pixel == True else 0 for pixel in row] for row in im] for im in images])
	else:
		if len(images.shape) == 4:
			images = np.array([[[sum(pixel) for pixel in row] for row in im] for im in images])

	if np.average(images[0]) < 1:
		images = images * 255

	if len(images[0].shape) > 2:
		images = np.array([[[sum(pixel) for pixel in row] for row in im] for im in images])

	gt = np.array([np.array(gt[idx]).astype(np.uint8).flatten() for idx, im in enumerate(gt)])
	images = np.array([np.array(im).astype(np.uint8).flatten() for idx, im in enumerate(images)])
	if fitness == -1:
		fitness = np.mean(np.array([adjusted_rand_score(np.array(gt[idx]).astype(np.uint8).flatten(), np.array(im).astype(np.uint8).flatten()) for idx, im in enumerate(images)]))

	# if fitness > 0.5:
	# 	im1 = Image.fromarray(colorize(images[0].reshape(og_shape)))
	# 	im2 = Image.fromarray(colorize(gt[0].reshape(og_shape)))

	# 	im1.save('images_' + str(int(time.time())) + '.png')
	# 	im2.save('gt_' + str(int(time.time())) + '.png')

	return fitness, gt, images

def colorize(im):
	colors = [[11, 115, 234], [200, 178, 17], [108, 185, 47], [178, 46, 244], [212, 64, 222], [45, 132, 199], [149, 240, 36], [186, 178, 92], [153, 155, 31], [156, 230, 158], [216, 87, 99], [85, 20, 232], [127, 58, 81], [34, 7, 121], [210, 57, 241], [160, 33, 209], [2, 229, 26], [132, 123, 92], [173, 119, 26], [114, 194, 151], [31, 132, 250], [5, 23, 232], [91, 255, 89], [209, 25, 79], [133, 185, 20], [47, 66, 24], [54, 151, 9], [115, 121, 22], [224, 165, 186], [18, 84, 61], [174, 19, 140], [87, 53, 131], [11, 76, 99], [13, 224, 139], [117, 192, 180], [70, 227, 83], [148, 132, 71], [214, 140, 44], [29, 181, 47], [197, 212, 62], [16, 76, 198], [15, 240, 179], [95, 95, 51], [229, 113, 127], [225, 41, 208], [171, 131, 30], [9, 13, 90], [185, 28, 119], [205, 241, 172], [216, 188, 68], [124, 196, 110], [5, 201, 216], [91, 16, 219], [76, 36, 53], [108, 62, 162], [213, 149, 84], [7, 144, 140], [223, 37, 221], [136, 172, 175], [55, 69, 133], [20, 156, 181], [58, 86, 188], [69, 142, 35], [236, 5, 108], [206, 186, 93], [28, 198, 65], [175, 236, 97], [146, 87, 69], [42, 186, 138], [53, 177, 131], [39, 102, 43], [205, 218, 52], [204, 215, 91], [99, 58, 134], [184, 135, 73], [144, 34, 40], [143, 152, 212], [196, 97, 60], [242, 169, 117], [165, 98, 134], [176, 86, 25], [196, 106, 0], [215, 122, 99], [46, 33, 71], [103, 49, 185], [191, 156, 12], [247, 55, 64], [28, 105, 170], [182, 202, 140], [185, 127, 33], [191, 222, 196], [244, 109, 224], [144, 128, 175], [154, 49, 75], [94, 222, 234], [148, 126, 144], [54, 27, 175], [14, 116, 171], [22, 43, 86], [39, 236, 144], [106, 212, 72], [48, 174, 0], [70, 137, 123], [82, 19, 234], [5, 23, 189], [200, 242, 156], [124, 203, 19], [245, 118, 55], [30, 188, 87], [19, 29, 124], [212, 144, 243], [95, 128, 7], [18, 214, 219], [56, 199, 121], [61, 151, 21], [197, 113, 198], [83, 86, 197], [218, 232, 249], [140, 50, 114], [231, 132, 193], [105, 206, 195], [81, 75, 134], [210, 160, 22], [134, 49, 248], [48, 232, 121], [228, 203, 161], [249, 101, 244], [164, 79, 164], [198, 92, 240], [128, 181, 152], [114, 41, 52], [214, 192, 249], [228, 76, 73], [59, 140, 235], [14, 152, 16], [147, 138, 67], [13, 84, 208], [146, 118, 211], [172, 97, 18], [6, 176, 245], [133, 152, 251], [173, 162, 89], [99, 185, 35], [76, 63, 220], [210, 79, 82], [195, 140, 146], [188, 216, 16], [106, 156, 181], [223, 56, 136], [3, 82, 250], [227, 210, 104], [4, 29, 13], [239, 37, 42], [16, 227, 120], [51, 228, 188], [239, 34, 151], [55, 10, 183], [36, 92, 202], [31, 52, 217], [78, 29, 204], [253, 162, 115], [11, 247, 0], [45, 25, 163], [128, 240, 154], [67, 101, 133], [228, 63, 103], [231, 112, 118], [165, 12, 216], [220, 255, 24], [148, 200, 176], [16, 144, 219], [215, 67, 202], [173, 219, 92], [251, 170, 64], [35, 29, 165], [117, 154, 91], [208, 8, 152], [19, 27, 17], [253, 107, 58], [196, 20, 32], [203, 6, 122], [247, 215, 26], [109, 193, 61], [171, 7, 30], [67, 136, 5], [143, 216, 121], [139, 127, 188], [223, 195, 68], [250, 176, 195], [58, 30, 60], [51, 169, 66], [22, 167, 14], [87, 96, 148], [96, 135, 94], [82, 26, 152], [145, 22, 55], [205, 184, 206], [54, 248, 214], [6, 83, 118], [151, 118, 160], [163, 35, 139], [103, 197, 33], [124, 185, 221], [148, 232, 207], [138, 185, 189], [38, 2, 126], [147, 106, 35], [78, 193, 104], [154, 33, 6], [114, 0, 153], [151, 115, 63], [20, 177, 235], [71, 125, 143], [62, 209, 179], [28, 92, 138], [180, 22, 174], [253, 58, 22], [88, 192, 193], [31, 42, 32], [222, 90, 21], [219, 181, 56], [251, 72, 70], [86, 27, 182], [8, 0, 32], [131, 125, 249], [195, 206, 48], [115, 170, 170], [16, 150, 155], [116, 222, 119], [63, 26, 118], [108, 203, 196], [249, 28, 34], [33, 134, 62], [220, 43, 73], [95, 138, 127], [129, 3, 118], [29, 1, 161], [91, 16, 62], [93, 201, 109], [24, 106, 40], [193, 96, 159], [170, 147, 138], [202, 137, 175], [43, 229, 138], [162, 62, 14], [73, 55, 140], [21, 251, 84], [110, 100, 208], [151, 241, 111], [135, 50, 192], [189, 242, 141], [104, 47, 83], [229, 38, 195], [131, 185, 215], [46, 0, 7], [20, 32, 135], [62, 250, 227], [104, 171, 37], [178, 103, 143], [236, 176, 166], [81, 64, 143], [37, 99, 215], [14, 200, 40], [115, 54, 55], [218, 122, 114], [112, 109, 243], [70, 125, 173], [45, 185, 0], [127, 188, 15], [200, 247, 190], [25, 193, 97], [139, 220, 124], [211, 188, 62], [128, 19, 81], [68, 235, 88], [78, 238, 231], [134, 160, 121], [165, 183, 69], [213, 239, 27], [211, 189, 251], [27, 231, 88], [5, 51, 225], [2, 133, 92], [115, 137, 140], [12, 195, 219], [54, 170, 203], [204, 88, 247], [184, 49, 59], [249, 171, 32], [143, 225, 224], [142, 101, 87], [33, 156, 55], [175, 238, 202], [238, 245, 246], [105, 49, 238], [98, 153, 73], [45, 17, 128], [29, 9, 30], [32, 108, 233], [69, 45, 153]]

	im = np.array([[colors[int(pixel)] for pixel in row] for row in im])

	return im

def infer_cluster_labels(model, actual_labels):
	inferred_labels = {}

	for i in list(set(model)):   
		labels = []
		index = np.where(model == i)
		labels.append(actual_labels[index])

		if len(labels[0]) == 1:
			counts = np.bincount(labels[0])
		else:
			counts = np.bincount(np.squeeze(labels))

		try:
			if np.argmax(counts) in inferred_labels:
				inferred_labels[np.argmax(counts)].append(i)
			else:
				inferred_labels[np.argmax(counts)] = [i]
		except:
			counts = None
		
	return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
	"""
	Determines label for each array, depending on the cluster it has been assigned to.
	returns: predicted labels for each array
	"""
	
	# empty array of len(X)
	predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
	
	for i, cluster in enumerate(X_labels):
		for key, value in cluster_labels.items():
			if cluster in value:
				predicted_labels[i] = key
				
	return predicted_labels


def convert_labels(labels_pred, labels_true):
	try:
		all_labels = list(set(labels_pred))
		labels_pred = np.array([all_labels.index(lp) for lp in labels_pred])
		cluster_labels = infer_cluster_labels(labels_pred, labels_true)
		predicted_labels = infer_data_labels(labels_pred, cluster_labels)
		return predicted_labels
	except:
		return labels_pred

def E(S1,S2,p):
    
    # Define version of image that only retains the set of interest
    R1  = np.zeros(S1.shape).astype(bool)
    R2c = np.zeros(S2.shape).astype(bool)
    R1[np.where(S1 == S1[p[0],p[1]])]  = True # Only keep set that contains the pixel at p
    R2c[np.where(S2 != S2[p[0],p[1]])] = True # Only keep complement of set that contains pixel at p
    
    # Calculate numerator and denominator of Local Error
    num = np.count_nonzero(np.logical_and(R1,R2c))
    den = np.count_nonzero(R1)
    
    # Return Local Error
    return float(num)/den


# https://github.com/DuckDuckPig/CH-ACWE/blob/main/Metrics/consistancyErrorMetrics.py#L42
def GCE(S1,S2):
    
    # Prepare for Loop
    E1 = 0
    E2 = 0
    n  = len(S1) * len(S1[0])
    
    # Calculate and dum local errors
    for i in range(len(S1)):
        for j in range(len(S1[0])):
            E1 += E(S1,S2,[i,j])
            E2 += E(S2,S1,[i,j])
            
    # Return GCE
    return 1.0/n * np.min([E1,E2])

# In[4]:
# Local Consistency Error
def LCE(S1,S2):
    
    # Prepare for Loop
    Eout = 0
    n    = len(S1) * len(S1[0])
    
    # Calculate local error and sum minimum of the two
    for i in range(len(S1)):
        for j in range(len(S1[0])):
            E1 = E(S1,S2,[i,j])
            E2 = E(S2,S1,[i,j])
            Eout += np.min([E1,E2])
            
    # Return LCE
    return 1.0/n * Eout
