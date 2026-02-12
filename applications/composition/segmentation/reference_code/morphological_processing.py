import scipy.ndimage as spim
import tensorflow as tf
import copy
import collections
import scipy
from skimage import img_as_float
from skimage import io, color, morphology
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import cv2
import numpy as np
from utils import correct_or_return_input, get_convex_hull, get_union, permute_union
import math

class morphological_processing:

	def __init__(self):
		return

	def process_images(self, images, labels, processing_type, args):
		if processing_type == 'P1': # vertical area extraction
			def get_vertical(im):
				rows = im.shape[0]
				verticalSize = rows // 30
				verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
				im = cv2.erode(im.astype(np.uint8), verticalStructure)
				im = cv2.dilate(im.astype(np.uint8), verticalStructure)
				return im

			images =  np.array([cv2.threshold(im.astype(np.uint8), args['thresh'], 255, 0)[1] for im in correct_or_return_input(images,  labels,(0,0), 'binary')])
			return np.array([get_vertical(im.astype(np.uint8)) for im in images])

		elif processing_type == 'P2': # horizontal area extraction
			def get_horizontal(im):
				cols = im.shape[1]
				horizontal_size = cols // 30
				horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
				horizontal = cv2.erode(im.astype(np.uint8), horizontalStructure)
				horizontal = cv2.dilate(im.astype(np.uint8), horizontalStructure)
				return im

			images =  np.array([cv2.threshold(im.astype(np.uint8), args['thresh'], 255, 0)[1] for im in correct_or_return_input(images,  labels,(0,0), 'binary')])
			return np.array([get_horizontal(im.astype(np.uint8)) for im in images])

		elif processing_type == 'P3': # thinning 
			def thinning(im):
				image_binary = im < 0.5
				out_skeletonize = morphology.skeletonize(image_binary)
				out_thin = morphology.thin(image_binary)
				return out_thin

			return np.array([img_as_float(im) for im in correct_or_return_input(images,  labels,(0,0), 'gray')])

		elif processing_type == 'P4': # dilation
			def dilate(im):
				kernel = cv2.getStructuringElement(args['shape'], (args['size'],args['size']))
				return cv2.dilate(im.astype('uint8'), kernel, iterations=args['iterations'])

			return np.array([dilate(im) for im in images])

		elif processing_type == 'P5': # erosion
			def erode(im):
				kernel = cv2.getStructuringElement(args['shape'], (args['size'],args['size']))
				
				return cv2.erode(im.astype('uint8'), kernel, iterations=args['iterations'])

			results =  np.array([erode(im) for im in images])

			return results

		elif processing_type == 'P6': # opening
			def open_(im):
				kernel = cv2.getStructuringElement(args['shape'], (args['size'],args['size']))
				return cv2.morphologyEx(im.astype('uint8'), cv2.MORPH_OPEN, kernel)

			return np.array([open_(im) for im in images])

		elif processing_type == 'P7': # closing
			def close_(im):
				kernel = cv2.getStructuringElement(args['shape'], (args['size'],args['size']))
				return cv2.morphologyEx(im.astype('uint8'), cv2.MORPH_CLOSE, kernel)

			return np.array([close_(im) for im in images])

		elif processing_type == 'P8': # union find algorithm
			def union(im):
				thresh = args['thresh']
				im_union = get_union(im, thresh)
				im_union = permute_union(im_union)

				im_union = np.expand_dims(im_union, axis=-1)
				im_union = np.repeat(im_union, 3, axis=-1)
				im_union = im_union.astype('float32') / 255
				return tf.image.resize(im_union, [im.shape[0], im.shape[1]])

			return np.array([union(im) for im in correct_or_return_input(images,  labels, (0,0), 'gray')])

		elif processing_type == 'P9': # convex hull algorithm
			return images # images = correct_or_return_input(images,  labels,(0,0), 'gray') 
			# def resize_ims(im):
			# 	im = np.expand_dims(im, axis=-1)
			# 	im = np.repeat(im, 3, axis=-1)
			# 	im = im.astype('float32') / 255
			# 	return tf.image.resize(im, [math.floor(im.shape[0]/2), math.floor(im.shape[1]/2)]) 
			# images = [resize_ims(im) for im in images]
			# return np.array([get_convex_hull(im) for im in images])
		
		elif processing_type == 'P10': # laplace
			
			return np.array([scipy.ndimage.filters.laplace(im) for im in images])

		elif processing_type == 'P11': # fill holes
			processed_images =  np.array([scipy.ndimage.binary_fill_holes(im, structure=np.ones((args['kernel'],args['kernel']))) for im in correct_or_return_input(images,  labels,(0,0), 'binary')])

			if 'bool' in str(type(processed_images[0][0][0])):
				if len(processed_images.shape) == 3:
					processed_images = np.array([[[1 if pixel == True else 0 for pixel in row] for row in im] for im in processed_images])

			return processed_images

		elif processing_type == 'P12': 	# top hat filter
			images = correct_or_return_input(images, labels,(1,1), 'gray')	
			return np.array([spim.white_tophat(im, size=(args['size'], args['size'])) for im in images])

		elif processing_type == 'P13':	# bottom hat filter
			images = correct_or_return_input(images, labels,(1,1), 'gray')				
			return np.array([spim.black_tophat(im, size=(args['size'], args['size'])) for im in images])