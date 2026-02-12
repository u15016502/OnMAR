import math
from scipy.fft import fft2, fftshift
import numpy as np
import cv2
from skimage import data, filters, measure, morphology
from utils import correct_or_return_input, get_nn_features
from skimage.feature import shape_index, hog, blob_dog, blob_log, blob_doh, haar_like_feature, haar_like_feature_coord, draw_haar_like_feature
from skimage.transform import rescale, resize

class representation:

	def __init__(self):
		return

	def get_representation(self, images, labels, representation_type, args):

		if representation_type == 'R1': # canny edge detection
			return np.array([cv2.Canny(im.astype(np.uint8), args['min_val'], args['max_val'], args['aperture_size']) for im in images])

		elif representation_type == 'R2': # shape formula
			def detect_shapes(im, im_gray):
				ret,thresh = cv2.threshold(im_gray.astype(np.uint8), args['thresh'], 255, 0)
				contours,hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

				for i, cnt in enumerate(contours):
					M = cv2.moments(cnt)
					x1, y1 = cnt[0,0]
					im = cv2.drawContours(im.astype(np.uint8), [cnt], -1, (0,255,255), 3)

				return im

			gray_images = correct_or_return_input(images, labels, (0,0), 'gray')
			return np.array([detect_shapes(im, gray_images[idx]) for idx, im in enumerate(images)])

		elif representation_type == 'R3': # hough transform 
			def draw_hough_transform(im):
				lines = cv2.HoughLines(im.astype(np.uint8), args['rho'], args['theta'], args['threshold'])
				if 'None' in str(type(lines)):
					return im
				for x in range(0, len(lines)):
					for rho,theta in lines[0]:
						a = np.cos(theta)
						b = np.sin(theta)
						x0 = a*rho
						y0 = b*rho
						x1 = int(x0 + 1000*(-b))
						y1 = int(y0 + 1000*(a))
						x2 = int(x0 - 1000*(-b))
						y2 = int(y0 - 1000*(a))

					cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)
				return im

			images = correct_or_return_input(images, labels, (0,0), 'gray')
			im_edges = np.array([cv2.Canny(im.astype(np.uint8), args['min_val'], args['max_val'], args['aperture_size']) for im in images])
			return np.array([
				draw_hough_transform(im)
				for im in im_edges
			])
		elif representation_type == 'R4': # find contour
			def find_contours(im, im_gray):
				ret,thresh = cv2.threshold(im_gray.astype(np.uint8), args['thresh'], 255, 0)
				contours,hierarchy = cv2.findContours(thresh.astype(np.uint8), args['mode'] ,cv2.CHAIN_APPROX_SIMPLE)

				if im.max() > 0:
					im = im * (255.0/im.max())

				img1 = cv2.drawContours(im.astype(np.uint8), contours, -1, (0,255,0), 3)
				return img1

			gray_images = correct_or_return_input(images, labels, (0,0), 'gray')
			return np.array([find_contours(im, gray_images[idx]) for idx, im in enumerate(images)])

		# elif representation_type == 'R6': # neural network
		# 	return get_nn_features(args['nn_type'], images)

		# elif representation_type == 'R7': # fourier transform
		# 	return np.array([np.abs(fftshift(fft2(im))) for im in correct_or_return_input(images, labels, (0,0), 'gray')])

		elif representation_type == 'R5': # shape index
			rep = np.array([shape_index(im) for im in correct_or_return_input(images, labels, (0,0), 'gray')])
			return rep
		# https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.shape_index

		elif representation_type == 'R6': # hog
			if len(images[0].shape) == 2:
				channel_axis = None
			else:
				channel_axis = -1

			return np.array([hog(im, 
				orientations=args['orientation'], 
				pixels_per_cell=(args['pixels_per_cell'], args['pixels_per_cell']), 
				cells_per_block=(args['cells_per_block'],args['cells_per_block']),
				channel_axis=channel_axis, visualize=True)[1] for im in images])

		# https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
		elif representation_type == 'R7': # haar like
			images = correct_or_return_input(images, labels, (0,0), 'gray')
			feature_coord, _ = haar_like_feature_coord(width=math.floor(images[0].shape[1] / 10), height=math.floor(images[0].shape[0] / 10), feature_type=args['feature_type'])
			return np.array([draw_haar_like_feature(im, 0, 0, im.shape[0], im.shape[1], feature_coord=feature_coord, max_n_features=1, alpha=0) for im in images])

		# https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.haar_like_feature
		elif representation_type == 'R8': # blob
			def draw_blob(im, blobs):
				im_ = rescale(im, 0.3, anti_aliasing=False)
				for b in blobs:
					x, y, radius = b
					im_ = cv2.circle(im_.astype(np.uint8), (int(x), int(y)), int(radius), (255,255,255), 4)
				return resize(im_, im.shape)

			if args['type'] == 'log':
				arr = np.array([draw_blob(im, blob_log(im, threshold=args['threshold'], max_sigma=args['max_sigma'], num_sigma=math.floor(args['max_sigma']/3))) for im in correct_or_return_input(images, labels, (0,0), 'gray')])
			if args['type'] == 'dog':
				arr = np.array([draw_blob(im, blob_dog(im, threshold=args['threshold'], max_sigma=args['max_sigma'])) for im in correct_or_return_input(images, labels, (0,0), 'gray')])
			if args['type'] == 'doh':
				arr = np.array([draw_blob(im, blob_doh(im, threshold=args['threshold'], max_sigma=args['max_sigma']))for im in correct_or_return_input(images, labels, (0,0), 'gray')])

			return arr
		# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py
