import math
from skimage.transform import rescale, resize
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image, ImageEnhance
import cv2
import scipy.ndimage as spim
from .utils import correct_or_return_input, make_lm_filters, apply_lm_filter
from skimage import graph
from skimage.transform import rescale, resize

class filters:

	def __init__(self):
		return

	def filter_images(self, images, labels, filter_type, args):
		if filter_type == 'F1': 	# gaussian filter
			images = correct_or_return_input(images, labels,(1,1))
			return np.array([resize(spim.gaussian_filter(rescale(im, 0.75), sigma=args['sigma']), im.shape) for im in images])

		elif filter_type == 'F2': 	# adaptive histogram equalization
			images = correct_or_return_input(images, labels,(1,1), 'gray')			
			clahe = cv2.createCLAHE(clipLimit=args['clip_limit'], tileGridSize=(args['tile_grid_size'],args['tile_grid_size']))
			return np.array([clahe.apply(im.astype(np.uint8)) for im in images])

		elif filter_type == 'F3': 	# threshold https://docs.opencv2.org/4.x/d7/d4d/tutorial_py_thresholding.html 
			images = correct_or_return_input(images, labels,(1,1), 'gray')
			return np.array([cv2.threshold(im.astype(np.uint8), args['threshold_value'], 255, args['threshold_type'])[1] for im in images])

		elif filter_type == 'F4': 	# grayscale then binary image
			return correct_or_return_input(correct_or_return_input(images, labels,(1,1), 'gray'), (1,1), 'binary')		
			
		elif filter_type == 'F5': 	# LM filter https://github.com/tonyjo/LM_filter_bank_python
			lm_filters = make_lm_filters()

			return np.array([resize(apply_lm_filter(rescale(im, 0.4, anti_aliasing=False), lm_filters), im.shape) for im in correct_or_return_input(images, labels,(0,0), 'gray')])

		elif filter_type == 'F6': 	# Binary image filter bank https://github.com/javad-sheikh/Texture-Segmentation/blob/master/HW4.ipynb
			images = correct_or_return_input(images, labels,(1,1), 'gray')			
			return np.array([local_binary_pattern(im, args['n_points'], args['radius'], args['method']) for im in images])

		elif filter_type == 'F7': 	# NxN median filter
			return np.array([cv2.medianBlur(im.astype(np.uint8), ksize=args['kernel_size']) for im in images])

		elif filter_type == 'F8': 	# grayscale image 0 - 5
			return correct_or_return_input(images, labels,(0,0), 'gray')

		elif filter_type == 'F9': 	# binary image
			return np.array([cv2.threshold(im.astype(np.uint8), args['thresh'], 255, 0)[1] for im in correct_or_return_input(images, labels,(0,0))])

		elif filter_type == 'F10': 	# enhance brightness
			if len(np.array(images).shape) == 3:
				mode = 'P'
			else:
				mode = 'RGB'
			
			return np.array([np.array(ImageEnhance.Brightness(Image.fromarray(im.astype(np.uint8))).enhance(args['factor'])) for im in images])

		elif filter_type == 'F11': 	# change colour space
			return correct_or_return_input(images, labels,(0,0,0), args['color_space'])
	
		elif filter_type == 'F12': # prewitt
			return np.array([spim.prewitt(im) for im in images])

		elif filter_type == 'F13': # sobel
			return np.array([spim.sobel(im) for im in images])


	

