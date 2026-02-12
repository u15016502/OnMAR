from classification import classification
from filters import filters
from representation import representation
from segmentation import segmentation

def evaluate(tup):
	images = tup[0]
	labels = tup[1]
	fitness_metric = tup[2] 
	preprocessing_order = tup[3]
	preprocessing_techniques = tup[4]
	postprocessing_order = tup[5]
	postprocessing_techniques = tup[6]
	segmentation_type = tup[7]
	segmentation_args = tup[8]

	f_ = filters()
	r_ = representation()
	p_ = morphological_processing()
	s_ = segmentation()

	for frp_idx, frp in enumerate(preprocessing_order):
		index = [idx for idx, cpt in enumerate(preprocessing_techniques) if cpt[0] == frp][0]
		images = run_frp(images, labels, (frp, preprocessing_techniques[index][1]), f_, r_, p_)

	images = s_.segment(images, labels, segmentation_type, segmentation_args)

	for frp_idx, frp in enumerate(postprocessing_order):
		index = [idx for idx, cpt in enumerate(postprocessing_techniques) if cpt[0] == frp][0]
		images = run_frp(images, labels, (frp, postprocessing_techniques[index][1]), f_, r_, p_)

	return get_fitness_metric(fitness_metric, labels, images, runtime)

def run_frp(self, images, labels, comp, f_, r_, p_):

	if 'F' in comp[0]:
		return f_.filter_images(images, labels, comp[0], comp[1])
	if 'R' in comp[0]:
		return r_.get_representation(images, labels, comp[0], comp[1])
	if 'P' in comp[0]:
		return p_.process_images(images, labels, comp[0], comp[1])
