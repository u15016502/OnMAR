from copy import deepcopy

def get_images_for_label(label, images, labels):
    _images = []

    for idx, image in enumerate(images):
        if labels[idx] == label:
            _images.append(deepcopy(image))

    return _images

def get_image_indices_for_label(label, images, labels):
    _images = []

    for idx, image in enumerate(images):
        if labels[idx] == label:
            _images.append(deepcopy(idx))

    return _images