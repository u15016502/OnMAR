"""
Segmentation Dataset Handling

This module handles loading and preprocessing of image segmentation datasets.
Datasets: BSD500, COVID, Pascal VOC2012
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from pathlib import Path
import os
from PIL import Image
import scipy.io as sio


class SegmentationDatasetLoader:
    """
    Handles loading and preprocessing of image segmentation datasets.

    Supported datasets:
    - bsd500: Berkeley Segmentation Dataset 500
    - covid: COVID-19 CT scan segmentation
    - pascal: Pascal VOC 2012 segmentation
    """

    def __init__(self, dataset_name: str, data_dir: str = None, random_seed: int = 42):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: Name of the dataset to load (bsd500, covid, pascal)
            data_dir: Directory containing the datasets (defaults to datasets folder)
            random_seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()

        # Default to datasets directory
        if data_dir is None:
            data_dir = Path(__file__).parent
        self.data_dir = Path(data_dir)

        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Dataset configurations
        self.dataset_configs = {
            'bsd500': {
                'path': self.data_dir / 'bsd500',
                'images_subdir': 'images',
                'masks_subdir': 'groundTruth',
                'image_ext': '.jpg',
                'mask_ext': '.mat',
                'has_splits': True
            },
            'covid': {
                'path': self.data_dir / 'covid',
                'images_subdir': 'frames',
                'masks_subdir': 'masks',
                'image_ext': '.png',
                'mask_ext': '.png',
                'has_splits': False
            },
            'pascal': {
                'path': self.data_dir / 'pascal' / 'VOC2012',
                'images_subdir': 'JPEGImages',
                'masks_subdir': 'SegmentationClass',
                'image_ext': '.jpg',
                'mask_ext': '.png',
                'has_splits': False
            }
        }

        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Supported: {list(self.dataset_configs.keys())}")

    def load_dataset(self, val_split: float = 0.15,
                    target_size: Tuple[int, int] = (256, 256)) -> Tuple[List, List, List]:
        """
        Load the dataset and split into train/val/test.

        Args:
            val_split: Fraction of training data to use for validation
            target_size: Target size for images (height, width)

        Returns:
            Tuple of (train_data, val_data, test_data)
            Each is a list of (image, mask) tuples
        """
        config = self.dataset_configs[self.dataset_name]

        if self.dataset_name == 'bsd500':
            return self._load_bsd500(config, val_split, target_size)
        elif self.dataset_name == 'covid':
            return self._load_covid(config, val_split, target_size)
        elif self.dataset_name == 'pascal':
            return self._load_pascal(config, val_split, target_size)
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented")

    def _load_bsd500(self, config: Dict, val_split: float,
                    target_size: Tuple[int, int]) -> Tuple[List, List, List]:
        """Load BSD500 dataset with existing train/val/test splits."""
        train_data = []
        val_data = []
        test_data = []

        base_path = config['path']
        images_dir = base_path / config['images_subdir']
        masks_dir = base_path / config['masks_subdir']

        # Load each split
        for split in ['train', 'val', 'test']:
            split_images_dir = images_dir / split
            split_masks_dir = masks_dir / split

            if not split_images_dir.exists():
                continue

            # Get all image files
            image_files = sorted(split_images_dir.glob(f'*{config["image_ext"]}'))

            for img_path in image_files:
                # Find corresponding mask
                mask_path = split_masks_dir / f"{img_path.stem}{config['mask_ext']}"

                if not mask_path.exists():
                    continue

                # Load image
                image = self._load_image(img_path, target_size)

                # Load mask (BSD500 uses .mat files)
                mask = self._load_bsd500_mask(mask_path, target_size)

                if image is not None and mask is not None:
                    if split == 'train':
                        train_data.append((image, mask))
                    elif split == 'val':
                        val_data.append((image, mask))
                    else:
                        test_data.append((image, mask))

        # If no val split exists, create one from train
        if len(val_data) == 0 and len(train_data) > 0:
            np.random.shuffle(train_data)
            val_size = int(len(train_data) * val_split)
            val_data = train_data[:val_size]
            train_data = train_data[val_size:]

        return train_data, val_data, test_data

    def _load_covid(self, config: Dict, val_split: float,
                   target_size: Tuple[int, int]) -> Tuple[List, List, List]:
        """Load COVID dataset."""
        all_data = []

        base_path = config['path']
        images_dir = base_path / config['images_subdir']
        masks_dir = base_path / config['masks_subdir']

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Get all image files
        image_files = sorted(images_dir.glob(f'*{config["image_ext"]}'))

        for img_path in image_files:
            # Mask has same name
            mask_path = masks_dir / img_path.name

            if not mask_path.exists():
                continue

            # Load image and mask
            image = self._load_image(img_path, target_size)
            mask = self._load_mask(mask_path, target_size)

            if image is not None and mask is not None:
                all_data.append((image, mask))

        # Split into train/val/test
        return self._split_data(all_data, val_split)

    def _load_pascal(self, config: Dict, val_split: float,
                    target_size: Tuple[int, int]) -> Tuple[List, List, List]:
        """Load Pascal VOC 2012 segmentation dataset."""
        all_data = []

        base_path = config['path']
        images_dir = base_path / config['images_subdir']
        masks_dir = base_path / config['masks_subdir']

        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Get all mask files (not all images have segmentation masks)
        mask_files = sorted(masks_dir.glob(f'*{config["mask_ext"]}'))

        for mask_path in mask_files:
            # Find corresponding image
            img_path = images_dir / f"{mask_path.stem}{config['image_ext']}"

            if not img_path.exists():
                continue

            # Load image and mask
            image = self._load_image(img_path, target_size)
            mask = self._load_mask(mask_path, target_size)

            if image is not None and mask is not None:
                all_data.append((image, mask))

        # Split into train/val/test
        return self._split_data(all_data, val_split)

    def _load_image(self, path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load and resize an image."""
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def _load_mask(self, path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load and resize a segmentation mask."""
        try:
            mask = Image.open(path)
            mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
            return np.array(mask, dtype=np.int32)
        except Exception as e:
            print(f"Error loading mask {path}: {e}")
            return None

    def _load_bsd500_mask(self, path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load BSD500 ground truth from .mat file."""
        try:
            mat = sio.loadmat(str(path))
            # BSD500 stores ground truth in 'groundTruth' field
            gt = mat['groundTruth']
            # Use first annotator's segmentation
            segmentation = gt[0, 0]['Segmentation'][0, 0]

            # Resize
            mask_img = Image.fromarray(segmentation.astype(np.uint8))
            mask_img = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
            return np.array(mask_img, dtype=np.int32)
        except Exception as e:
            print(f"Error loading BSD500 mask {path}: {e}")
            return None

    def _split_data(self, data: List, val_split: float,
                   test_split: float = 0.15) -> Tuple[List, List, List]:
        """Split data into train/val/test sets."""
        np.random.shuffle(data)

        n = len(data)
        test_size = int(n * test_split)
        val_size = int(n * val_split)

        test_data = data[:test_size]
        val_data = data[test_size:test_size + val_size]
        train_data = data[test_size + val_size:]

        return train_data, val_data, test_data

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        config = self.dataset_configs[self.dataset_name]

        # Count files
        images_dir = config['path'] / config['images_subdir']

        if config['has_splits']:
            total_images = 0
            for split in ['train', 'val', 'test']:
                split_dir = images_dir / split
                if split_dir.exists():
                    total_images += len(list(split_dir.glob(f'*{config["image_ext"]}')))
        else:
            if images_dir.exists():
                total_images = len(list(images_dir.glob(f'*{config["image_ext"]}')))
            else:
                total_images = 0

        return {
            'name': self.dataset_name,
            'total_images': total_images,
            'channels': 3,
            'height': 256,  # Default target size
            'width': 256
        }


class SegmentationDataLoader:
    """
    Simple data loader for segmentation data that mimics PyTorch DataLoader interface.
    """

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]],
                 batch_size: int = 16, shuffle: bool = True):
        """
        Initialize the data loader.

        Args:
            data: List of (image, mask) tuples
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = list(range(len(data)))

    @property
    def dataset(self):
        """Return data for compatibility."""
        return self.data

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)

        for i in range(0, len(self.data), self.batch_size):
            batch_indices = self._indices[i:i + self.batch_size]
            images = np.stack([self.data[j][0] for j in batch_indices])
            masks = np.stack([self.data[j][1] for j in batch_indices])
            yield images, masks
