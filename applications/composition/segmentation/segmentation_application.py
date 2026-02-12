"""
Image Segmentation Composition Application

Automated composition of unsupervised image segmentation algorithms.
Uses a genetic algorithm to evolve variable-length chromosomes representing
sequences of algorithmic components (FIL, MOR, FD, CIS, CG).
Design space based on Appendix A.2 of the thesis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from applications.base_application import BaseApplication
from datasets.segmentation_datasets import SegmentationDatasetLoader, SegmentationDataLoader
from applications.composition.segmentation.meta_features import SegmentationMetaFeatureExtractor

# Import reference code modules for segmentation pipeline
from applications.composition.segmentation.reference_code.segmentation import segmentation
from applications.composition.segmentation.reference_code.utils import ari


class SegmentationCompositionApplication(BaseApplication):
    """
    Image segmentation composition application for automated algorithm design.
    Uses GA to evolve variable-length chromosomes of segmentation components.
    """

    def __init__(self, dataset_name: str, random_seed: int = 42, n_jobs: int = None):
        """
        Initialize the segmentation composition application.

        Args:
            dataset_name: Name of the dataset (bsd500, covid, pascal)
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel workers for GA fitness evaluation (None = CPU count)
        """
        super().__init__(dataset_name, random_seed)

        # Initialize dataset loader
        self.dataset_loader = SegmentationDatasetLoader(dataset_name, random_seed=random_seed)
        self.dataset_info = self.dataset_loader.get_dataset_info()

        # Data loaders (initialized in load_data)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Raw data (for actual segmentation evaluation)
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Current best design
        self.current_design = None

        # Meta-feature extractor
        self.meta_feature_extractor = SegmentationMetaFeatureExtractor()

        # Parallelization
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)

        # Set random seed
        np.random.seed(random_seed)

    def load_data(self) -> None:
        """Load and prepare the dataset."""
        # Load raw data (list of (image, mask) tuples)
        self.train_data, self.val_data, self.test_data = self.dataset_loader.load_dataset(
            val_split=0.15,
            target_size=(256, 256)
        )

        # Create data loaders
        self.train_loader = SegmentationDataLoader(self.train_data, batch_size=16, shuffle=True)
        self.val_loader = SegmentationDataLoader(self.val_data, batch_size=16, shuffle=False)
        self.test_loader = SegmentationDataLoader(self.test_data, batch_size=16, shuffle=False)

        print(f"Loaded {self.dataset_name}: "
              f"Train={len(self.train_data)}, "
              f"Val={len(self.val_data)}, "
              f"Test={len(self.test_data)}")

    def get_design_space(self) -> Dict[str, List[Any]]:
        """
        Get the design space for image segmentation composition.

        Returns design space from Appendix A.2:
        - Filtering (FIL): 8 options
        - Morphology (MOR): 12 options
        - Feature Detection (FD): 14 options
        - Image-specific Clustering (CIS): 6 options
        - Generic Clustering (CG): 10 options

        Each gene in the chromosome is represented as a string like 'FIL_1', 'MOR_5', etc.
        """
        return {
            'FIL': [
                'gaussian_filter',           # 1
                'adaptive_histogram_eq',     # 2
                'thresholding',              # 3
                'convert_to_binary',         # 4
                'leung_malik_filter',        # 5
                'median_blur',               # 6
                'brightness_enhance',        # 7
                'color_space_change'         # 8
            ],
            'MOR': [
                'vertical_area_extraction',  # 1
                'horizontal_area_extraction',# 2
                'skeletonization',           # 3
                'dilation',                  # 4
                'erosion',                   # 5
                'opening',                   # 6
                'closing',                   # 7
                'union_find',                # 8
                'convex_hull',               # 9
                'hole_filling',              # 10
                'top_hat',                   # 11
                'bottom_hat'                 # 12
            ],
            'FD': [
                'hough_transform',           # 1
                'prewitt',                   # 2
                'sobel',                     # 3
                'canny_edge',                # 4
                'haar',                      # 5
                'laplacian',                 # 6
                'det_of_hessian',            # 7
                'diff_of_gaussian',          # 8
                'laplacian_of_gaussian',     # 9
                'shape_index',               # 10
                'hog',                       # 11
                'shape_formula',             # 12
                'contour_finding',           # 13
                'local_binary_pattern'       # 14
            ],
            'CIS': [
                'slic_superpixels',          # 1
                'chan_vese',                 # 2
                'felsenszwalb',              # 3
                'quickshift',                # 4
                'gabor_filter',              # 5
                'watershed'                  # 6
            ],
            'CG': [
                'kmeans_plusplus',           # 1
                'affinity_propagation',      # 2
                'mean_shift',                # 3
                'spectral_clustering',       # 4
                'agglomerative',             # 5
                'dbscan',                    # 6
                'optics',                    # 7
                'gaussian_mixtures',         # 8
                'birch',                     # 9
                'bisecting_kmeans'           # 10
            ]
        }

    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train/evolve segmentation algorithms with the given design using GA.

        Args:
            design: Dictionary specifying a chromosome (sequence of algorithmic components)
                   Format: {'chromosome': ['FIL_1', 'MOR_5', 'FD_3', 'CIS_2']}
            timesteps: Number of GA generations (default: 60)

        Returns:
            Dictionary of performance metrics including IoU
        """
        if not self.validate_design(design):
            raise ValueError(f"Invalid design: {design}")

        # Set default timesteps (GA generations)
        if timesteps is None:
            timesteps = 60

        # GA parameters from Chapter 7, Section 7.4.4
        population_size = 60
        tournament_size = 3
        crossover_rate = 0.75
        mutation_rate = 0.25

        # Start timing
        start_time = time.time()

        # Initialize population with the provided design
        population = self._initialize_population(population_size, design)

        best_fitness = 0.0
        best_individual = None
        fitness_history = []

        # Evolution loop
        for generation in range(timesteps):
            # Evaluate population (PARALLELIZED)
            if self.n_jobs > 1:
                # Parallel fitness evaluation
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    # Submit all individuals for evaluation
                    future_to_idx = {
                        executor.submit(self._evaluate_design_static, ind, self.val_data): i
                        for i, ind in enumerate(population)
                    }

                    # Collect results
                    fitnesses = [None] * len(population)
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            fitnesses[idx] = future.result()
                        except Exception as e:
                            print(f"Warning: Fitness evaluation failed for individual {idx}: {e}")
                            fitnesses[idx] = 0.0  # Assign worst fitness
            else:
                # Sequential evaluation (fallback)
                fitnesses = [self._evaluate_design(ind) for ind in population]

            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            fitness_history.append(best_fitness)

            if (generation + 1) % 10 == 0:
                print(f"Generation [{generation+1}/{timesteps}], "
                      f"Best IoU: {best_fitness:.4f} ({self.n_jobs} workers)")

            # Create next generation
            new_population = []

            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                # Crossover
                if np.random.random() < crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = parent1.copy()

                # Mutation
                if np.random.random() < mutation_rate:
                    offspring = self._mutate(offspring)

                new_population.append(offspring)

            population = new_population

        training_time = time.time() - start_time
        self.is_trained = True
        self.current_design = best_individual

        return {
            'best_iou': best_fitness,
            'final_iou': fitness_history[-1],
            'training_time': training_time,
            'generations': timesteps
        }

    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate the segmentation algorithm on the test set.

        Args:
            design: Optional design to train and evaluate. If None, use current best.

        Returns:
            Dictionary of performance metrics on test set (IoU)
        """
        if design is not None:
            # Train with this design
            self.train(design)

        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        # Evaluate on test set using the best individual
        test_iou = self._evaluate_on_test_set(self.current_design)

        return {
            'test_iou': test_iou
        }

    def _initialize_population(self, size: int, seed_design: Dict[str, Any]) -> List[List[str]]:
        """Initialize population with variable-length chromosomes."""
        population = []

        # Add the seed design
        if 'chromosome' in seed_design:
            population.append(seed_design['chromosome'])

        # Generate random individuals
        design_space = self.get_design_space()
        component_types = list(design_space.keys())

        while len(population) < size:
            # Random length between 3 and 8 components
            length = np.random.randint(3, 9)
            chromosome = []

            for _ in range(length):
                # Select random component type
                comp_type = np.random.choice(component_types)
                # Select random component of that type
                comp_value = np.random.choice(design_space[comp_type])
                chromosome.append(f"{comp_type}_{comp_value}")

            population.append(chromosome)

        return population

    def _tournament_selection(self, population: List[List[str]],
                             fitnesses: List[float], k: int) -> List[str]:
        """Select individual using tournament selection."""
        indices = np.random.choice(len(population), size=k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()

    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Perform single-point crossover on variable-length chromosomes."""
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1.copy() if len(parent1) > 0 else parent2.copy()

        point1 = np.random.randint(0, len(parent1))
        point2 = np.random.randint(0, len(parent2))

        offspring = parent1[:point1] + parent2[point2:]
        return offspring

    def _mutate(self, chromosome: List[str]) -> List[str]:
        """Mutate chromosome by changing, adding, or removing components."""
        if len(chromosome) == 0:
            return chromosome

        mutation_type = np.random.choice(['change', 'add', 'remove'])
        design_space = self.get_design_space()
        component_types = list(design_space.keys())

        if mutation_type == 'change' and len(chromosome) > 0:
            # Change a random gene
            idx = np.random.randint(0, len(chromosome))
            comp_type = np.random.choice(component_types)
            comp_value = np.random.choice(design_space[comp_type])
            chromosome[idx] = f"{comp_type}_{comp_value}"

        elif mutation_type == 'add' and len(chromosome) < 10:
            # Add a new component
            comp_type = np.random.choice(component_types)
            comp_value = np.random.choice(design_space[comp_type])
            chromosome.append(f"{comp_type}_{comp_value}")

        elif mutation_type == 'remove' and len(chromosome) > 2:
            # Remove a component
            idx = np.random.randint(0, len(chromosome))
            chromosome.pop(idx)

        return chromosome

    def _evaluate_design(self, chromosome: List[str]) -> float:
        """
        Evaluate a chromosome by computing ARI on validation set.

        Args:
            chromosome: List of components defining the segmentation pipeline

        Returns:
            Fitness score (ARI - Adjusted Rand Index)
        """
        if not self.val_data:
            return 0.0

        # Extract images and labels from validation data
        images = np.array([img for img, _ in self.val_data])
        labels = np.array([label for _, label in self.val_data])

        # Execute pipeline and get fitness
        fitness, _, _ = self._execute_segmentation_pipeline(chromosome, images, labels)

        return fitness

    @staticmethod
    def _evaluate_design_static(chromosome: List[str], val_data) -> float:
        """
        Static version of _evaluate_design for parallel execution.

        This method can be pickled by multiprocessing.

        Args:
            chromosome: Chromosome to evaluate
            val_data: Validation data (list of (image, label) tuples)

        Returns:
            Fitness score (ARI - Adjusted Rand Index)
        """
        if not val_data:
            return 0.0

        try:
            # Extract images and labels from validation data
            images = np.array([img for img, _ in val_data])
            labels = np.array([label for _, label in val_data])

            # Initialize segmentation object
            seg = segmentation()

            # Find the segmentation component in chromosome (CIS or CG)
            segmentation_type = 'S1'  # Default SLIC
            seg_args = {'n_segments': 100, 'compactness': 10, 'sigma': 1}

            for component in chromosome:
                if component.startswith('CIS_') or component.startswith('CG_'):
                    # Parse and map to segmentation type
                    parts = component.split('_', 1)
                    if len(parts) >= 2:
                        comp_name = parts[1]

                        # CIS mappings
                        cis_mapping = {
                            'slic_superpixels': ('S1', {'n_segments': 100, 'compactness': 10, 'sigma': 1}),
                            'chan_vese': ('S2', {'mu': 0.25}),
                            'felsenszwalb': ('S3', {'scale': 100, 'sigma': 0.5}),
                            'quickshift': ('S5', {'max_dist': 6, 'ratio': 0.5, 'kernel_size': 5, 'sigma': 0}),
                            'watershed': ('S6', {'compactness': 0.001}),
                        }

                        # CG mappings
                        cg_mapping = {
                            'kmeans_plusplus': ('S8', {}),
                            'affinity_propagation': ('S9', {'damping': 0.5}),
                            'mean_shift': ('S10', {}),
                            'spectral_clustering': ('S11', {'num_clusters': 5}),
                            'agglomerative': ('S12', {'linkage': 'ward'}),
                            'dbscan': ('S13', {'metric': 'euclidean'}),
                            'optics': ('S14', {'max_eps': np.inf, 'metric': 'euclidean'}),
                            'gaussian_mixtures': ('S15', {}),
                            'birch': ('S16', {'threshold': 0.5, 'n_clusters': 5}),
                            'bisecting_kmeans': ('S17', {'bisecting_strategy': 'biggest_inertia'})
                        }

                        if parts[0] == 'CIS' and comp_name in cis_mapping:
                            segmentation_type, seg_args = cis_mapping[comp_name]
                        elif parts[0] == 'CG' and comp_name in cg_mapping:
                            segmentation_type, seg_args = cg_mapping[comp_name]
                    break

            # Apply segmentation
            segmented = seg.segment(images, labels, segmentation_type, seg_args)

            # Compute fitness using ARI
            fitness, _, _ = ari(labels, segmented)

            return fitness
        except Exception as e:
            # If segmentation fails, return low fitness
            return 0.0

    def _execute_segmentation_pipeline(
        self,
        chromosome: List[str],
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Execute the segmentation pipeline defined by the chromosome.

        Args:
            chromosome: List of components like ['CIS_slic_superpixels', 'MOR_dilation']
            images: Input images
            labels: Ground truth labels

        Returns:
            Tuple of (fitness_score, ground_truth, segmented_images)
        """
        # Initialize segmentation object
        seg = segmentation()

        # Find the segmentation component in chromosome (CIS or CG)
        segmentation_type = None
        seg_args = {}

        for component in chromosome:
            if component.startswith('CIS_') or component.startswith('CG_'):
                # Map our component names to reference code segmentation types
                segmentation_type, seg_args = self._map_to_segmentation_type(component)
                break

        # If no segmentation found, use default (SLIC)
        if segmentation_type is None:
            segmentation_type = 'S1'  # SLIC
            seg_args = {'n_segments': 100, 'compactness': 10, 'sigma': 1}

        # Apply segmentation
        try:
            segmented = seg.segment(images, labels, segmentation_type, seg_args)

            # Compute fitness using ARI (Adjusted Rand Index)
            fitness, gt, seg_images = ari(labels, segmented)

            return fitness, gt, seg_images
        except Exception as e:
            # If segmentation fails, return low fitness
            print(f"Segmentation pipeline failed: {e}")
            return 0.0, labels, images

    def _map_to_segmentation_type(self, component: str) -> Tuple[str, Dict[str, Any]]:
        """
        Map chromosome component to reference code segmentation type and arguments.

        Args:
            component: Component string like 'CIS_slic_superpixels' or 'CG_kmeans_plusplus'

        Returns:
            Tuple of (segmentation_type, arguments_dict)
        """
        # Image-specific clustering (CIS) mappings
        cis_mapping = {
            'slic_superpixels': ('S1', {'n_segments': 100, 'compactness': 10, 'sigma': 1}),
            'chan_vese': ('S2', {'mu': 0.25}),
            'felsenszwalb': ('S3', {'scale': 100, 'sigma': 0.5}),
            'quickshift': ('S5', {'max_dist': 6, 'ratio': 0.5, 'kernel_size': 5, 'sigma': 0}),
            'watershed': ('S6', {'compactness': 0.001}),
            'gabor_filter': ('S18', {'n_segments': 100, 'compactness': 10, 'sigma': 1})  # Mask SLIC
        }

        # Generic clustering (CG) mappings
        cg_mapping = {
            'kmeans_plusplus': ('S8', {}),  # kmeans
            'affinity_propagation': ('S9', {'damping': 0.5}),
            'mean_shift': ('S10', {}),
            'spectral_clustering': ('S11', {'num_clusters': 5}),
            'agglomerative': ('S12', {'linkage': 'ward'}),
            'dbscan': ('S13', {'metric': 'euclidean'}),
            'optics': ('S14', {'max_eps': np.inf, 'metric': 'euclidean'}),
            'gaussian_mixtures': ('S15', {}),
            'birch': ('S16', {'threshold': 0.5, 'n_clusters': 5}),
            'bisecting_kmeans': ('S17', {'bisecting_strategy': 'biggest_inertia'})
        }

        # Parse component
        parts = component.split('_', 1)
        if len(parts) < 2:
            return 'S1', {'n_segments': 100, 'compactness': 10, 'sigma': 1}

        comp_type = parts[0]
        comp_name = parts[1]

        if comp_type == 'CIS' and comp_name in cis_mapping:
            return cis_mapping[comp_name]
        elif comp_type == 'CG' and comp_name in cg_mapping:
            return cg_mapping[comp_name]
        else:
            # Default to SLIC
            return 'S1', {'n_segments': 100, 'compactness': 10, 'sigma': 1}

    def _evaluate_on_test_set(self, chromosome: List[str]) -> float:
        """Evaluate chromosome on test set."""
        if not self.test_data:
            return 0.0

        # Extract images and labels from test data
        images = np.array([img for img, _ in self.test_data])
        labels = np.array([label for _, label in self.test_data])

        # Execute pipeline
        fitness, _, _ = self._execute_segmentation_pipeline(chromosome, images, labels)

        return fitness

    def get_meta_features(self) -> Dict[str, float]:
        """Extract meta-features from the dataset."""
        num_train = len(self.train_data) if self.train_data else 0
        num_val = len(self.val_data) if self.val_data else 0
        num_test = len(self.test_data) if self.test_data else 0

        return {
            'num_train_samples': num_train,
            'num_val_samples': num_val,
            'num_test_samples': num_test,
            'input_channels': self.dataset_info['channels'],
            'input_height': self.dataset_info['height'],
            'input_width': self.dataset_info['width'],
            'total_pixels': self.dataset_info['height'] * self.dataset_info['width'] * self.dataset_info['channels']
        }

    def get_application_type(self) -> str:
        """Get the type of application."""
        return 'composition'

    def supports_dynamic_designs(self) -> bool:
        """Segmentation supports dynamic designs (changing algorithm during execution)."""
        return True

    def get_num_timesteps(self) -> int:
        """Return default number of GA generations."""
        return 60

    def extract_meta_features(self, timestep: int = 0) -> Dict[str, Any]:
        """
        Extract meta-features for the current state of evolution.

        Args:
            timestep: Current timestep (generation)

        Returns:
            Dictionary of meta-features
        """
        # For segmentation, we need a predicted segmentation and ground truth
        # Since we're in composition mode, we'll use the best chromosome's output
        # For now, return basic features if we don't have segmentations yet

        if self.current_design is None or len(self.val_data) == 0:
            # Return only basic dataset features
            return {
                'num_classes': self.dataset_info.get('num_classes', 2),
                'num_instances': len(self.train_data) if self.train_data else 0,
                'input_height': self.dataset_info.get('height', 256),
                'input_width': self.dataset_info.get('width', 256),
                'timestep': timestep
            }

        # Get a sample image and ground truth from validation set
        sample_image, sample_gt = self.val_data[0]

        # Apply current best design to get predicted segmentation
        try:
            predicted_seg = self.apply_chromosome(self.current_design, sample_image)

            return self.meta_feature_extractor.extract_meta_features(
                predicted_segmentation=predicted_seg,
                ground_truth_segmentation=sample_gt,
                dataset_info=self.dataset_info,
                timestep=timestep
            )
        except:
            # If application fails, return basic features
            return {
                'num_classes': self.dataset_info.get('num_classes', 2),
                'num_instances': len(self.train_data) if self.train_data else 0,
                'input_height': self.dataset_info.get('height', 256),
                'input_width': self.dataset_info.get('width', 256),
                'timestep': timestep
            }
