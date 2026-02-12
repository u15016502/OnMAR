"""
Text Dataset Handling

This module handles loading and preprocessing of text datasets for
the Fuzzy ART choice function generation application.
Datasets: ChatGPT, Enron, IMDB
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from pathlib import Path
import pandas as pd
import re


class TextDatasetLoader:
    """
    Handles loading and preprocessing of text datasets for clustering.

    Supported datasets:
    - chatgpt: ChatGPT tweets classification dataset
    - enron: Enron emails dataset
    - imdb: IMDB movie reviews sentiment dataset
    """

    def __init__(self, dataset_name: str, data_dir: str = None, random_seed: int = 42):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: Name of the dataset to load (chatgpt, enron, imdb)
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
            'chatgpt': {
                'filename': 'chatgpt.csv',
                'text_column': 'tweets',
                'label_column': 'labels',
                'separator': ','
            },
            'enron': {
                'filename': 'enron.csv',
                'text_column': 'email',
                'label_column': 'label',
                'separator': ','
            },
            'imdb': {
                'filename': 'imdb.csv',
                'text_column': 'review',
                'label_column': 'sentiment',
                'separator': ','
            }
        }

        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Supported: {list(self.dataset_configs.keys())}")

    def load_dataset(self, max_samples: Optional[int] = None,
                    max_features: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the dataset and convert to feature vectors.

        Args:
            max_samples: Maximum number of samples to load (None = all)
            max_features: Maximum number of features for text vectorization

        Returns:
            Tuple of (features, labels)
            - features: numpy array of shape (n_samples, n_features)
            - labels: numpy array of shape (n_samples,)
        """
        config = self.dataset_configs[self.dataset_name]
        file_path = self.data_dir / config['filename']

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load CSV
        df = pd.read_csv(file_path, sep=config['separator'], on_bad_lines='skip')

        # Get text and labels
        texts = df[config['text_column']].astype(str).tolist()
        labels = df[config['label_column']].values

        # Limit samples if specified
        if max_samples is not None and len(texts) > max_samples:
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]

        # Convert text to features using simple bag-of-words
        features = self._text_to_features(texts, max_features)

        return features, labels

    def _text_to_features(self, texts: List[str], max_features: int) -> np.ndarray:
        """
        Convert text to numerical features using bag-of-words.

        Args:
            texts: List of text strings
            max_features: Maximum vocabulary size

        Returns:
            Feature matrix of shape (n_samples, max_features)
        """
        # Build vocabulary from all texts
        word_counts: Dict[str, int] = {}

        for text in texts:
            words = self._tokenize(text)
            for word in set(words):  # Count each word once per document
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:max_features])}

        # Convert texts to feature vectors
        features = np.zeros((len(texts), max_features), dtype=np.float32)

        for i, text in enumerate(texts):
            words = self._tokenize(text)
            word_freq: Dict[str, int] = {}

            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            for word, freq in word_freq.items():
                if word in vocabulary:
                    features[i, vocabulary[word]] = freq

        # Normalize features (TF normalization)
        row_sums = features.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        features = features / row_sums

        return features

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, remove punctuation, split on whitespace.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Split and filter short words
        words = [w for w in text.split() if len(w) > 2]

        return words

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        config = self.dataset_configs[self.dataset_name]
        file_path = self.data_dir / config['filename']

        # Count samples without loading all data
        try:
            df = pd.read_csv(file_path, sep=config['separator'], on_bad_lines='skip')
            num_samples = len(df)
            num_classes = df[config['label_column']].nunique()
        except Exception:
            num_samples = 0
            num_classes = 0

        return {
            'name': self.dataset_name,
            'num_samples': num_samples,
            'num_classes': num_classes,
            'text_column': config['text_column'],
            'label_column': config['label_column']
        }


# Alias for backward compatibility with fuzzyart_application.py
ClusteringDatasetLoader = TextDatasetLoader
