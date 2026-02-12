# PhD Thesis: Experiment Recreation & Extension

## Project Overview
This repository contains the implementation for recreating and extending experiments from the PhD thesis on automated machine learning (AutoML) with meta-learning approaches.

## Project Structure

```
.
├── applications/           # Six AutoML applications
│   ├── base_application.py    # Abstract base class for all applications
│   ├── configuration/         # Configuration applications
│   │   ├── cnn/              # CNN hyperparameter optimization
│   │   └── video/            # Video classification pipeline configuration
│   ├── composition/          # Composition applications
│   │   ├── clustering/       # Image clustering algorithm composition
│   │   └── segmentation/     # Image segmentation algorithm composition
│   └── generation/           # Generation applications
│       ├── feature/          # Image feature extraction function generation
│       └── fuzzyart/         # Fuzzy ART choice function generation
│
├── datasets/              # Dataset handling
│   └── image_datasets.py      # Image dataset loader (MNIST, CIFAR, etc.)
│
├── OnMAR/                # Online meta-learning implementation
│   ├── accuracy-prediction/
│   └── design-prediction/
│
├── OffMAR/               # Offline meta-learning implementation
│   ├── accuracy-prediction/
│   └── design-prediction/
│
├── sota/                 # State-of-the-art baseline (AutoSklearn)
│
├── Thesis/               # Reference materials from thesis
│
└── requirements.txt      # Python dependencies
```

## Current Implementation Status

### ✅ Completed
- **Base Infrastructure**
  - Abstract base application interface for plug-and-play design
  - Standard interface for all applications (train, evaluate, get_design_space, etc.)

- **CNN Configuration Application** (`applications/configuration/cnn/`)
  - Configurable CNN architecture based on thesis Appendix A.3
  - Support for multiple datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
  - Design space includes:
    - Convolutional filters (8-2048)
    - Batch normalization (on/off)
    - Activation functions (ELU, GELU, ReLU, SELU, Sigmoid, Softmax, Softplus, Swish, Tanh)
    - Dropout rates
    - Max pooling sizes
    - Dense layer nodes
    - Optimizers (Adam, Adamax, RMSprop, AdaGrad, AdaDelta, SGD, Nadam)
    - Learning rates
    - Number of convolutional layers

- **Dataset Infrastructure** (`datasets/`)
  - Image dataset loader with support for standard datasets
  - Automatic downloading and preprocessing
  - Train/validation/test splits

### 🚧 TODO
- Video classification application
- Image clustering composition application
- Image segmentation composition application
- Image feature extraction generation application
- Fuzzy ART choice function generation application
- OnMAR implementation (online meta-learning)
- OffMAR implementation (offline meta-learning)
- AutoSklearn baseline integration
- Additional datasets (ISIC Melanoma, Mosquito, FruitsGB, video datasets, etc.)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Testing the CNN Configuration Application

```python
from applications.configuration.cnn import CNNConfigurationApplication

# Initialize application
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Load data
app.load_data()

# Define a design
design = {
    'conv_filters': 32,
    'batch_norm': 1,
    'activation': 3,  # ReLU
    'dropout': 0.2,
    'max_pool_size': 2,
    'dense_nodes': 128,
    'optimizer': 1,  # Adam
    'learning_rate': 0.001,
    'num_conv_layers': 2
}

# Train and evaluate
train_metrics = app.train(design, timesteps=50)
test_metrics = app.evaluate()

print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
```

Or run the test script:
```bash
python applications/configuration/cnn/test_cnn.py
```

## Design Philosophy

### Plug-and-Play Interface
All applications inherit from `BaseApplication` and implement:
- `load_data()`: Load and prepare datasets
- `get_design_space()`: Define the search space for designs
- `train(design, timesteps)`: Train with a given design
- `evaluate(design)`: Evaluate performance
- `get_meta_features()`: Extract meta-features for meta-learning
- `get_application_type()`: Return 'configuration', 'composition', or 'generation'

This standardized interface allows OnMAR, OffMAR, and AutoSklearn to work with any application seamlessly.

### Efficient and Distributable
- PyTorch for efficient GPU utilization
- Support for both local (Macbook) and HPC cluster execution
- Automatic device detection (CPU/GPU)
- Efficient data loading with multiple workers
- Type hints throughout for clarity and IDE support

## Key References
- Chapter 8 (Meta-Learning): Describes OnMAR and OffMAR approaches
- Chapter 6 (Dynamic Designs): First set of applications (configuration, composition, generation)
- Chapter 7 (Dynamic Design Option Values): Second set of applications
- Chapter 5 (Research Methodology): Dataset descriptions
- Appendix A: Complete design space specifications

## Citation
[Add thesis citation once published]

## License
[Specify license]
