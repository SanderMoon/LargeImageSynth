# TiledDummyGen

A Python library for generating synthetic tiled datasets with image embeddings for machine learning research.

## Features

- **Configurable Synthetic Data Generation**: Generate synthetic images with customizable properties like background color, bar orientation, and thickness
- **Image Embedding**: Automatically embed images using pre-trained Vision Transformer models
- **Multiple Output Formats**: Save data in HDF5 format or structured file formats
- **Tiled Dataset Support**: Generate multi-tile images for applications requiring spatial relationships
- **Command-Line Interface**: Easy-to-use CLI for running experiments
- **JSON Configuration**: Flexible experiment configuration system

## Installation

### From Source

```bash
git clone https://github.com/sandermoonemans/TiledDummyGen.git
cd TiledDummyGen
pip install -e .
```

### With CLI Support

```bash
pip install -e ".[cli]"
```

### For Development

```bash
pip install -e ".[dev]"
```

## Quick Start

### Using the Command Line

1. **Validate a configuration**:
```bash
tiled-dummy-gen validate examples/experiments/binary.json
```

2. **Run complete pipeline**:
```bash
tiled-dummy-gen run examples/experiments/binary.json --verbose
```

3. **Generate data without embeddings**:
```bash
tiled-dummy-gen generate examples/experiments/binary.json --no-embedding
```

### Using the Python API

```python
from tiled_dummy_gen import ConfigParser, SyntheticDataPipeline

# Load configuration
parser = ConfigParser('examples/experiments/binary.json')
config = parser.get_experiment_config()

# Create and run pipeline
pipeline = SyntheticDataPipeline(config=config)
pipeline.run_with_embedding()

# Save results
pipeline.save_to_hdf5('output/dataset.hdf5')
```

## Configuration

Experiments are configured using JSON files. Here's a basic example:

```json
{
  "class_template": {
    "image_bar_orientation": "diagonal",
    "image_bar_thickness": "thick",
    "num_samples": 8,
    "augment_images": true,
    "augment_images_noise_level": 10.0,
    "augment_images_zoom_factor": [1.0, 1.3],
    "augment_texts": false
  },
  "variations": {
    "image_background_color": ["blue", "red"]
  },
  "embedder_config": {
    "embedder_name": "google/vit-base-patch16-224",
    "device": "cpu"
  },
  "dataset_config": {
    "hdf5_filename": "synthetic_dataset.hdf5"
  },
  "split_config": {
    "split": true,
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
  },
  "output_dir": "data/output",
  "num_tiles_base": 3,
  "image_size": [224, 224]
}
```

## Project Structure

```
TiledDummyGen/
├── src/tiled_dummy_gen/          # Main package
│   ├── core/                     # Core functionality
│   │   ├── generator.py         # Synthetic data generation
│   │   ├── embedder.py          # Image embedding
│   │   └── pipeline.py          # Complete pipeline
│   ├── config/                  # Configuration system
│   │   └── parser.py           # JSON config parsing
│   ├── utils/                  # Utility functions
│   └── cli.py                  # Command-line interface
├── examples/                   # Examples and experiments
│   ├── experiments/           # JSON configuration files
│   └── example.ipynb         # Jupyter notebook examples
├── tests/                    # Test suite
└── pyproject.toml           # Package configuration
```

## CLI Commands

- `tiled-dummy-gen run <config>` - Run complete pipeline with embeddings
- `tiled-dummy-gen generate <config>` - Generate synthetic data only
- `tiled-dummy-gen embed <config>` - Generate embeddings for existing data
- `tiled-dummy-gen validate <config>` - Validate configuration file

Use `--help` with any command for detailed options.

## Development

### Setup Development Environment

```bash
git clone https://github.com/sandermoonemans/TiledDummyGen.git
cd TiledDummyGen
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code  
ruff src/ tests/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests to the main branch.