# LargeImageSynth

A Python library for generating large synthetic images with spatial objects, automatic tiling, and embeddings for machine learning research.

## Features

- **Scene-Based Generation**: Create complex multi-object scenes with spatial relationships for spatial feature learning
- **Dual API Architecture**: Both legacy bar-based generation and new flexible scene composition system
- **Automatic Tiling**: Generate large images (e.g., 672×672) and automatically tile them into smaller patches (e.g., 224×224)
- **Spatial Layouts**: Support for random, grid-based, and relationship-based object positioning with customizable spread
- **Extensible Shape System**: Built-in support for bars, stars, circles with easy extension for custom shapes
- **Image Embedding**: Automatically embed tiles using pre-trained Vision Transformer models  
- **Multiple Output Formats**: Export to HDF5, WebDataset, or structured file formats
- **Command-Line Interface**: Easy-to-use CLI for running experiments
- **JSON Configuration**: Flexible configuration supporting both legacy and scene-based formats

## Installation

### From Source

```bash
git clone https://github.com/sandermoonemans/LargeImageSynth.git
cd LargeImageSynth
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

1. **Generate spatial learning dataset**:
```bash
large-image-synth run examples/experiments/spatial_stars_example.json --verbose
```

2. **Validate a configuration**:
```bash
large-image-synth validate examples/experiments/binary.json
```

3. **Generate data without embeddings**:
```bash
large-image-synth generate examples/experiments/spatial_stars_example.json
```

### Using the Python API

**Scene-based API (Recommended)**:
```python
from tiled_dummy_gen.config.scene_config import SceneConfigLoader
from tiled_dummy_gen.core.scene_generator import SceneBasedPipeline

# Load scene configuration
loader = SceneConfigLoader('examples/experiments/spatial_stars_example.json')
config = loader.load_config()

# Create and run pipeline
pipeline = SceneBasedPipeline(config)
pipeline.run_with_embedding()
pipeline.save_data("hdf5")
```

**Legacy API (Still supported)**:
```python
from tiled_dummy_gen.config.parser import ExperimentConfigLoader
from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

# Load configuration
loader = ExperimentConfigLoader('examples/experiments/binary.json')
config = loader.get_experiment_config()

# Create and run pipeline
pipeline = SyntheticDataPipeline(config=config)
pipeline.run_with_embedding()
pipeline.save_data("hdf5")
```

## Configuration

### Scene-Based Configuration

Create complex multi-object scenes with spatial relationships:

```json
{
  "scenes": [
    {
      "scene_id": "star1_on_top",
      "background_color": "white",
      "canvas_size": [672, 672],
      "objects": [
        {
          "object_id": "star1", 
          "shape_type": "star",
          "color": "red",
          "size": 0.3
        },
        {
          "object_id": "star2",
          "shape_type": "star", 
          "color": "blue",
          "size": 0.3
        }
      ],
      "layout": {
        "layout_type": "relational",
        "layout_params": {
          "separation_distance": 200,
          "horizontal_spread": 150,
          "vertical_spread": 120
        },
        "relationships": [
          {
            "object1": "star1",
            "object2": "star2", 
            "relation": "above"
          }
        ]
      },
      "task": {
        "task_type": "binary_classification",
        "num_samples": 50
      }
    }
  ],
  "embedder_config": {
    "embedder_name": "google/vit-base-patch16-224",
    "device": "cpu"
  },
  "dataset_config": {
    "hdf5_filename": "spatial_stars_dataset.hdf5"
  },
  "num_tiles_base": 3,
  "image_size": [672, 672]
}
```

### Legacy Configuration (Still Supported)

```json
{
  "class_template": {
    "image_bar_orientation": "diagonal",
    "image_bar_thickness": "thick", 
    "num_samples": 8
  },
  "variations": {
    "image_background_color": ["blue", "red"]
  },
  "embedder_config": {
    "embedder_name": "google/vit-base-patch16-224"
  },
  "num_tiles_base": 3
}
```

## Project Structure

```
LargeImageSynth/
├── src/tiled_dummy_gen/          # Main package
│   ├── core/                     # Core functionality  
│   │   ├── scene_generator.py   # Scene-based pipeline│   │   ├── generator.py         # Legacy synthetic data generation
│   │   ├── embedder.py          # Image embedding
│   │   └── pipeline.py          # Legacy pipeline
│   ├── config/                  # Configuration system
│   │   ├── scene_config.py      # Scene-based config (NEW) 
│   │   └── parser.py            # Legacy config parsing
│   ├── shapes/                  # Shape system│   │   ├── base.py             # Abstract shape classes
│   │   ├── factory.py          # Shape factory
│   │   ├── bar.py              # Bar shape
│   │   ├── star.py             # Star shape  
│   │   └── circle.py           # Circle shape
│   ├── scene/                   # Scene composition│   │   ├── objects.py          # Scene and object classes
│   │   ├── layout.py           # Spatial layout strategies
│   │   └── relationships.py    # Spatial relationship analysis
│   ├── tasks/                   # Task generators│   │   ├── generators.py       # Task-specific generators
│   │   └── presets.py          # Common task presets
│   ├── export/                  # Export formats
│   │   ├── hdf5_exporter.py    # HDF5 export
│   │   └── webdataset_exporter.py # WebDataset export
│   ├── compat/                  # Backward compatibility│   │   ├── adapters.py         # Legacy API adapters
│   │   └── converters.py       # Config converters
│   └── cli.py                   # Command-line interface
├── examples/                    # Examples and experiments
│   └── experiments/            # JSON configuration files
├── tests/                      # Test suite  
└── pyproject.toml             # Package configuration
```

## CLI Commands

- `large-image-synth run <config>` - Run complete pipeline with embeddings
- `large-image-synth generate <config>` - Generate synthetic data only
- `large-image-synth embed <config>` - Generate embeddings for existing data  
- `large-image-synth validate <config>` - Validate configuration file

Use `--help` with any command for detailed options.

## Development

### Setup Development Environment

```bash
git clone https://github.com/sandermoonemans/LargeImageSynth.git
cd LargeImageSynth
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