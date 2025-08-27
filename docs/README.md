# TiledDummyGen Documentation

## Overview

TiledDummyGen is a Python library for generating synthetic tiled datasets with image embeddings, designed for machine learning research and experimentation.

## Documentation Structure

### Getting Started
- **[Quickstart Guide](quickstart.md)** - 5-minute setup and basic usage
- **[CLI Reference](cli-reference.md)** - Complete command-line interface documentation  
- **[API Reference](api-reference.md)** - Python API classes and methods

### Key Features

🎨 **Configurable Synthetic Data**
- Generate images with customizable background colors, bar orientations, and thicknesses
- Template-based configuration with automatic variation generation

🧠 **Image Embeddings**  
- Automatic embedding using Vision Transformers (ViT, CLIP, etc.)
- Configurable preprocessing and device selection

🔧 **Multiple Export Formats**
- HDF5 format for ML frameworks
- PyTorch-compatible file structures
- CSV annotations and embeddings

⚡ **Command-Line Interface**
- Validate configurations before running
- Generate data with or without embeddings
- Verbose output for debugging

🎯 **Tiled Dataset Support**
- Single images or multi-tile grids
- Spatial relationship modeling
- Automatic tile coordinate tracking

## Quick Reference

### CLI Commands
```bash
tiled-dummy-gen validate config.json    # Validate configuration
tiled-dummy-gen run config.json         # Complete pipeline  
tiled-dummy-gen generate config.json    # Data generation only
tiled-dummy-gen embed config.json       # Embedding generation only
```

### Python API
```python
from tiled_dummy_gen import ConfigParser, SyntheticDataPipeline

parser = ConfigParser("config.json")
pipeline = SyntheticDataPipeline(parser.get_experiment_config())
pipeline.run_with_embedding()
```

### Configuration Structure
```json
{
  "class_template": { "num_samples": 10, "augment_images": true },
  "variations": { "image_background_color": ["blue", "red"] },
  "embedder_config": { "embedder_name": "google/vit-base-patch16-224" },
  "output_dir": "./output"
}
```

## Examples

Find complete examples in the `examples/` directory:
- `binary.json` - Binary classification setup
- `multiclass.json` - Multi-class scenario
- `great_variety.json` - Large variation experiment