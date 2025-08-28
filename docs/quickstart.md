# Quickstart Guide

## Installation

```bash
# Clone repository
git clone https://github.com/sandermoonemans/LargeImageSynth.git
cd LargeImageSynth

# Install package
pip install -e ".[cli]"

# Verify installation
large-image-synth --help
```

## 5-Minute Example

### 1. Validate Example Configuration
```bash
large-image-synth validate examples/experiments/binary.json
```

### 2. Run Complete Pipeline
```bash
large-image-synth run examples/experiments/binary.json --verbose
```

This generates:
- Synthetic images in `../data/synthetic_data_binary_class/images/`
- Image embeddings as CSV
- HDF5 dataset file
- Train/validation/test splits

### 3. Use Python API

```python
from tiled_dummy_gen import ConfigParser, SyntheticDataPipeline

# Load configuration
parser = ConfigParser("examples/experiments/binary.json")
config = parser.get_experiment_config()

# Run pipeline
pipeline = SyntheticDataPipeline(config=config)
pipeline.run_with_embedding()

# Export results
pipeline.save_to_hdf5("my_dataset.hdf5")
print("✅ Done!")
```

## Key Concepts

### Configuration-Driven
All experiments are defined in JSON files specifying:
- **Class templates** - Base parameters for synthetic classes
- **Variations** - Parameters that vary across classes (generates combinations)
- **Output formats** - HDF5, file structures, or both
- **Embedder settings** - Which vision model to use

### Tiled Datasets
- `num_tiles_base=1` - Single images
- `num_tiles_base=3` - 3×3 tiled images (9 tiles per image)
- Useful for spatial relationship modeling

### Two-Step Process
1. **Generation** - Create synthetic images with descriptions
2. **Embedding** - Convert images to feature vectors using Vision Transformers

## Next Steps

- Explore `examples/experiments/` for different configurations
- Check `docs/api-reference.md` for detailed API documentation  
- See `docs/cli-reference.md` for all CLI options