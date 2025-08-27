# API Reference

## Core Classes

### `SyntheticDataGenerator`

Generates synthetic images with configurable visual properties.

```python
from tiled_dummy_gen import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    image_size=(256, 256),  # Output image dimensions
    noise_level=10,         # Gaussian noise level
    base_image_size=(256, 256)  # Base size for thickness scaling
)

# Generate single sample
image, text = generator.generate_sample(class_config)
```

**Key Methods:**
- `generate_sample(class_config, image_size=None)` - Generate image with description

### `ImageEmbedder`

Embeds images using pre-trained Vision Transformers.

```python
from tiled_dummy_gen import ImageEmbedder, EmbedderConfig

config = EmbedderConfig(
    embedder_name="google/vit-base-patch16-224",
    device="cpu"  # or "cuda"
)
embedder = ImageEmbedder(config)

# Embed single image
embedding = embedder.embed_image(pil_image)  # Returns numpy array
```

**Key Methods:**
- `embed_image(image)` - Generate embedding for PIL Image
- `save_preprocessed_visualization(image, path)` - Save preprocessing visualization

### `SyntheticDataPipeline`

Complete pipeline orchestrating generation, embedding, and data export.

```python
from tiled_dummy_gen import SyntheticDataPipeline, ConfigParser

# Load configuration
parser = ConfigParser("config.json")
config = parser.get_experiment_config()

# Run pipeline
pipeline = SyntheticDataPipeline(config=config)
pipeline.run_with_embedding()  # Generate + embed
# or
pipeline.run()  # Generate only

# Export data
pipeline.save_to_hdf5("dataset.hdf5")
pipeline.save_to_file_structure("output_dir/")
```

**Key Methods:**
- `run()` - Generate data and annotations
- `run_with_embedding()` - Generate data, annotations, and embeddings
- `save_to_hdf5(path)` - Export to HDF5 format
- `save_to_file_structure(dir)` - Export to structured file format
- `create_zero_shot_labels()` - Generate zero-shot learning labels

## Configuration Classes

### `ExperimentConfig`

Main configuration container.

```python
from tiled_dummy_gen.config import *

config = ExperimentConfig(
    class_configs=[...],          # List of ClassConfig
    embedder_config=embedder_cfg, # EmbedderConfig
    dataset_config=dataset_cfg,   # DatasetConfig  
    split_config=split_cfg,       # SplitConfig
    output_dir="./output",        # Output directory
    num_tiles_base=3,            # Grid size for tiling (3x3)
    image_size=(224, 224)        # Base image dimensions
)
```

### `ClassConfig`

Defines a class of synthetic data to generate.

```python
class_config = ClassConfig(
    name="subject_001",
    num_samples=10,
    image_background_color="blue",     # "blue", "red", "green", etc.
    image_bar_orientation="horizontal", # "horizontal", "vertical", "diagonal"  
    image_bar_thickness="medium",      # "thin", "medium", "thick"
    augment_images=True,
    augment_images_noise_level=10.0,
    augment_images_zoom_factor=(1.0, 1.3),
    augment_texts=False
)
```

### `EmbedderConfig`

Configuration for image embedding.

```python
from tiled_dummy_gen.config import EmbedderConfig, PreprocessingConfig

embedder_config = EmbedderConfig(
    embedder_name="google/vit-base-patch16-224",  # HuggingFace model
    device="cpu",  # "cpu" or "cuda"
    preprocessing=PreprocessingConfig(
        resize=True,
        resize_size=(224, 224),
        normalize=True,
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225)
    )
)
```

## Configuration Parser

### `ConfigParser`

Loads and validates JSON configurations.

```python
from tiled_dummy_gen import ConfigParser

parser = ConfigParser("experiment.json")
config = parser.get_experiment_config()  # Returns ExperimentConfig
```

**JSON Structure:**
```json
{
  "class_template": {
    "num_samples": 10,
    "augment_images": true,
    "augment_images_noise_level": 10.0
  },
  "variations": {
    "image_background_color": ["blue", "red"],
    "image_bar_orientation": ["horizontal", "vertical"]
  },
  "embedder_config": {
    "embedder_name": "google/vit-base-patch16-224",
    "device": "cpu"
  },
  "dataset_config": {
    "hdf5_filename": "dataset.hdf5"
  },
  "split_config": {
    "train": 0.7, "val": 0.2, "test": 0.1
  },
  "output_dir": "./output",
  "num_tiles_base": 1,
  "image_size": [224, 224]
}
```

The `variations` section generates all combinations of specified parameters, creating multiple `ClassConfig` instances automatically.