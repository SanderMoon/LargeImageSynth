# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the LargeImageSynth codebase.

## Development Commands

### Installation and Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install with CLI support only
pip install -e ".[cli]"
```

### Testing and Quality Assurance
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/tiled_dummy_gen

# Format code
black src/ tests/

# Lint code  
ruff src/ tests/

# Type checking
mypy src/
```

### CLI Usage
```bash
# Run complete pipeline with embeddings (scene-based)
large-image-synth run examples/experiments/spatial_stars_example.json --verbose

# Generate synthetic data only (no embeddings)
large-image-synth generate examples/experiments/spatial_stars_example.json

# Validate configuration file
large-image-synth validate examples/experiments/binary.json

# Embed existing data
large-image-synth embed examples/experiments/spatial_stars_example.json

# Legacy bar-based generation still works
large-image-synth run examples/experiments/binary.json --verbose
```

## Architecture Overview

### Dual API Architecture (NEW)

LargeImageSynth provides **two complementary APIs**:

1. **Legacy API** (original, fully supported): Bar-based generation with `SyntheticDataGenerator`
2. **Scene API** (new, enhanced): Multi-object scenes with spatial relationships and automatic tiling

Both APIs share the same underlying infrastructure and work seamlessly together.

### Scene-Based Pipeline Flow (Primary)
1. **Configuration Loading** (`config/scene_config.py`) - Loads scene-based JSON configurations with automatic legacy format conversion
2. **Shape System** (`shapes/`) - Extensible shape factory with abstract base classes (bars, stars, circles, custom shapes)
3. **Scene Composition** (`scene/`) - Multi-object scenes with flexible spatial layouts and relationship-based positioning
4. **Scene Generation** (`core/scene_generator.py`) - Generates large images (e.g., 672×672) with spatial relationships
5. **Automatic Tiling** - Splits large images into smaller tiles (e.g., 3×3 grid of 224×224 patches) 
6. **Task Generation** (`tasks/`) - High-level generators for spatial learning scenarios with proper labeling
7. **Embedding & Export** - ViT embedding of individual tiles with spatial metadata and HDF5/WebDataset export

### Key Components

**SceneBasedPipeline** (NEW): Primary pipeline that generates large images, tiles them automatically, and creates spatial learning datasets. Handles both full-image storage and tile extraction with proper coordinate tracking.

**Shape System** (NEW): 
- Abstract `Shape` base class with consistent drawing/description interface
- Concrete implementations: `Bar`, `Star`, `Circle` with extensible parameters and size control
- `ShapeFactory` for dynamic shape creation from configuration
- Easy extension for custom shapes

**Scene Composition** (NEW):
- `Scene`: Container for multiple objects with spatial relationships and analysis
- `SceneObject`: Shape + position + metadata with automatic coordinate tracking
- Layout strategies: `RandomLayout`, `GridLayout`, `RelationalLayout` with spread parameters
- Relationship analysis for generating spatial descriptions and binary classification labels

**Automatic Tiling System** (NEW):
- Splits large images (e.g., 672×672) into tile grids (e.g., 3×3 of 224×224)
- Maintains tile coordinates (tile_x, tile_y) for spatial analysis
- Links tiles to original full images via `full_image_filename` reference
- Handles both individual tile processing and full-scene context

**Task Generators** (NEW): High-level interfaces for spatial learning:
- `SpatialBinaryTaskGenerator`: "Is star A above star B?" with proper class labeling
- `ColorComparisonTaskGenerator`: "Are both objects the same color?" 
- Preset functions like `create_two_star_binary_classification()` with configurable relationships

**Legacy Components** (unchanged): `SyntheticDataGenerator`, `ImageEmbedder`, `SyntheticDataPipeline` work exactly as before but now use the new architecture internally via compatibility adapters.

### Configuration System

**New Scene Format**: Multi-object scenes with spatial relationships and tiling:
```json
{
  "scenes": [{
    "scene_id": "star1_on_top",
    "background_color": "white",
    "canvas_size": [672, 672],
    "objects": [
      {"object_id": "star1", "shape_type": "star", "color": "red", "size": 0.3},
      {"object_id": "star2", "shape_type": "star", "color": "blue", "size": 0.3}
    ],
    "layout": {
      "layout_type": "relational",
      "layout_params": {
        "separation_distance": 200,
        "horizontal_spread": 150, 
        "vertical_spread": 120
      },
      "relationships": [{"object1": "star1", "object2": "star2", "relation": "above"}]
    },
    "task": {"task_type": "binary_classification", "num_samples": 50}
  }],
  "num_tiles_base": 3,
  "image_size": [672, 672]
}
```

**Legacy Format**: Still fully supported - automatically converted internally:
```json
{
  "class_template": {"image_bar_orientation": "diagonal", "num_samples": 8},
  "variations": {"image_background_color": ["blue", "red"]},
  "num_tiles_base": 3
}
```

### Spatial Feature Learning Support

Perfect for spatial reasoning and large-scale vision tasks:
- **Large image generation**: Create high-resolution scenes (672×672, 1024×1024, etc.)
- **Automatic tiling**: Split into ML-ready patches (224×224) with coordinate tracking
- **Multi-object scenes**: 2+ objects with independent properties and spatial relationships
- **Spatial relationships**: Above/below, left/right with configurable spread and variation
- **Task-oriented labeling**: Binary classification for "which object is above/below which"
- **Full traceability**: Each tile linked to source image and spatial context
- **Extensible shapes**: Built-in bars, stars, circles + easy custom shape addition

## Common Development Patterns

### Adding New Export Formats
1. Create new exporter class inheriting from `BaseExporter` in `export/`
2. Implement `export()` method following the interface
3. Register in `SyntheticDataPipeline.save_data()` method

### Adding New Embedding Models  
1. Extend `ImageEmbedder` configuration in `config/parser.py`
2. Add model loading logic in `embedder.py`
3. Update preprocessing pipeline as needed

### Configuration Validation
All configurations use dataclasses with validation in `config/parser.py`. Split ratios must sum to 1.0, and required fields are enforced at runtime.