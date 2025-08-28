"""TiledDummyGen: A synthetic data generation library for creating tiled dummy datasets with image embeddings.

This package provides tools for generating synthetic image datasets with configurable
properties, embedding images using pre-trained models, and organizing the data in
various formats suitable for machine learning applications.

## Legacy Components (Original API - fully supported):
- SyntheticDataGenerator: Generate synthetic images with configurable properties
- ImageEmbedder: Embed images using pre-trained vision transformers
- SyntheticDataPipeline: Complete pipeline for data generation and processing
- Configuration system: JSON-based configuration for experiments

## New Components (Enhanced API for spatial feature learning):
- SceneGenerator: Generate multi-object scenes with spatial relationships
- Shape system: Extensible shapes (bars, stars, circles) with factory pattern
- Scene composition: Multi-object scenes with flexible layouts
- Task generators: Spatial feature learning task presets
- Backward compatibility: Legacy API continues to work unchanged

## Quick Start (New API):
```python
from tiled_dummy_gen.tasks.presets import create_two_star_binary_classification

# Create spatial learning task with two stars
config = create_two_star_binary_classification(
    relation="left_of",
    num_scenes=100,
    colors=["red", "blue", "yellow"]
)
```
"""

__version__ = "0.1.0"

# Legacy API (backward compatible)
from tiled_dummy_gen.core.generator import SyntheticDataGenerator
from tiled_dummy_gen.core.embedder import ImageEmbedder
from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline
from tiled_dummy_gen.config.parser import (
    ExperimentConfigLoader,
    ExperimentConfig,
    ClassConfig,
    DatasetConfig,
    SplitConfig,
    EmbedderConfig,
    PreprocessingConfig,
)

# New API components
from tiled_dummy_gen.core.scene_generator import SceneGenerator
from tiled_dummy_gen.config.scene_config import (
    SceneConfigLoader,
    ExperimentSceneConfig,
    SceneConfig,
    ObjectConfig,
    TaskConfig,
    LayoutConfig,
    TaskType,
)
from tiled_dummy_gen.shapes import ShapeFactory, Shape, Bar, Star, Circle
from tiled_dummy_gen.scene import (
    Scene,
    SceneObject,
    SpatialLayout,
    RandomLayout,
    GridLayout,
    RelationalLayout,
)
from tiled_dummy_gen.tasks import (
    SpatialBinaryTaskGenerator,
    ColorComparisonTaskGenerator,
    SizeComparisonTaskGenerator,
)

# Data management and export (shared between APIs)
from tiled_dummy_gen.data.manager import DataManager
from tiled_dummy_gen.export.hdf5_exporter import HDF5Exporter
from tiled_dummy_gen.export.file_structure_exporter import FileStructureExporter

# Backward compatibility adapters (for seamless migration)
from tiled_dummy_gen.compat import (
    LegacyGeneratorAdapter,
    LegacyPipelineAdapter,
    ConfigConverter,
)

# Convenience aliases for common use cases
ConfigParser = ExperimentConfigLoader  # Alias for legacy compatibility


__all__ = [
    # Legacy API (unchanged - ensures backward compatibility)
    "SyntheticDataGenerator",
    "ImageEmbedder",
    "SyntheticDataPipeline",
    "ExperimentConfigLoader",
    "ExperimentConfig",
    "ClassConfig",
    "DatasetConfig",
    "SplitConfig",
    "EmbedderConfig",
    "PreprocessingConfig",
    "ConfigParser",  # Legacy alias
    # New API components
    "SceneGenerator",
    "SceneConfigLoader",
    "ExperimentSceneConfig",
    "SceneConfig",
    "ObjectConfig",
    "TaskConfig",
    "LayoutConfig",
    "TaskType",
    # Shape system
    "ShapeFactory",
    "Shape",
    "Bar",
    "Star",
    "Circle",
    # Scene composition
    "Scene",
    "SceneObject",
    "SpatialLayout",
    "RandomLayout",
    "GridLayout",
    "RelationalLayout",
    # Task generators
    "SpatialBinaryTaskGenerator",
    "ColorComparisonTaskGenerator",
    "SizeComparisonTaskGenerator",
    # Shared components
    "DataManager",
    "HDF5Exporter",
    "FileStructureExporter",
    # Compatibility layer
    "LegacyGeneratorAdapter",
    "LegacyPipelineAdapter",
    "ConfigConverter",
]
