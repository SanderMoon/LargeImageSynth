"""TiledDummyGen: A synthetic data generation library for creating tiled dummy datasets with image embeddings.

This package provides tools for generating synthetic image datasets with configurable
properties, embedding images using pre-trained models, and organizing the data in
various formats suitable for machine learning applications.

Main Components:
- SyntheticDataGenerator: Generate synthetic images with configurable properties
- ImageEmbedder: Embed images using pre-trained vision transformers
- SyntheticDataPipeline: Complete pipeline for data generation and processing
- Configuration system: JSON-based configuration for experiments
"""

__version__ = "0.1.0"

from .core.generator import SyntheticDataGenerator
from .core.embedder import ImageEmbedder
from .core.pipeline import SyntheticDataPipeline
from .config.parser import (
    ExperimentConfigLoader,
    ExperimentConfig,
    ClassConfig,
    DatasetConfig,
    SplitConfig,
    EmbedderConfig,
    PreprocessingConfig,
)
from .data.manager import DataManager
from .export.hdf5_exporter import HDF5Exporter
from .export.file_structure_exporter import FileStructureExporter


__all__ = [
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
    "DataManager",
    "HDF5Exporter",
    "FileStructureExporter",
]