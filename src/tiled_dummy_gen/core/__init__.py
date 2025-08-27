"""Core functionality for TiledDummyGen.

This module contains the main classes for synthetic data generation,
image embedding, and pipeline orchestration.
"""

from .generator import SyntheticDataGenerator
from .embedder import ImageEmbedder
from .pipeline import SyntheticDataPipeline

__all__ = ["SyntheticDataGenerator", "ImageEmbedder", "SyntheticDataPipeline"]