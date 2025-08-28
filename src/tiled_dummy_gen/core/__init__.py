"""Core functionality for TiledDummyGen.

This module contains the main classes for synthetic data generation,
image embedding, and pipeline orchestration.
"""

from tiled_dummy_gen.core.generator import SyntheticDataGenerator
from tiled_dummy_gen.core.embedder import ImageEmbedder
from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

__all__ = ["SyntheticDataGenerator", "ImageEmbedder", "SyntheticDataPipeline"]
