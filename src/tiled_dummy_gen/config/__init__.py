"""Configuration system for TiledDummyGen.

This module provides configuration classes and parsing functionality
for setting up experiments and managing parameters.
"""

from .parser import (
    ExperimentConfigLoader,
    ExperimentConfig,
    ClassConfig,
    DatasetConfig,
    SplitConfig,
    EmbedderConfig,
    PreprocessingConfig,
)

__all__ = [
    "ExperimentConfigLoader",
    "ExperimentConfig", 
    "ClassConfig",
    "DatasetConfig",
    "SplitConfig",
    "EmbedderConfig",
    "PreprocessingConfig",
]