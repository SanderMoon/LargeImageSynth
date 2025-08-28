"""Backward compatibility layer for existing bar-based functionality.

This module provides adapters and wrappers to ensure that existing
code using the old SyntheticDataGenerator continues to work while
benefiting from the new scene-based architecture internally.
"""

from tiled_dummy_gen.compat.adapters import (
    LegacyGeneratorAdapter,
    LegacyPipelineAdapter,
)
from tiled_dummy_gen.compat.converters import ConfigConverter, ClassConfigConverter

__all__ = [
    "LegacyGeneratorAdapter",
    "LegacyPipelineAdapter",
    "ConfigConverter",
    "ClassConfigConverter",
]
