"""Shape system for synthetic data generation.

This module provides an extensible shape system that supports:
- Abstract base classes for all shapes
- Concrete implementations for various shapes (bars, stars, circles, etc.)
- Factory pattern for dynamic shape creation
- Standardized drawing and description interfaces
"""

from tiled_dummy_gen.shapes.base import Shape, ShapeProperties
from tiled_dummy_gen.shapes.bar import Bar
from tiled_dummy_gen.shapes.star import Star
from tiled_dummy_gen.shapes.circle import Circle
from tiled_dummy_gen.shapes.factory import ShapeFactory

__all__ = ["Shape", "ShapeProperties", "Bar", "Star", "Circle", "ShapeFactory"]
