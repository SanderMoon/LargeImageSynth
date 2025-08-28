"""Scene composition system for multi-object generation.

This module provides:
- Scene objects that contain multiple shapes
- Spatial layout strategies for positioning objects
- Relationship rules for expressing spatial constraints
- Text generation based on scene composition
"""

from tiled_dummy_gen.scene.objects import SceneObject, Scene
from tiled_dummy_gen.scene.layout import (
    SpatialLayout,
    RandomLayout,
    GridLayout,
    RelationalLayout,
)
from tiled_dummy_gen.scene.relationships import RelationshipRule, SpatialRelation

__all__ = [
    "SceneObject",
    "Scene",
    "SpatialLayout",
    "RandomLayout",
    "GridLayout",
    "RelationalLayout",
    "RelationshipRule",
    "SpatialRelation",
]
