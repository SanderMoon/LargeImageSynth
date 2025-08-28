"""Base classes for the shape system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw


@dataclass
class ShapeProperties:
    """Properties that can be applied to any shape."""

    color: Tuple[int, int, int]  # RGB tuple
    size: float  # Relative size factor (1.0 = default)
    rotation: float = 0.0  # Rotation angle in degrees
    opacity: float = 1.0  # Alpha value (0.0 to 1.0)
    fill: bool = True  # Whether shape is filled or outline only

    def __post_init__(self):
        """Validate property values."""
        if not (0.0 <= self.opacity <= 1.0):
            raise ValueError(f"Opacity must be between 0.0 and 1.0, got {self.opacity}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")


class Shape(ABC):
    """Abstract base class for all shapes in the generation system.

    Provides a consistent interface for:
    - Drawing shapes on PIL images
    - Generating text descriptions
    - Calculating bounding boxes
    - Handling shape-specific parameters
    """

    def __init__(self, properties: ShapeProperties):
        """Initialize shape with common properties.

        Args:
            properties: Common properties like color, size, rotation
        """
        self.properties = properties
        self._validate_properties()

    @abstractmethod
    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        canvas_size: Tuple[int, int],
        **kwargs,
    ) -> None:
        """Draw the shape on the given canvas.

        Args:
            draw: PIL ImageDraw object
            position: (x, y) position for shape center
            canvas_size: (width, height) of the canvas
            **kwargs: Shape-specific parameters
        """
        pass

    @abstractmethod
    def get_description(self, position_description: str = "") -> str:
        """Generate text description of the shape.

        Args:
            position_description: Optional spatial description (e.g., "on the left")

        Returns:
            Text description of the shape
        """
        pass

    @abstractmethod
    def get_bounding_box(
        self, position: Tuple[int, int], canvas_size: Tuple[int, int], **kwargs
    ) -> Tuple[int, int, int, int]:
        """Calculate the bounding box of the shape.

        Args:
            position: (x, y) position for shape center
            canvas_size: (width, height) of the canvas
            **kwargs: Shape-specific parameters

        Returns:
            (left, top, right, bottom) bounding box coordinates
        """
        pass

    def _validate_properties(self) -> None:
        """Validate shape properties. Override for shape-specific validation."""
        pass

    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to color name for descriptions."""
        color_map = {
            (255, 0, 0): "red",
            (0, 255, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
            (255, 0, 255): "magenta",
            (0, 255, 255): "cyan",
            (255, 165, 0): "orange",
            (128, 0, 128): "purple",
            (255, 192, 203): "pink",
            (0, 128, 128): "teal",
            (128, 128, 128): "gray",
            (255, 255, 255): "white",
            (0, 0, 0): "black",
            (165, 42, 42): "brown",
            (0, 128, 0): "dark green",
            (0, 0, 128): "navy",
            (128, 0, 0): "maroon",
        }
        return color_map.get(rgb, "unknown color")

    def _scale_size(self, base_size: int, canvas_size: Tuple[int, int]) -> int:
        """Scale size based on canvas dimensions and size factor.

        Args:
            base_size: Base size in pixels
            canvas_size: Canvas dimensions for scaling reference

        Returns:
            Scaled size in pixels
        """
        # Scale based on minimum canvas dimension to maintain aspect ratio
        min_dim = min(canvas_size)
        scaling_factor = min_dim / 224  # Assume 224x224 as reference size
        return max(1, int(base_size * self.properties.size * scaling_factor))
