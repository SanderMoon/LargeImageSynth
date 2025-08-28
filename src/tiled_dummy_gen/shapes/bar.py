"""Bar shape implementation - maintains existing bar functionality."""

import math
from typing import Tuple
from PIL import ImageDraw
from tiled_dummy_gen.shapes.base import Shape, ShapeProperties


class Bar(Shape):
    """Bar/line shape with configurable orientation and thickness.

    Supports the same orientations as the original system:
    - horizontal
    - vertical
    - diagonal
    """

    def __init__(
        self,
        properties: ShapeProperties,
        orientation: str = "horizontal",
        thickness: str = "medium",
    ):
        """Initialize bar shape.

        Args:
            properties: Common shape properties
            orientation: One of 'horizontal', 'vertical', 'diagonal'
            thickness: One of 'thin', 'medium', 'thick'
        """
        super().__init__(properties)
        self.orientation = orientation
        self.thickness = thickness

        # Thickness mapping from original code
        self.thickness_map = {"thin": 5, "medium": 20, "thick": 50}

        if orientation not in ["horizontal", "vertical", "diagonal"]:
            raise ValueError(f"Invalid orientation: {orientation}")
        if thickness not in self.thickness_map:
            raise ValueError(f"Invalid thickness: {thickness}")

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        canvas_size: Tuple[int, int],
        **kwargs,
    ) -> None:
        """Draw bar on the canvas.

        Args:
            draw: PIL ImageDraw object
            position: Center position (ignored for bars that span full canvas)
            canvas_size: Canvas dimensions
        """
        width, height = canvas_size
        base_thickness = self.thickness_map[self.thickness]
        thickness_px = self._scale_size(base_thickness, canvas_size)
        color = self.properties.color

        if self.orientation == "horizontal":
            y = height // 2
            draw.rectangle(
                [0, y - thickness_px // 2, width, y + thickness_px // 2], fill=color
            )
        elif self.orientation == "vertical":
            x = width // 2
            draw.rectangle(
                [x - thickness_px // 2, 0, x + thickness_px // 2, height], fill=color
            )
        elif self.orientation == "diagonal":
            draw.line([(0, 0), (width, height)], fill=color, width=thickness_px)

    def get_description(self, position_description: str = "") -> str:
        """Generate text description of the bar.

        Args:
            position_description: Ignored for bars (they don't have relative positions)

        Returns:
            Text description
        """
        color_name = self._get_color_name(self.properties.color)

        templates = [
            f"a {self.thickness} {self.orientation} {color_name} bar",
            f"a {self.thickness}, {self.orientation} {color_name} bar",
            f"a {color_name} bar oriented {self.orientation} with {self.thickness} thickness",
        ]

        return templates[0]  # Use first template for consistency

    def get_bounding_box(
        self, position: Tuple[int, int], canvas_size: Tuple[int, int], **kwargs
    ) -> Tuple[int, int, int, int]:
        """Get bounding box for the bar.

        Args:
            position: Ignored (bars span the canvas)
            canvas_size: Canvas dimensions

        Returns:
            (left, top, right, bottom) bounding box
        """
        width, height = canvas_size
        base_thickness = self.thickness_map[self.thickness]
        thickness_px = self._scale_size(base_thickness, canvas_size)

        if self.orientation == "horizontal":
            return (
                0,
                height // 2 - thickness_px // 2,
                width,
                height // 2 + thickness_px // 2,
            )
        elif self.orientation == "vertical":
            return (
                width // 2 - thickness_px // 2,
                0,
                width // 2 + thickness_px // 2,
                height,
            )
        else:  # diagonal
            # Diagonal bar spans the entire canvas
            return (0, 0, width, height)
