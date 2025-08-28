"""Circle shape implementation."""

from typing import Tuple
from PIL import ImageDraw
from tiled_dummy_gen.shapes.base import Shape, ShapeProperties


class Circle(Shape):
    """Circle shape for spatial learning tasks.

    Useful for scenarios like:
    - Size comparison tasks
    - Color identification
    - Spatial positioning
    """

    def __init__(self, properties: ShapeProperties):
        """Initialize circle shape.

        Args:
            properties: Common shape properties
        """
        super().__init__(properties)

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        canvas_size: Tuple[int, int],
        **kwargs,
    ) -> None:
        """Draw circle at the specified position.

        Args:
            draw: PIL ImageDraw object
            position: (x, y) center position of the circle
            canvas_size: Canvas dimensions for size scaling
        """
        # Base radius scaled by size property and canvas
        base_radius = 40  # Base radius in pixels
        radius = self._scale_size(base_radius, canvas_size)

        cx, cy = position
        left = cx - radius
        top = cy - radius
        right = cx + radius
        bottom = cy + radius

        bbox = [left, top, right, bottom]

        if self.properties.fill:
            draw.ellipse(bbox, fill=self.properties.color)
        else:
            draw.ellipse(bbox, outline=self.properties.color, width=2)

    def get_description(self, position_description: str = "") -> str:
        """Generate text description of the circle.

        Args:
            position_description: Spatial description (e.g., "on the left")

        Returns:
            Text description
        """
        color_name = self._get_color_name(self.properties.color)
        size_desc = ""

        if self.properties.size > 1.2:
            size_desc = "large "
        elif self.properties.size < 0.8:
            size_desc = "small "

        base_description = f"a {size_desc}{color_name} circle"

        if position_description:
            return f"{base_description} {position_description}"
        return base_description

    def get_bounding_box(
        self, position: Tuple[int, int], canvas_size: Tuple[int, int], **kwargs
    ) -> Tuple[int, int, int, int]:
        """Get bounding box for the circle.

        Args:
            position: (x, y) center position
            canvas_size: Canvas dimensions

        Returns:
            (left, top, right, bottom) bounding box
        """
        base_radius = 40
        radius = self._scale_size(base_radius, canvas_size)

        cx, cy = position
        return (cx - radius, cy - radius, cx + radius, cy + radius)
