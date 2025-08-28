"""Star shape implementation for spatial feature learning."""

import math
from typing import Tuple, List
from PIL import ImageDraw
from tiled_dummy_gen.shapes.base import Shape, ShapeProperties


class Star(Shape):
    """Star shape with configurable number of points and size.

    Perfect for spatial relationship tasks like:
    - "Which star is on the left?"
    - "Which star is bigger?"
    - "What color is the star above?"
    """

    def __init__(
        self,
        properties: ShapeProperties,
        num_points: int = 5,
        inner_radius_ratio: float = 0.4,
    ):
        """Initialize star shape.

        Args:
            properties: Common shape properties
            num_points: Number of star points (default: 5)
            inner_radius_ratio: Ratio of inner to outer radius (0.0 to 1.0)
        """
        super().__init__(properties)
        self.num_points = num_points
        self.inner_radius_ratio = inner_radius_ratio

        if num_points < 3:
            raise ValueError(f"Star must have at least 3 points, got {num_points}")
        if not (0.0 < inner_radius_ratio < 1.0):
            raise ValueError(
                f"Inner radius ratio must be between 0.0 and 1.0, got {inner_radius_ratio}"
            )

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        canvas_size: Tuple[int, int],
        **kwargs,
    ) -> None:
        """Draw star at the specified position.

        Args:
            draw: PIL ImageDraw object
            position: (x, y) center position of the star
            canvas_size: Canvas dimensions for size scaling
        """
        # Base radius scaled by size property and canvas
        base_radius = 40  # Base radius in pixels
        radius = self._scale_size(base_radius, canvas_size)
        inner_radius = radius * self.inner_radius_ratio

        # Generate star points
        points = self._generate_star_points(position, radius, inner_radius)

        # Draw the star
        if self.properties.fill:
            draw.polygon(points, fill=self.properties.color)
        else:
            # Draw outline by connecting points
            for i in range(len(points)):
                start = points[i]
                end = points[(i + 1) % len(points)]
                draw.line([start, end], fill=self.properties.color, width=2)

    def _generate_star_points(
        self, center: Tuple[int, int], outer_radius: int, inner_radius: int
    ) -> List[Tuple[int, int]]:
        """Generate the points that define the star polygon.

        Args:
            center: (x, y) center of the star
            outer_radius: Distance from center to outer points
            inner_radius: Distance from center to inner points

        Returns:
            List of (x, y) points defining the star
        """
        points = []
        cx, cy = center

        # Angle between points
        angle_step = 2 * math.pi / (2 * self.num_points)

        for i in range(2 * self.num_points):
            angle = i * angle_step - math.pi / 2  # Start at top

            if i % 2 == 0:  # Outer point
                radius = outer_radius
            else:  # Inner point
                radius = inner_radius

            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((int(x), int(y)))

        return points

    def get_description(self, position_description: str = "") -> str:
        """Generate text description of the star.

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

        base_description = f"a {size_desc}{color_name} star"

        if position_description:
            return f"{base_description} {position_description}"
        return base_description

    def get_bounding_box(
        self, position: Tuple[int, int], canvas_size: Tuple[int, int], **kwargs
    ) -> Tuple[int, int, int, int]:
        """Get bounding box for the star.

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
