"""Factory for creating shapes dynamically."""

from typing import Dict, Type, Any, Tuple
from tiled_dummy_gen.shapes.base import Shape, ShapeProperties
from tiled_dummy_gen.shapes.bar import Bar
from tiled_dummy_gen.shapes.star import Star
from tiled_dummy_gen.shapes.circle import Circle


class ShapeFactory:
    """Factory class for creating shapes dynamically.

    Provides a central registry of available shapes and creates them
    based on string identifiers and configuration parameters.
    """

    def __init__(self):
        """Initialize the factory with default shape types."""
        self._shape_registry: Dict[str, Type[Shape]] = {}
        self._register_default_shapes()

    def _register_default_shapes(self):
        """Register the built-in shape types."""
        self.register_shape("bar", Bar)
        self.register_shape("star", Star)
        self.register_shape("circle", Circle)

    def register_shape(self, name: str, shape_class: Type[Shape]):
        """Register a new shape type.

        Args:
            name: String identifier for the shape
            shape_class: Class that implements the Shape interface
        """
        if not issubclass(shape_class, Shape):
            raise ValueError(f"Shape class must inherit from Shape, got {shape_class}")

        self._shape_registry[name.lower()] = shape_class

    def create_shape(
        self,
        shape_type: str,
        color: Tuple[int, int, int] = (0, 0, 0),
        size: float = 1.0,
        **kwargs,
    ) -> Shape:
        """Create a shape instance.

        Args:
            shape_type: Type of shape to create ("bar", "star", "circle")
            color: RGB color tuple
            size: Size scaling factor
            **kwargs: Shape-specific parameters

        Returns:
            Shape instance

        Raises:
            ValueError: If shape_type is not registered
        """
        shape_type_lower = shape_type.lower()

        if shape_type_lower not in self._shape_registry:
            available_shapes = list(self._shape_registry.keys())
            raise ValueError(
                f"Unknown shape type: {shape_type}. Available: {available_shapes}"
            )

        # Create common properties
        properties = ShapeProperties(
            color=color,
            size=size,
            rotation=kwargs.pop("rotation", 0.0),
            opacity=kwargs.pop("opacity", 1.0),
            fill=kwargs.pop("fill", True),
        )

        # Get the shape class and create instance
        shape_class = self._shape_registry[shape_type_lower]

        try:
            return shape_class(properties, **kwargs)
        except TypeError as e:
            # Provide helpful error message for invalid parameters
            raise ValueError(f"Invalid parameters for {shape_type}: {e}")

    def get_available_shapes(self) -> list[str]:
        """Get list of available shape types.

        Returns:
            List of registered shape type names
        """
        return list(self._shape_registry.keys())

    def create_bar(
        self,
        color: Tuple[int, int, int] = (0, 0, 0),
        orientation: str = "horizontal",
        thickness: str = "medium",
        **kwargs,
    ) -> Bar:
        """Convenience method for creating bars.

        Args:
            color: RGB color tuple
            orientation: "horizontal", "vertical", or "diagonal"
            thickness: "thin", "medium", or "thick"
            **kwargs: Additional properties

        Returns:
            Bar instance
        """
        return self.create_shape(
            "bar", color=color, orientation=orientation, thickness=thickness, **kwargs
        )

    def create_star(
        self,
        color: Tuple[int, int, int] = (255, 255, 0),
        size: float = 1.0,
        num_points: int = 5,
        **kwargs,
    ) -> Star:
        """Convenience method for creating stars.

        Args:
            color: RGB color tuple
            size: Size scaling factor
            num_points: Number of star points
            **kwargs: Additional properties

        Returns:
            Star instance
        """
        return self.create_shape(
            "star", color=color, size=size, num_points=num_points, **kwargs
        )

    def create_circle(
        self, color: Tuple[int, int, int] = (255, 0, 0), size: float = 1.0, **kwargs
    ) -> Circle:
        """Convenience method for creating circles.

        Args:
            color: RGB color tuple
            size: Size scaling factor
            **kwargs: Additional properties

        Returns:
            Circle instance
        """
        return self.create_shape("circle", color=color, size=size, **kwargs)


# Global factory instance
default_factory = ShapeFactory()
