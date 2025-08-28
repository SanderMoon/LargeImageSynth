"""Scene and SceneObject classes for multi-object composition."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw
from tiled_dummy_gen.shapes.base import Shape


@dataclass
class SceneObject:
    """An object within a scene - combines a shape with position and metadata.

    This allows the same shape type to appear multiple times in a scene
    with different positions, sizes, colors, etc.
    """

    shape: Shape
    position: Tuple[int, int]  # (x, y) position on canvas
    object_id: str  # Unique identifier within the scene
    metadata: Dict[str, Any] = None  # Additional properties for ML tasks

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def draw(self, draw: ImageDraw.ImageDraw, canvas_size: Tuple[int, int]) -> None:
        """Draw this object on the canvas.

        Args:
            draw: PIL ImageDraw object
            canvas_size: Canvas dimensions
        """
        self.shape.draw(draw, self.position, canvas_size)

    def get_bounding_box(
        self, canvas_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Get bounding box for this object.

        Args:
            canvas_size: Canvas dimensions

        Returns:
            (left, top, right, bottom) bounding box
        """
        return self.shape.get_bounding_box(self.position, canvas_size)

    def get_description(self, position_description: str = "") -> str:
        """Get text description of this object.

        Args:
            position_description: Spatial description relative to other objects

        Returns:
            Text description
        """
        return self.shape.get_description(position_description)


class Scene:
    """A scene containing multiple objects with spatial relationships.

    Manages:
    - Multiple SceneObject instances
    - Background properties
    - Scene-level metadata
    - Rendering and text generation
    """

    def __init__(
        self,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        canvas_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize empty scene.

        Args:
            background_color: RGB background color
            canvas_size: Canvas dimensions
        """
        self.background_color = background_color
        self.canvas_size = canvas_size
        self.objects: List[SceneObject] = []
        self.metadata: Dict[str, Any] = {}

    def add_object(self, scene_object: SceneObject) -> None:
        """Add an object to the scene.

        Args:
            scene_object: Object to add

        Raises:
            ValueError: If object_id already exists
        """
        # Check for duplicate IDs
        existing_ids = {obj.object_id for obj in self.objects}
        if scene_object.object_id in existing_ids:
            raise ValueError(
                f"Object ID '{scene_object.object_id}' already exists in scene"
            )

        self.objects.append(scene_object)

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get object by ID.

        Args:
            object_id: ID of object to retrieve

        Returns:
            SceneObject if found, None otherwise
        """
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def remove_object(self, object_id: str) -> bool:
        """Remove object by ID.

        Args:
            object_id: ID of object to remove

        Returns:
            True if object was removed, False if not found
        """
        for i, obj in enumerate(self.objects):
            if obj.object_id == object_id:
                del self.objects[i]
                return True
        return False

    def render(self) -> Image.Image:
        """Render the scene to an image.

        Returns:
            PIL Image of the rendered scene
        """
        # Create background
        image = Image.new("RGB", self.canvas_size, self.background_color)
        draw = ImageDraw.Draw(image)

        # Draw all objects
        for obj in self.objects:
            obj.draw(draw, self.canvas_size)

        return image

    def get_spatial_relationships(self) -> List[Dict[str, Any]]:
        """Analyze spatial relationships between objects.

        Returns:
            List of relationship dictionaries with format:
            {
                'object1': object_id,
                'object2': object_id,
                'relation': 'left_of' | 'right_of' | 'above' | 'below',
                'distance': float
            }
        """
        relationships = []

        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue

                x1, y1 = obj1.position
                x2, y2 = obj2.position

                # Determine primary relationship
                dx = x2 - x1
                dy = y2 - y1
                distance = (dx**2 + dy**2) ** 0.5

                # Primary relationship based on largest difference
                if abs(dx) > abs(dy):
                    relation = "right_of" if dx > 0 else "left_of"
                else:
                    relation = "below" if dy > 0 else "above"

                relationships.append(
                    {
                        "object1": obj1.object_id,
                        "object2": obj2.object_id,
                        "relation": relation,
                        "distance": distance,
                        "dx": dx,
                        "dy": dy,
                    }
                )

        return relationships

    def get_scene_description(self) -> str:
        """Generate text description of the entire scene.

        Returns:
            Natural language description of the scene
        """
        if not self.objects:
            return "An empty scene with a background"

        if len(self.objects) == 1:
            obj = self.objects[0]
            return f"A scene with {obj.get_description()}"

        # Multi-object description with spatial relationships
        relationships = self.get_spatial_relationships()

        if not relationships:
            # Fallback if no clear relationships
            object_descriptions = [obj.get_description() for obj in self.objects]
            return f"A scene with {', '.join(object_descriptions[:-1])} and {object_descriptions[-1]}"

        # Use the first clear relationship for description
        rel = relationships[0]
        obj1 = self.get_object(rel["object1"])
        obj2 = self.get_object(rel["object2"])

        # Create positional descriptions
        if rel["relation"] == "left_of":
            desc1 = obj1.get_description("on the left")
            desc2 = obj2.get_description("on the right")
        elif rel["relation"] == "right_of":
            desc1 = obj1.get_description("on the right")
            desc2 = obj2.get_description("on the left")
        elif rel["relation"] == "above":
            desc1 = obj1.get_description("above")
            desc2 = obj2.get_description("below")
        else:  # below
            desc1 = obj1.get_description("below")
            desc2 = obj2.get_description("above")

        return f"A scene with {desc1} and {desc2}"

    def get_object_count(self) -> int:
        """Get number of objects in the scene.

        Returns:
            Object count
        """
        return len(self.objects)

    def clear(self) -> None:
        """Remove all objects from the scene."""
        self.objects.clear()
        self.metadata.clear()
