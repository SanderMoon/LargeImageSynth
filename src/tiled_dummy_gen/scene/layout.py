"""Spatial layout strategies for positioning objects in scenes."""

import random
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from tiled_dummy_gen.scene.objects import SceneObject


class SpatialLayout(ABC):
    """Abstract base class for spatial layout strategies.

    Layout strategies determine how objects are positioned within a scene.
    Different strategies enable different types of spatial learning tasks.
    """

    @abstractmethod
    def position_objects(
        self, objects: List[SceneObject], canvas_size: Tuple[int, int]
    ) -> List[SceneObject]:
        """Position the objects according to this layout strategy.

        Args:
            objects: List of objects to position (positions may be modified)
            canvas_size: Canvas dimensions

        Returns:
            List of positioned objects (may be the same list, modified in-place)
        """
        pass


class RandomLayout(SpatialLayout):
    """Randomly position objects within the canvas, avoiding overlaps.

    Good for generating varied datasets where spatial relationships
    are not predetermined.
    """

    def __init__(
        self, margin: int = 50, max_attempts: int = 100, min_distance: int = 60
    ):
        """Initialize random layout.

        Args:
            margin: Minimum distance from canvas edges
            max_attempts: Maximum attempts to place each object without overlap
            min_distance: Minimum distance between object centers
        """
        self.margin = margin
        self.max_attempts = max_attempts
        self.min_distance = min_distance

    def position_objects(
        self, objects: List[SceneObject], canvas_size: Tuple[int, int]
    ) -> List[SceneObject]:
        """Position objects randomly, avoiding overlaps.

        Args:
            objects: Objects to position
            canvas_size: Canvas dimensions

        Returns:
            Positioned objects
        """
        if not objects:
            return objects

        width, height = canvas_size
        positioned_objects = []

        for obj in objects:
            placed = False

            for attempt in range(self.max_attempts):
                # Generate random position within margins
                x = random.randint(self.margin, width - self.margin)
                y = random.randint(self.margin, height - self.margin)

                # Check for overlaps with already placed objects
                valid_position = True
                for placed_obj in positioned_objects:
                    px, py = placed_obj.position
                    distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                    if distance < self.min_distance:
                        valid_position = False
                        break

                if valid_position:
                    obj.position = (x, y)
                    placed = True
                    break

            if not placed:
                # Fallback: place randomly even if overlapping
                x = random.randint(self.margin, width - self.margin)
                y = random.randint(self.margin, height - self.margin)
                obj.position = (x, y)

            positioned_objects.append(obj)

        return positioned_objects


class GridLayout(SpatialLayout):
    """Position objects in a regular grid pattern.

    Useful for systematic spatial relationship studies and
    creating balanced datasets.
    """

    def __init__(
        self, rows: Optional[int] = None, cols: Optional[int] = None, margin: int = 50
    ):
        """Initialize grid layout.

        Args:
            rows: Number of grid rows (auto-calculated if None)
            cols: Number of grid columns (auto-calculated if None)
            margin: Margin from canvas edges
        """
        self.rows = rows
        self.cols = cols
        self.margin = margin

    def position_objects(
        self, objects: List[SceneObject], canvas_size: Tuple[int, int]
    ) -> List[SceneObject]:
        """Position objects in a grid.

        Args:
            objects: Objects to position
            canvas_size: Canvas dimensions

        Returns:
            Positioned objects
        """
        if not objects:
            return objects

        n_objects = len(objects)
        width, height = canvas_size

        # Calculate grid dimensions
        if self.rows and self.cols:
            rows, cols = self.rows, self.cols
        elif self.rows:
            rows = self.rows
            cols = math.ceil(n_objects / rows)
        elif self.cols:
            cols = self.cols
            rows = math.ceil(n_objects / cols)
        else:
            # Auto-calculate: try to make grid roughly square
            cols = math.ceil(math.sqrt(n_objects))
            rows = math.ceil(n_objects / cols)

        # Calculate grid cell size
        available_width = width - 2 * self.margin
        available_height = height - 2 * self.margin
        cell_width = available_width / cols
        cell_height = available_height / rows

        # Position objects in grid cells
        for i, obj in enumerate(objects):
            row = i // cols
            col = i % cols

            # Center of grid cell
            x = self.margin + cell_width * (col + 0.5)
            y = self.margin + cell_height * (row + 0.5)

            obj.position = (int(x), int(y))

        return objects


class RelationalLayout(SpatialLayout):
    """Position objects based on specified spatial relationships.

    Perfect for creating targeted spatial learning tasks like:
    - "Object A is to the left of Object B"
    - "Star 1 is above Star 2"
    """

    def __init__(
        self,
        relationships: List[Dict[str, Any]],
        margin: int = 50,
        separation_distance: int = 80,
        horizontal_spread: int = 0,
        vertical_spread: int = 0,
    ):
        """Initialize relational layout.

        Args:
            relationships: List of relationship specs:
                [
                    {
                        'object1': 'obj1_id',
                        'object2': 'obj2_id',
                        'relation': 'left_of'|'right_of'|'above'|'below'
                    }
                ]
            margin: Margin from canvas edges
            separation_distance: Distance between related objects
            horizontal_spread: Additional random horizontal offset range (±spread)
            vertical_spread: Additional random vertical offset range (±spread)
        """
        self.relationships = relationships
        self.margin = margin
        self.separation_distance = separation_distance
        self.horizontal_spread = horizontal_spread
        self.vertical_spread = vertical_spread

    def position_objects(
        self, objects: List[SceneObject], canvas_size: Tuple[int, int]
    ) -> List[SceneObject]:
        """Position objects according to specified relationships.

        Args:
            objects: Objects to position
            canvas_size: Canvas dimensions

        Returns:
            Positioned objects
        """
        if not objects:
            return objects

        width, height = canvas_size
        positioned_ids = set()

        # Create object lookup
        obj_map = {obj.object_id: obj for obj in objects}

        for relationship in self.relationships:
            obj1_id = relationship["object1"]
            obj2_id = relationship["object2"]
            relation = relationship["relation"]

            if obj1_id not in obj_map or obj2_id not in obj_map:
                continue  # Skip invalid relationships

            obj1 = obj_map[obj1_id]
            obj2 = obj_map[obj2_id]

            # If neither object is positioned, place obj1 first
            if obj1_id not in positioned_ids and obj2_id not in positioned_ids:
                # Place obj1 at a reasonable default position
                obj1.position = (width // 2, height // 2)
                positioned_ids.add(obj1_id)

            # Position the second object relative to the first
            if obj1_id in positioned_ids and obj2_id not in positioned_ids:
                self._position_relative(obj1, obj2, relation, canvas_size)
                positioned_ids.add(obj2_id)
            elif obj2_id in positioned_ids and obj1_id not in positioned_ids:
                # Reverse the relationship
                reversed_relation = self._reverse_relation(relation)
                self._position_relative(obj2, obj1, reversed_relation, canvas_size)
                positioned_ids.add(obj1_id)

        # Position any remaining objects randomly
        unpositioned = [obj for obj in objects if obj.object_id not in positioned_ids]
        if unpositioned:
            random_layout = RandomLayout(margin=self.margin)
            random_layout.position_objects(unpositioned, canvas_size)

        return objects

    def _position_relative(
        self,
        reference_obj: SceneObject,
        target_obj: SceneObject,
        relation: str,
        canvas_size: Tuple[int, int],
    ) -> None:
        """Position target object relative to reference object.

        Args:
            reference_obj: Object to position relative to
            target_obj: Object to be positioned
            relation: Spatial relationship ('left_of', 'right_of', 'above', 'below')
            canvas_size: Canvas dimensions
        """
        ref_x, ref_y = reference_obj.position
        width, height = canvas_size

        if relation == "left_of":
            x = max(self.margin, ref_x - self.separation_distance)
            y = ref_y
        elif relation == "right_of":
            x = min(width - self.margin, ref_x + self.separation_distance)
            y = ref_y
        elif relation == "above":
            x = ref_x
            y = max(self.margin, ref_y - self.separation_distance)
        elif relation == "below":
            x = ref_x
            y = min(height - self.margin, ref_y + self.separation_distance)
        else:
            # Unknown relation, place randomly nearby
            x = ref_x + random.randint(-50, 50)
            y = ref_y + random.randint(-50, 50)

        # Apply random spread to add variation while maintaining relationships
        if self.horizontal_spread > 0:
            x += random.randint(-self.horizontal_spread, self.horizontal_spread)
        if self.vertical_spread > 0:
            y += random.randint(-self.vertical_spread, self.vertical_spread)

        # Ensure position is within canvas bounds
        x = max(self.margin, min(width - self.margin, x))
        y = max(self.margin, min(height - self.margin, y))

        target_obj.position = (x, y)

    def _reverse_relation(self, relation: str) -> str:
        """Get the reverse of a spatial relationship.

        Args:
            relation: Original relation

        Returns:
            Reversed relation
        """
        reverse_map = {
            "left_of": "right_of",
            "right_of": "left_of",
            "above": "below",
            "below": "above",
        }
        return reverse_map.get(relation, relation)
