"""Spatial relationship rules and analysis."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from tiled_dummy_gen.scene.objects import Scene, SceneObject


class SpatialRelation(Enum):
    """Enumeration of spatial relationships between objects."""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR = "far"


@dataclass
class RelationshipRule:
    """A rule defining a spatial relationship between two objects.

    Used for:
    - Validating scene layouts
    - Generating task-specific descriptions
    - Creating training labels for ML tasks
    """

    object1_id: str
    object2_id: str
    relation: SpatialRelation
    confidence_threshold: float = 0.8  # Confidence required to classify relationship

    def evaluate(self, scene: Scene) -> Dict[str, Any]:
        """Evaluate whether this relationship holds in the given scene.

        Args:
            scene: Scene to evaluate

        Returns:
            Dictionary with evaluation results:
            {
                'holds': bool,
                'confidence': float,
                'details': dict
            }
        """
        obj1 = scene.get_object(self.object1_id)
        obj2 = scene.get_object(self.object2_id)

        if not obj1 or not obj2:
            return {
                "holds": False,
                "confidence": 0.0,
                "details": {"error": "One or both objects not found"},
            }

        return self._evaluate_relationship(obj1, obj2, scene.canvas_size)

    def _evaluate_relationship(
        self, obj1: SceneObject, obj2: SceneObject, canvas_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Evaluate the specific relationship between two objects.

        Args:
            obj1: First object
            obj2: Second object
            canvas_size: Canvas dimensions

        Returns:
            Evaluation results
        """
        x1, y1 = obj1.position
        x2, y2 = obj2.position

        dx = x2 - x1
        dy = y2 - y1
        distance = (dx**2 + dy**2) ** 0.5

        canvas_width, canvas_height = canvas_size
        max_distance = (canvas_width**2 + canvas_height**2) ** 0.5

        if self.relation == SpatialRelation.LEFT_OF:
            # obj1 is left of obj2 if obj1.x < obj2.x significantly
            confidence = max(0, dx) / (
                canvas_width * 0.5
            )  # Normalize by half canvas width
            confidence = min(1.0, confidence)
            holds = confidence >= self.confidence_threshold

        elif self.relation == SpatialRelation.RIGHT_OF:
            # obj1 is right of obj2 if obj1.x > obj2.x significantly
            confidence = max(0, -dx) / (canvas_width * 0.5)
            confidence = min(1.0, confidence)
            holds = confidence >= self.confidence_threshold

        elif self.relation == SpatialRelation.ABOVE:
            # obj1 is above obj2 if obj1.y < obj2.y significantly
            confidence = max(0, dy) / (canvas_height * 0.5)
            confidence = min(1.0, confidence)
            holds = confidence >= self.confidence_threshold

        elif self.relation == SpatialRelation.BELOW:
            # obj1 is below obj2 if obj1.y > obj2.y significantly
            confidence = max(0, -dy) / (canvas_height * 0.5)
            confidence = min(1.0, confidence)
            holds = confidence >= self.confidence_threshold

        elif self.relation == SpatialRelation.NEAR:
            # Objects are near if distance is small relative to canvas
            normalized_distance = distance / max_distance
            confidence = 1.0 - normalized_distance  # Closer = higher confidence
            holds = confidence >= self.confidence_threshold

        elif self.relation == SpatialRelation.FAR:
            # Objects are far if distance is large relative to canvas
            normalized_distance = distance / max_distance
            confidence = normalized_distance  # Further = higher confidence
            holds = confidence >= self.confidence_threshold

        else:
            confidence = 0.0
            holds = False

        return {
            "holds": holds,
            "confidence": confidence,
            "details": {
                "distance": distance,
                "dx": dx,
                "dy": dy,
                "position1": obj1.position,
                "position2": obj2.position,
            },
        }


class RelationshipAnalyzer:
    """Analyzes spatial relationships in scenes and generates descriptions."""

    def __init__(self, confidence_threshold: float = 0.6):
        """Initialize relationship analyzer.

        Args:
            confidence_threshold: Minimum confidence to report a relationship
        """
        self.confidence_threshold = confidence_threshold

    def analyze_scene(self, scene: Scene) -> List[Dict[str, Any]]:
        """Analyze all spatial relationships in a scene.

        Args:
            scene: Scene to analyze

        Returns:
            List of detected relationships with confidence scores
        """
        relationships = []

        # Check all pairs of objects
        for i, obj1 in enumerate(scene.objects):
            for j, obj2 in enumerate(scene.objects):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue

                # Test each possible relationship
                for relation in SpatialRelation:
                    rule = RelationshipRule(
                        obj1.object_id,
                        obj2.object_id,
                        relation,
                        self.confidence_threshold,
                    )

                    result = rule.evaluate(scene)
                    if result["holds"]:
                        relationships.append(
                            {
                                "object1": obj1.object_id,
                                "object2": obj2.object_id,
                                "relation": relation.value,
                                "confidence": result["confidence"],
                                "details": result["details"],
                            }
                        )

        # Sort by confidence (highest first)
        relationships.sort(key=lambda x: x["confidence"], reverse=True)
        return relationships

    def get_primary_relationships(
        self, scene: Scene, max_relationships: int = 3
    ) -> List[Dict[str, Any]]:
        """Get the most confident spatial relationships in a scene.

        Args:
            scene: Scene to analyze
            max_relationships: Maximum number of relationships to return

        Returns:
            List of primary relationships
        """
        all_relationships = self.analyze_scene(scene)
        return all_relationships[:max_relationships]

    def generate_relationship_description(self, scene: Scene) -> str:
        """Generate natural language description based on spatial relationships.

        Args:
            scene: Scene to describe

        Returns:
            Natural language description
        """
        if len(scene.objects) < 2:
            if len(scene.objects) == 1:
                obj = scene.objects[0]
                return f"A scene with {obj.get_description()}"
            return "An empty scene"

        primary_relationships = self.get_primary_relationships(
            scene, max_relationships=1
        )

        if not primary_relationships:
            # Fallback: just list objects
            object_descriptions = [obj.get_description() for obj in scene.objects]
            return f"A scene with {', '.join(object_descriptions)}"

        # Use the primary relationship for description
        rel = primary_relationships[0]
        obj1 = scene.get_object(rel["object1"])
        obj2 = scene.get_object(rel["object2"])

        relation = rel["relation"]

        if relation == "left_of":
            desc = f"A scene with {obj1.get_description()} to the left of {obj2.get_description()}"
        elif relation == "right_of":
            desc = f"A scene with {obj1.get_description()} to the right of {obj2.get_description()}"
        elif relation == "above":
            desc = (
                f"A scene with {obj1.get_description()} above {obj2.get_description()}"
            )
        elif relation == "below":
            desc = (
                f"A scene with {obj1.get_description()} below {obj2.get_description()}"
            )
        elif relation == "near":
            desc = (
                f"A scene with {obj1.get_description()} near {obj2.get_description()}"
            )
        elif relation == "far":
            desc = f"A scene with {obj1.get_description()} far from {obj2.get_description()}"
        else:
            desc = f"A scene with {obj1.get_description()} and {obj2.get_description()}"

        return desc

    def create_binary_classification_labels(
        self, scene: Scene, target_relation: SpatialRelation
    ) -> Dict[str, Any]:
        """Create binary classification labels for a specific spatial relationship.

        Args:
            scene: Scene to analyze
            target_relation: Relationship to create labels for

        Returns:
            Classification labels and metadata
        """
        if len(scene.objects) != 2:
            raise ValueError("Binary classification requires exactly 2 objects")

        obj1, obj2 = scene.objects
        rule = RelationshipRule(
            obj1.object_id, obj2.object_id, target_relation, self.confidence_threshold
        )

        result = rule.evaluate(scene)

        return {
            "label": 1 if result["holds"] else 0,
            "confidence": result["confidence"],
            "question": f"Is {obj1.get_description()} {target_relation.value.replace('_', ' ')} {obj2.get_description()}?",
            "objects": {
                "object1": {
                    "id": obj1.object_id,
                    "description": obj1.get_description(),
                    "position": obj1.position,
                },
                "object2": {
                    "id": obj2.object_id,
                    "description": obj2.get_description(),
                    "position": obj2.position,
                },
            },
            "details": result["details"],
        }
