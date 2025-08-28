"""Task generators for creating spatial learning scenarios."""

import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
from tiled_dummy_gen.config.scene_config import (
    SceneConfig,
    ObjectConfig,
    LayoutConfig,
    TaskConfig,
    TaskType,
    ExperimentSceneConfig,
)
from tiled_dummy_gen.scene.relationships import SpatialRelation


class TaskGenerator(ABC):
    """Abstract base class for task generators.

    Task generators create scene configurations for specific
    machine learning tasks and spatial reasoning scenarios.
    """

    @abstractmethod
    def generate_task_configs(self, num_samples: int = 100) -> List[SceneConfig]:
        """Generate scene configurations for this task.

        Args:
            num_samples: Number of scene configurations to generate

        Returns:
            List of scene configurations
        """
        pass

    @abstractmethod
    def get_task_description(self) -> str:
        """Get human-readable description of the task.

        Returns:
            Task description string
        """
        pass


class SpatialBinaryTaskGenerator(TaskGenerator):
    """Generates binary classification tasks based on spatial relationships.

    Example: "Is the red star to the left of the blue star?"
    Perfect for testing spatial reasoning capabilities.
    """

    def __init__(
        self,
        target_relation: SpatialRelation,
        shape_types: List[str] = None,
        colors: List[str] = None,
        canvas_size: Tuple[int, int] = (224, 224),
        num_samples_per_config: int = 10,
    ):
        """Initialize spatial binary task generator.

        Args:
            target_relation: Spatial relationship to classify
            shape_types: List of shape types to use
            colors: List of colors for objects
            canvas_size: Canvas dimensions
            num_samples_per_config: Samples per scene configuration
        """
        self.target_relation = target_relation
        self.shape_types = shape_types or ["star", "circle"]
        self.colors = colors or ["red", "blue", "yellow", "green"]
        self.canvas_size = canvas_size
        self.num_samples_per_config = num_samples_per_config

    def generate_task_configs(self, num_configs: int = 20) -> List[SceneConfig]:
        """Generate scene configurations for binary spatial classification.

        Args:
            num_configs: Number of different scene configurations

        Returns:
            List of scene configurations
        """
        configs = []

        for i in range(num_configs):
            # Create two objects with different properties
            shape1 = random.choice(self.shape_types)
            shape2 = random.choice(self.shape_types)
            color1 = random.choice(self.colors)
            color2 = random.choice(
                [c for c in self.colors if c != color1]
            )  # Different colors

            # Create objects
            obj1 = ObjectConfig(
                object_id="obj1",
                shape_type=shape1,
                color=color1,
                size=random.uniform(0.8, 1.2),
            )

            obj2 = ObjectConfig(
                object_id="obj2",
                shape_type=shape2,
                color=color2,
                size=random.uniform(0.8, 1.2),
            )

            # Create layout that enforces or violates the target relationship
            enforce_relation = random.choice(
                [True, False]
            )  # 50/50 split for balanced dataset

            if enforce_relation:
                # Create layout that satisfies the target relationship
                relationships = [
                    {
                        "object1": "obj1",
                        "object2": "obj2",
                        "relation": self.target_relation.value,
                    }
                ]
                layout = LayoutConfig(
                    layout_type="relational",
                    layout_params={"separation_distance": 80},
                    relationships=relationships,
                )
            else:
                # Use random layout (likely to violate the relationship)
                layout = LayoutConfig(
                    layout_type="random", layout_params={"min_distance": 60}
                )

            # Create task configuration
            task = TaskConfig(
                task_type=TaskType.BINARY_CLASSIFICATION,
                num_samples=self.num_samples_per_config,
                task_params={
                    "target_relation": self.target_relation.value,
                    "question_template": f"Is the {color1} {shape1} {self.target_relation.value.replace('_', ' ')} the {color2} {shape2}?",
                },
            )

            # Create scene configuration
            scene_config = SceneConfig(
                scene_id=f"spatial_binary_{i}_{enforce_relation}",
                background_color="white",
                canvas_size=self.canvas_size,
                objects=[obj1, obj2],
                layout=layout,
                task=task,
                metadata={
                    "target_relation": self.target_relation.value,
                    "expected_label": 1 if enforce_relation else 0,
                    "object1_desc": f"{color1} {shape1}",
                    "object2_desc": f"{color2} {shape2}",
                },
            )

            configs.append(scene_config)

        return configs

    def get_task_description(self) -> str:
        """Get description of this spatial binary task."""
        relation_name = self.target_relation.value.replace("_", " ")
        return f"Binary classification task: determine if one object is {relation_name} another object"


class ColorComparisonTaskGenerator(TaskGenerator):
    """Generates tasks for color identification and comparison.

    Examples:
    - "What color is the star on the left?"
    - "Are both objects the same color?"
    """

    def __init__(
        self,
        colors: List[str] = None,
        shape_types: List[str] = None,
        canvas_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize color comparison task generator.

        Args:
            colors: List of colors to use
            shape_types: List of shape types
            canvas_size: Canvas dimensions
        """
        self.colors = colors or ["red", "blue", "yellow", "green", "purple", "orange"]
        self.shape_types = shape_types or ["star", "circle"]
        self.canvas_size = canvas_size

    def generate_task_configs(self, num_configs: int = 30) -> List[SceneConfig]:
        """Generate color identification task configurations."""
        configs = []

        for i in range(num_configs):
            # Decide task type: same color or different colors
            same_color = random.choice([True, False])

            if same_color:
                color1 = color2 = random.choice(self.colors)
            else:
                color1 = random.choice(self.colors)
                color2 = random.choice([c for c in self.colors if c != color1])

            # Create objects
            shape1 = random.choice(self.shape_types)
            shape2 = random.choice(self.shape_types)

            obj1 = ObjectConfig(object_id="left_obj", shape_type=shape1, color=color1)

            obj2 = ObjectConfig(object_id="right_obj", shape_type=shape2, color=color2)

            # Use relational layout to ensure left/right positioning
            layout = LayoutConfig(
                layout_type="relational",
                relationships=[
                    {
                        "object1": "left_obj",
                        "object2": "right_obj",
                        "relation": "left_of",
                    }
                ],
            )

            # Create task
            task = TaskConfig(
                task_type=TaskType.BINARY_CLASSIFICATION,
                num_samples=10,
                task_params={
                    "task_variant": "color_comparison",
                    "question": "Are both objects the same color?",
                    "expected_answer": same_color,
                },
            )

            scene_config = SceneConfig(
                scene_id=f"color_comparison_{i}",
                objects=[obj1, obj2],
                layout=layout,
                task=task,
                metadata={
                    "same_color": same_color,
                    "colors": [color1, color2],
                    "left_color": color1,
                    "right_color": color2,
                },
            )

            configs.append(scene_config)

        return configs

    def get_task_description(self) -> str:
        """Get description of color comparison task."""
        return "Color identification and comparison task: determine if objects have the same color"


class SizeComparisonTaskGenerator(TaskGenerator):
    """Generates tasks for size comparison between objects."""

    def __init__(
        self,
        shape_types: List[str] = None,
        colors: List[str] = None,
        size_ranges: List[Tuple[float, float]] = None,
    ):
        """Initialize size comparison task generator.

        Args:
            shape_types: List of shape types
            colors: List of colors
            size_ranges: List of (min_size, max_size) tuples for size categories
        """
        self.shape_types = shape_types or ["star", "circle"]
        self.colors = colors or ["red", "blue", "green", "yellow"]
        self.size_ranges = size_ranges or [(0.6, 0.8), (1.2, 1.5)]  # Small  # Large

    def generate_task_configs(self, num_configs: int = 25) -> List[SceneConfig]:
        """Generate size comparison task configurations."""
        configs = []

        for i in range(num_configs):
            # Create objects with different sizes
            size_category1 = random.choice([0, 1])  # Small or large
            size_category2 = random.choice([0, 1])

            size1 = random.uniform(*self.size_ranges[size_category1])
            size2 = random.uniform(*self.size_ranges[size_category2])

            obj1 = ObjectConfig(
                object_id="obj1",
                shape_type=random.choice(self.shape_types),
                color=random.choice(self.colors),
                size=size1,
            )

            obj2 = ObjectConfig(
                object_id="obj2",
                shape_type=random.choice(self.shape_types),
                color=random.choice(self.colors),
                size=size2,
            )

            # Random layout
            layout = LayoutConfig(
                layout_type="random", layout_params={"min_distance": 70}
            )

            # Task: is first object larger?
            is_larger = size1 > size2

            task = TaskConfig(
                task_type=TaskType.BINARY_CLASSIFICATION,
                num_samples=8,
                task_params={
                    "task_variant": "size_comparison",
                    "question": "Is the first object larger than the second?",
                    "expected_answer": is_larger,
                },
            )

            scene_config = SceneConfig(
                scene_id=f"size_comparison_{i}",
                objects=[obj1, obj2],
                layout=layout,
                task=task,
                metadata={
                    "obj1_size": size1,
                    "obj2_size": size2,
                    "obj1_larger": is_larger,
                    "size_diff": abs(size1 - size2),
                },
            )

            configs.append(scene_config)

        return configs

    def get_task_description(self) -> str:
        """Get description of size comparison task."""
        return "Size comparison task: determine which object is larger"


class MultiObjectTaskGenerator(TaskGenerator):
    """Generates complex multi-object spatial reasoning tasks."""

    def __init__(
        self,
        num_objects_range: Tuple[int, int] = (3, 5),
        shape_types: List[str] = None,
        colors: List[str] = None,
    ):
        """Initialize multi-object task generator.

        Args:
            num_objects_range: (min, max) number of objects per scene
            shape_types: Available shape types
            colors: Available colors
        """
        self.num_objects_range = num_objects_range
        self.shape_types = shape_types or ["star", "circle", "bar"]
        self.colors = colors or ["red", "blue", "green", "yellow", "purple"]

    def generate_task_configs(self, num_configs: int = 15) -> List[SceneConfig]:
        """Generate multi-object task configurations."""
        configs = []

        for i in range(num_configs):
            num_objects = random.randint(*self.num_objects_range)

            # Create diverse objects
            objects = []
            for j in range(num_objects):
                obj = ObjectConfig(
                    object_id=f"obj_{j}",
                    shape_type=random.choice(self.shape_types),
                    color=random.choice(self.colors),
                    size=random.uniform(0.7, 1.3),
                )
                objects.append(obj)

            # Use grid layout for systematic positioning
            layout = LayoutConfig(layout_type="grid", layout_params={"margin": 30})

            task = TaskConfig(
                task_type=TaskType.MULTICLASS_CLASSIFICATION,
                num_samples=5,
                task_params={
                    "task_variant": "multi_object_scene",
                    "classification_target": "object_count",
                },
            )

            scene_config = SceneConfig(
                scene_id=f"multi_object_{i}",
                objects=objects,
                layout=layout,
                task=task,
                metadata={
                    "object_count": num_objects,
                    "shape_distribution": {
                        shape: sum(1 for obj in objects if obj.shape_type == shape)
                        for shape in self.shape_types
                    },
                },
            )

            configs.append(scene_config)

        return configs

    def get_task_description(self) -> str:
        """Get description of multi-object task."""
        return "Multi-object spatial reasoning: analyze complex scenes with multiple objects"
