"""Preset functions for creating common spatial learning tasks.

These functions provide easy-to-use interfaces for generating
popular spatial reasoning scenarios without needing to configure
all the details manually.
"""

from typing import List, Dict, Any, Tuple
from tiled_dummy_gen.config.scene_config import ExperimentSceneConfig
from tiled_dummy_gen.scene.relationships import SpatialRelation
from tiled_dummy_gen.tasks.generators import (
    SpatialBinaryTaskGenerator,
    ColorComparisonTaskGenerator,
    SizeComparisonTaskGenerator,
    MultiObjectTaskGenerator,
)


def create_left_right_task(
    num_scenes: int = 50,
    shape_types: List[str] = None,
    colors: List[str] = None,
    canvas_size: Tuple[int, int] = (224, 224),
) -> ExperimentSceneConfig:
    """Create a left-right spatial classification task.

    Perfect for testing basic spatial reasoning: "Is object A to the left of object B?"

    Args:
        num_scenes: Number of scene configurations to generate
        shape_types: Object shapes to use
        colors: Object colors to use
        canvas_size: Image dimensions

    Returns:
        Complete experiment configuration
    """
    if shape_types is None:
        shape_types = ["star", "circle"]
    if colors is None:
        colors = ["red", "blue", "yellow", "green"]

    generator = SpatialBinaryTaskGenerator(
        target_relation=SpatialRelation.LEFT_OF,
        shape_types=shape_types,
        colors=colors,
        canvas_size=canvas_size,
    )

    scene_configs = generator.generate_task_configs(num_scenes)

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        output_dir="data/left_right_task",
        metadata={
            "task_name": "left_right_spatial_classification",
            "description": "Binary classification: is object A to the left of object B?",
            "target_relation": "left_of",
            "num_scenes": num_scenes,
        },
    )


def create_above_below_task(
    num_scenes: int = 50, shape_types: List[str] = None, colors: List[str] = None
) -> ExperimentSceneConfig:
    """Create an above-below spatial classification task.

    Tests vertical spatial reasoning: "Is object A above object B?"

    Args:
        num_scenes: Number of scene configurations
        shape_types: Object shapes to use
        colors: Object colors to use

    Returns:
        Complete experiment configuration
    """
    if shape_types is None:
        shape_types = ["star", "circle"]
    if colors is None:
        colors = ["red", "blue", "yellow", "green", "purple"]

    generator = SpatialBinaryTaskGenerator(
        target_relation=SpatialRelation.ABOVE, shape_types=shape_types, colors=colors
    )

    scene_configs = generator.generate_task_configs(num_scenes)

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        output_dir="data/above_below_task",
        metadata={
            "task_name": "above_below_spatial_classification",
            "description": "Binary classification: is object A above object B?",
            "target_relation": "above",
            "num_scenes": num_scenes,
        },
    )


def create_color_identification_task(
    num_scenes: int = 40, colors: List[str] = None
) -> ExperimentSceneConfig:
    """Create a color identification and comparison task.

    Tests color reasoning: "Are both objects the same color?"

    Args:
        num_scenes: Number of scene configurations
        colors: Colors to use in the task

    Returns:
        Complete experiment configuration
    """
    if colors is None:
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]

    generator = ColorComparisonTaskGenerator(colors=colors)
    scene_configs = generator.generate_task_configs(num_scenes)

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        output_dir="data/color_identification_task",
        metadata={
            "task_name": "color_identification",
            "description": "Color comparison: are both objects the same color?",
            "colors_used": colors,
            "num_scenes": num_scenes,
        },
    )


def create_size_comparison_task(num_scenes: int = 35) -> ExperimentSceneConfig:
    """Create a size comparison task.

    Tests size reasoning: "Which object is larger?"

    Args:
        num_scenes: Number of scene configurations

    Returns:
        Complete experiment configuration
    """
    generator = SizeComparisonTaskGenerator()
    scene_configs = generator.generate_task_configs(num_scenes)

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        output_dir="data/size_comparison_task",
        metadata={
            "task_name": "size_comparison",
            "description": "Size comparison: which object is larger?",
            "num_scenes": num_scenes,
        },
    )


def create_star_positioning_task(
    num_scenes: int = 60, colors: List[str] = None, include_size_variation: bool = True
) -> ExperimentSceneConfig:
    """Create a comprehensive star positioning task.

    Your specific use case: two stars with spatial relationships.
    Perfect for testing "which star is above/below/left/right of which".

    Args:
        num_scenes: Number of scene configurations
        colors: Star colors to use
        include_size_variation: Whether to vary star sizes

    Returns:
        Complete experiment configuration
    """
    if colors is None:
        colors = ["red", "blue", "yellow", "green", "purple", "orange"]

    # Create multiple task variants
    scene_configs = []

    # Left-right positioning task
    left_right_gen = SpatialBinaryTaskGenerator(
        target_relation=SpatialRelation.LEFT_OF,
        shape_types=["star"],  # Only stars
        colors=colors,
        num_samples_per_config=8,
    )
    scene_configs.extend(left_right_gen.generate_task_configs(num_scenes // 3))

    # Above-below positioning task
    above_below_gen = SpatialBinaryTaskGenerator(
        target_relation=SpatialRelation.ABOVE,
        shape_types=["star"],
        colors=colors,
        num_samples_per_config=8,
    )
    scene_configs.extend(above_below_gen.generate_task_configs(num_scenes // 3))

    # Color identification with stars
    color_gen = ColorComparisonTaskGenerator(colors=colors, shape_types=["star"])
    scene_configs.extend(color_gen.generate_task_configs(num_scenes // 3))

    # Add size variation if requested
    if include_size_variation:
        for config in scene_configs[-10:]:  # Vary size for last 10 configs
            for obj in config.objects:
                obj.size *= (
                    0.7 if obj.object_id == "obj1" else 1.3
                )  # Make obj1 smaller, obj2 larger

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        output_dir="data/star_positioning_task",
        metadata={
            "task_name": "star_positioning_comprehensive",
            "description": "Comprehensive star positioning task with multiple spatial relationships",
            "task_variants": ["left_right", "above_below", "color_identification"],
            "colors_used": colors,
            "include_size_variation": include_size_variation,
            "num_scenes": len(scene_configs),
        },
    )


def create_complex_spatial_reasoning_task(
    num_scenes: int = 30,
) -> ExperimentSceneConfig:
    """Create a complex spatial reasoning task with multiple objects.

    Advanced task with 3-5 objects per scene, testing complex spatial relationships.

    Args:
        num_scenes: Number of scene configurations

    Returns:
        Complete experiment configuration
    """
    generator = MultiObjectTaskGenerator(
        num_objects_range=(3, 5),
        shape_types=["star", "circle", "bar"],
        colors=["red", "blue", "green", "yellow", "purple", "orange"],
    )

    scene_configs = generator.generate_task_configs(num_scenes)

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        output_dir="data/complex_spatial_task",
        metadata={
            "task_name": "complex_spatial_reasoning",
            "description": "Complex multi-object spatial reasoning task",
            "object_count_range": [3, 5],
            "num_scenes": num_scenes,
        },
    )


# Convenience function for your specific use case
def create_two_star_binary_classification(
    relation: str = "left_of", num_scenes: int = 100, colors: List[str] = None
) -> ExperimentSceneConfig:
    """Create your specific two-star binary classification task.

    This is perfect for your example: "two stars, binary classification
    task where we teach the model which star is above which one."

    Args:
        relation: Spatial relation to test ("left_of", "right_of", "above", "below")
        num_scenes: Number of scenes to generate
        colors: Star colors to use

    Returns:
        Complete experiment configuration ready to use
    """
    if colors is None:
        colors = ["red", "blue", "yellow", "green", "purple"]

    # Map string to enum
    relation_map = {
        "left_of": SpatialRelation.LEFT_OF,
        "right_of": SpatialRelation.RIGHT_OF,
        "above": SpatialRelation.ABOVE,
        "below": SpatialRelation.BELOW,
    }

    spatial_relation = relation_map.get(relation, SpatialRelation.LEFT_OF)

    generator = SpatialBinaryTaskGenerator(
        target_relation=spatial_relation,
        shape_types=["star"],  # Only stars as requested
        colors=colors,
        num_samples_per_config=15,  # More samples per configuration
    )

    scene_configs = generator.generate_task_configs(num_scenes)

    return ExperimentSceneConfig(
        scene_configs=scene_configs,
        embedder_config={
            "embedder_name": "google/vit-base-patch16-224",
            "device": "cpu",
        },
        dataset_config={"hdf5_filename": f"two_star_{relation}_dataset.hdf5"},
        split_config={"split": True, "train": 0.7, "val": 0.2, "test": 0.1},
        output_dir=f"data/two_star_{relation}_task",
        image_size=(224, 224),
        metadata={
            "task_name": f"two_star_{relation}_binary_classification",
            "description": f'Binary classification: determine if one star is {relation.replace("_", " ")} the other star',
            "target_relation": relation,
            "colors_used": colors,
            "num_scenes": num_scenes,
            "designed_for": "spatial_feature_learning",
        },
    )
