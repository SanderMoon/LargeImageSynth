"""Example showing how to use the new scene-based API for spatial feature learning.

This demonstrates your specific use case: generating binary classification tasks
where ML models learn spatial relationships between objects (like stars).
"""

from tiled_dummy_gen.tasks.presets import create_two_star_binary_classification
from tiled_dummy_gen.core.scene_generator import SceneGenerator
from tiled_dummy_gen.config.scene_config import SceneConfigLoader


def example_1_using_presets():
    """Example 1: Using convenience presets for common spatial learning tasks."""
    print("=== Example 1: Using Presets for Two-Star Binary Classification ===")

    # Create a binary classification task: "which star is to the left of which"
    experiment_config = create_two_star_binary_classification(
        relation="left_of",  # Can be "left_of", "right_of", "above", "below"
        num_scenes=50,
        colors=["red", "blue", "yellow", "green", "purple"],
    )

    print(
        f"Created experiment with {len(experiment_config.scene_configs)} scene configurations"
    )
    print(f"Output directory: {experiment_config.output_dir}")
    print(f"Task: {experiment_config.metadata['description']}")

    # Generate a sample image and description
    generator = SceneGenerator()
    scene_config = experiment_config.scene_configs[0]  # Take first config

    image, text, metadata = generator.generate_sample(scene_config)
    print(f"Sample text: {text}")
    print(f"Spatial relationships: {metadata['spatial_relationships']}")

    # Save the sample image
    import os

    os.makedirs("output/examples", exist_ok=True)
    image.save("output/examples/sample_two_stars.png")
    print("Saved sample image to: output/examples/sample_two_stars.png")


def example_2_using_json_config():
    """Example 2: Using JSON configuration files."""
    print("\n=== Example 2: Using JSON Configuration ===")

    # Load configuration from JSON file
    config_path = "examples/experiments/spatial_stars_example.json"
    loader = SceneConfigLoader(config_path)
    experiment_config = loader.load_config()

    print(
        f"Loaded experiment: {experiment_config.metadata.get('experiment_name', 'Unknown')}"
    )
    print(
        f"Description: {experiment_config.metadata.get('description', 'No description')}"
    )

    # Generate samples from each scene configuration
    generator = SceneGenerator()

    for i, scene_config in enumerate(experiment_config.scene_configs):
        print(f"\nScene {i+1}: {scene_config.scene_id}")

        # Generate multiple samples
        samples = generator.batch_generate(scene_config, num_samples=3)

        for j, (image, text, metadata) in enumerate(samples):
            print(f"  Sample {j+1}: {text}")

            # Save sample
            filename = f"output/examples/{scene_config.scene_id}_sample_{j+1}.png"
            os.makedirs("output/examples", exist_ok=True)
            image.save(filename)


def example_3_programmatic_scene_creation():
    """Example 3: Creating scenes programmatically."""
    print("\n=== Example 3: Programmatic Scene Creation ===")

    from tiled_dummy_gen.config.scene_config import (
        SceneConfig,
        ObjectConfig,
        LayoutConfig,
        TaskConfig,
        TaskType,
    )

    # Create two star objects
    star1 = ObjectConfig(
        object_id="left_star",
        shape_type="star",
        color="red",
        size=1.2,
        shape_params={"num_points": 6},  # 6-pointed star
    )

    star2 = ObjectConfig(
        object_id="right_star",
        shape_type="star",
        color="blue",
        size=0.8,
        shape_params={"num_points": 5},  # 5-pointed star
    )

    # Create relational layout
    layout = LayoutConfig(
        layout_type="relational",
        relationships=[
            {"object1": "left_star", "object2": "right_star", "relation": "left_of"}
        ],
    )

    # Create task configuration
    task = TaskConfig(
        task_type=TaskType.BINARY_CLASSIFICATION,
        num_samples=5,
        task_params={
            "target_relation": "left_of",
            "question": "Is the red star to the left of the blue star?",
        },
    )

    # Create scene configuration
    scene_config = SceneConfig(
        scene_id="programmatic_example",
        background_color="lightgray",
        objects=[star1, star2],
        layout=layout,
        task=task,
    )

    # Generate samples
    generator = SceneGenerator()
    samples = generator.batch_generate(scene_config)

    print(f"Generated {len(samples)} samples")
    for i, (image, text, metadata) in enumerate(samples):
        print(f"Sample {i+1}: {text}")
        filename = f"output/examples/programmatic_sample_{i+1}.png"
        os.makedirs("output/examples", exist_ok=True)
        image.save(filename)


def example_4_backward_compatibility():
    """Example 4: Backward compatibility - old API still works."""
    print("\n=== Example 4: Backward Compatibility ===")

    # Old API continues to work unchanged
    from tiled_dummy_gen import SyntheticDataGenerator, ClassConfig

    # Create legacy configuration
    class_config = ClassConfig(
        name="blue_horizontal_bar",
        num_samples=3,
        image_background_color="blue",
        image_bar_orientation="horizontal",
        image_bar_thickness="thick",
        augment_images=True,
        augment_images_noise_level=5.0,
        augment_images_zoom_factor=(1.0, 1.2),
        augment_texts=False,
    )

    # Use legacy generator (automatically uses new system internally)
    generator = SyntheticDataGenerator()

    for i in range(3):
        image, text = generator.generate_sample(class_config)
        print(f"Legacy sample {i+1}: {text}")

        filename = f"output/examples/legacy_sample_{i+1}.png"
        os.makedirs("output/examples", exist_ok=True)
        image.save(filename)

    print("Legacy API works seamlessly with new architecture!")


if __name__ == "__main__":
    # Run all examples
    example_1_using_presets()
    example_2_using_json_config()
    example_3_programmatic_scene_creation()
    example_4_backward_compatibility()

    print("\n" + "=" * 60)
    print(
        "All examples completed! Check the output/examples/ directory for generated images."
    )
    print("The new architecture enables:")
    print("✓ Multi-object scenes with spatial relationships")
    print("✓ Extensible shape system (stars, circles, bars, etc.)")
    print("✓ Task-oriented generation for ML experiments")
    print("✓ Full backward compatibility with existing code")
    print("✓ Easy expansion for new spatial learning scenarios")
