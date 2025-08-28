"""Configuration converters for backward compatibility."""

from typing import Tuple, Dict, Any
from tiled_dummy_gen.config.parser import ClassConfig, ExperimentConfig
from tiled_dummy_gen.config.scene_config import (
    SceneConfig,
    ObjectConfig,
    TaskConfig,
    TaskType,
    ExperimentSceneConfig,
    LayoutConfig,
)


class ClassConfigConverter:
    """Converts legacy ClassConfig to new SceneConfig format."""

    def convert_class_config(
        self, class_config: ClassConfig, image_size: Tuple[int, int] = (224, 224)
    ) -> SceneConfig:
        """Convert a single ClassConfig to SceneConfig.

        Args:
            class_config: Legacy ClassConfig to convert
            image_size: Image dimensions

        Returns:
            Equivalent SceneConfig
        """
        # Create bar object configuration
        bar_object = ObjectConfig(
            object_id="legacy_bar",
            shape_type="bar",
            color=(0, 0, 0),  # Bars are black in legacy system
            size=1.0,
            shape_params={
                "orientation": class_config.image_bar_orientation,
                "thickness": class_config.image_bar_thickness,
            },
            metadata={"legacy_object": True, "original_class_name": class_config.name},
        )

        # Create task configuration
        task_config = TaskConfig(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            num_samples=class_config.num_samples,
            augment_images=class_config.augment_images,
            augment_images_noise_level=class_config.augment_images_noise_level,
            augment_images_zoom_factor=class_config.augment_images_zoom_factor,
            augment_texts=class_config.augment_texts,
            task_params={
                "legacy_task": True,
                "classification_target": "bar_properties",
            },
        )

        # Create scene configuration
        scene_config = SceneConfig(
            scene_id=f"legacy_{class_config.name}",
            background_color=class_config.image_background_color,
            canvas_size=image_size,
            objects=[bar_object],
            layout=None,  # Bars don't need explicit layout (they span the canvas)
            task=task_config,
            metadata={
                "converted_from_legacy": True,
                "original_class_config": {
                    "name": class_config.name,
                    "background_color": class_config.image_background_color,
                    "bar_orientation": class_config.image_bar_orientation,
                    "bar_thickness": class_config.image_bar_thickness,
                },
            },
        )

        return scene_config


class ConfigConverter:
    """Converts complete legacy ExperimentConfig to new ExperimentSceneConfig."""

    def __init__(self):
        """Initialize converter with class config converter."""
        self.class_converter = ClassConfigConverter()

    def convert_experiment_config(
        self, experiment_config: ExperimentConfig
    ) -> ExperimentSceneConfig:
        """Convert legacy ExperimentConfig to new ExperimentSceneConfig.

        Args:
            experiment_config: Legacy experiment configuration

        Returns:
            New scene-based experiment configuration
        """
        # Convert each class config to scene config
        scene_configs = []
        for class_config in experiment_config.class_configs:
            scene_config = self.class_converter.convert_class_config(
                class_config, experiment_config.image_size
            )
            scene_configs.append(scene_config)

        # Convert embedder config to dict if present
        embedder_config_dict = None
        if experiment_config.embedder_config:
            embedder_config_dict = {
                "embedder_name": experiment_config.embedder_config.embedder_name,
                "device": experiment_config.embedder_config.device,
                "preprocessing": {
                    "resize": experiment_config.embedder_config.preprocessing.resize,
                    "resize_size": experiment_config.embedder_config.preprocessing.resize_size,
                    "normalize": experiment_config.embedder_config.preprocessing.normalize,
                    "normalization_mean": experiment_config.embedder_config.preprocessing.normalization_mean,
                    "normalization_std": experiment_config.embedder_config.preprocessing.normalization_std,
                    "additional_transforms": experiment_config.embedder_config.preprocessing.additional_transforms,
                },
            }

        # Convert dataset config to dict
        dataset_config_dict = {
            "hdf5_filename": experiment_config.dataset_config.hdf5_filename,
            "features_key": experiment_config.dataset_config.features_key,
            "positions_key": experiment_config.dataset_config.positions_key,
            "tile_keys_key": experiment_config.dataset_config.tile_keys_key,
            "text_key": experiment_config.dataset_config.text_key,
        }

        # Convert split config to dict
        split_config_dict = {
            "split": experiment_config.split_config.split,
            "train": experiment_config.split_config.train,
            "val": experiment_config.split_config.val,
            "test": experiment_config.split_config.test,
        }

        return ExperimentSceneConfig(
            scene_configs=scene_configs,
            embedder_config=embedder_config_dict,
            dataset_config=dataset_config_dict,
            split_config=split_config_dict,
            output_dir=experiment_config.output_dir,
            num_tiles_base=experiment_config.num_tiles_base,
            image_size=experiment_config.image_size,
            metadata={
                "converted_from_legacy_experiment": True,
                "original_class_count": len(experiment_config.class_configs),
            },
        )

    def create_legacy_compatible_config(self, scene_config: SceneConfig) -> ClassConfig:
        """Create a legacy ClassConfig from SceneConfig (reverse conversion).

        This is useful for maintaining compatibility when new scene configs
        need to work with legacy code that expects ClassConfig.

        Args:
            scene_config: Scene configuration to convert

        Returns:
            Legacy-compatible ClassConfig

        Raises:
            ValueError: If scene_config can't be converted to legacy format
        """
        # Check if this scene can be converted to legacy format
        if len(scene_config.objects) != 1:
            raise ValueError("Legacy format only supports single object scenes")

        obj = scene_config.objects[0]
        if obj.shape_type != "bar":
            raise ValueError("Legacy format only supports bar shapes")

        # Extract bar properties
        orientation = obj.shape_params.get("orientation", "horizontal")
        thickness = obj.shape_params.get("thickness", "medium")

        # Extract task properties
        num_samples = 10
        augment_images = True
        augment_images_noise_level = 10.0
        augment_images_zoom_factor = (1.0, 1.3)
        augment_texts = False

        if scene_config.task:
            num_samples = scene_config.task.num_samples
            augment_images = scene_config.task.augment_images
            augment_images_noise_level = scene_config.task.augment_images_noise_level
            augment_images_zoom_factor = scene_config.task.augment_images_zoom_factor
            augment_texts = scene_config.task.augment_texts

        # Create legacy ClassConfig
        class_config = ClassConfig(
            name=scene_config.scene_id,
            num_samples=num_samples,
            image_background_color=scene_config.background_color,
            image_bar_orientation=orientation,
            image_bar_thickness=thickness,
            augment_images=augment_images,
            augment_images_noise_level=augment_images_noise_level,
            augment_images_zoom_factor=augment_images_zoom_factor,
            augment_texts=augment_texts,
        )

        return class_config

    def extract_legacy_variations(self, scene_configs: list) -> Dict[str, Any]:
        """Extract legacy-style variations from a list of scene configs.

        Args:
            scene_configs: List of scene configurations

        Returns:
            Dictionary of variations in legacy format
        """
        variations = {}

        # Extract background colors
        background_colors = set()
        bar_orientations = set()
        bar_thicknesses = set()

        for scene_config in scene_configs:
            background_colors.add(scene_config.background_color)

            # Extract bar properties
            for obj in scene_config.objects:
                if obj.shape_type == "bar":
                    bar_orientations.add(
                        obj.shape_params.get("orientation", "horizontal")
                    )
                    bar_thicknesses.add(obj.shape_params.get("thickness", "medium"))

        if len(background_colors) > 1:
            variations["image_background_color"] = list(background_colors)
        if len(bar_orientations) > 1:
            variations["image_bar_orientation"] = list(bar_orientations)
        if len(bar_thicknesses) > 1:
            variations["image_bar_thickness"] = list(bar_thicknesses)

        return variations
