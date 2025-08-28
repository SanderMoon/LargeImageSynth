"""New flexible configuration system for scene-based generation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum
import json
import os


class TaskType(Enum):
    """Types of machine learning tasks supported."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    SPATIAL_RELATIONSHIP = "spatial_relationship"
    OBJECT_DETECTION = "object_detection"


@dataclass
class ObjectConfig:
    """Configuration for a single object in a scene."""

    object_id: str
    shape_type: str  # "bar", "star", "circle", etc.
    color: Union[str, Tuple[int, int, int]]  # Color name or RGB tuple
    size: float = 1.0  # Size scaling factor

    # Shape-specific parameters (stored as dict for flexibility)
    shape_params: Dict[str, Any] = field(default_factory=dict)

    # Object metadata for ML tasks
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration."""
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")

        # Convert color names to RGB if needed (done in factory/generator)
        if isinstance(self.color, str):
            self.color = self.color.lower()


@dataclass
class LayoutConfig:
    """Configuration for spatial layout of objects."""

    layout_type: str  # "random", "grid", "relational"

    # Layout-specific parameters
    layout_params: Dict[str, Any] = field(default_factory=dict)

    # For relational layouts: spatial relationships
    relationships: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Validate layout configuration."""
        valid_layouts = {"random", "grid", "relational"}
        if self.layout_type not in valid_layouts:
            raise ValueError(
                f"Invalid layout_type: {self.layout_type}. Must be one of {valid_layouts}"
            )


@dataclass
class TaskConfig:
    """Configuration for ML task generation."""

    task_type: TaskType
    num_samples: int = 10

    # Task-specific parameters
    task_params: Dict[str, Any] = field(default_factory=dict)

    # Augmentation settings
    augment_images: bool = True
    augment_images_noise_level: float = 10.0
    augment_images_zoom_factor: Tuple[float, float] = (1.0, 1.3)
    augment_texts: bool = False

    def __post_init__(self):
        """Validate task configuration."""
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if self.augment_images_noise_level < 0:
            raise ValueError(
                f"Noise level must be non-negative, got {self.augment_images_noise_level}"
            )


@dataclass
class SceneConfig:
    """Configuration for generating a scene with multiple objects."""

    scene_id: str
    background_color: Union[str, Tuple[int, int, int]] = "white"
    canvas_size: Tuple[int, int] = (224, 224)

    # Objects in the scene
    objects: List[ObjectConfig] = field(default_factory=list)

    # Spatial layout configuration
    layout: Optional[LayoutConfig] = None

    # Task configuration
    task: Optional[TaskConfig] = None

    # Scene-level metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate scene configuration."""
        if not self.objects:
            raise ValueError("Scene must contain at least one object")

        # Check for duplicate object IDs
        object_ids = [obj.object_id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Object IDs must be unique within a scene")

        # Ensure canvas size is valid
        width, height = self.canvas_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Canvas size must be positive, got {self.canvas_size}")

    def add_object(self, obj_config: ObjectConfig) -> None:
        """Add an object to the scene.

        Args:
            obj_config: Object configuration to add

        Raises:
            ValueError: If object ID already exists
        """
        existing_ids = {obj.object_id for obj in self.objects}
        if obj_config.object_id in existing_ids:
            raise ValueError(f"Object ID '{obj_config.object_id}' already exists")

        self.objects.append(obj_config)


@dataclass
class ExperimentSceneConfig:
    """Top-level configuration for scene-based experiments."""

    # Scene configurations - can have multiple scene templates
    scene_configs: List[SceneConfig]

    # Generation settings
    variations: Dict[str, List[Any]] = field(default_factory=dict)

    # Embedder configuration (reusing existing)
    embedder_config: Optional[Dict[str, Any]] = None

    # Dataset configuration (reusing existing)
    dataset_config: Optional[Dict[str, Any]] = None

    # Split configuration (reusing existing)
    split_config: Optional[Dict[str, Any]] = None

    # Output settings
    output_dir: str = "output"
    num_tiles_base: int = 1
    image_size: Tuple[int, int] = (224, 224)

    # Global metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experiment configuration."""
        if not self.scene_configs:
            raise ValueError("Experiment must contain at least one scene configuration")

        # Check for duplicate scene IDs
        scene_ids = [scene.scene_id for scene in self.scene_configs]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("Scene IDs must be unique within an experiment")


class SceneConfigLoader:
    """Loader for scene-based JSON configurations.

    Supports both new scene-based format and backward compatibility
    with existing bar-based configurations.
    """

    def __init__(self, config_path: str):
        """Initialize config loader.

        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self._config_data: Optional[Dict[str, Any]] = None

    def load_config(self) -> ExperimentSceneConfig:
        """Load and parse configuration from JSON.

        Returns:
            Parsed experiment configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config_data = json.load(f)

        # Determine if this is a legacy or new format
        if self._is_legacy_format():
            return self._convert_legacy_config()
        else:
            return self._parse_scene_config()

    def _is_legacy_format(self) -> bool:
        """Check if this is a legacy bar-based configuration.

        Returns:
            True if legacy format, False if new scene format
        """
        # Legacy format has 'class_template' key
        return "class_template" in self._config_data

    def _convert_legacy_config(self) -> ExperimentSceneConfig:
        """Convert legacy bar configuration to scene format.

        Returns:
            Converted scene configuration
        """
        data = self._config_data

        # Extract legacy class template
        class_template = data["class_template"]
        variations = data.get("variations", {})

        # Create scene configurations for each variation combination
        scene_configs = []

        # Generate combinations from variations
        if variations:
            import itertools

            variation_keys = list(variations.keys())
            variation_values = list(variations.values())

            for combination in itertools.product(*variation_values):
                variation_dict = dict(zip(variation_keys, combination))
                scene_config = self._create_legacy_scene_config(
                    class_template, variation_dict, len(scene_configs)
                )
                scene_configs.append(scene_config)
        else:
            # No variations, create single scene
            scene_config = self._create_legacy_scene_config(class_template, {}, 0)
            scene_configs.append(scene_config)

        return ExperimentSceneConfig(
            scene_configs=scene_configs,
            embedder_config=data.get("embedder_config"),
            dataset_config=data.get("dataset_config"),
            split_config=data.get("split_config"),
            output_dir=data.get("output_dir", "output"),
            num_tiles_base=data.get("num_tiles_base", 1),
            image_size=tuple(data.get("image_size", [224, 224])),
            metadata={"converted_from": "legacy_format"},
        )

    def _create_legacy_scene_config(
        self, class_template: Dict[str, Any], variations: Dict[str, Any], scene_idx: int
    ) -> SceneConfig:
        """Create a scene config from legacy class template.

        Args:
            class_template: Legacy class template
            variations: Variation values for this scene
            scene_idx: Index for unique scene ID

        Returns:
            Scene configuration
        """
        # Merge template with variations
        merged_config = {**class_template, **variations}

        # Create single bar object
        bar_object = ObjectConfig(
            object_id="bar_0",
            shape_type="bar",
            color=merged_config.get(
                "image_background_color", "white"
            ),  # This is actually bar color
            shape_params={
                "orientation": merged_config.get("image_bar_orientation", "horizontal"),
                "thickness": merged_config.get("image_bar_thickness", "medium"),
            },
        )

        # Create task config
        task_config = TaskConfig(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            num_samples=merged_config.get("num_samples", 10),
            augment_images=merged_config.get("augment_images", False),
            augment_images_noise_level=merged_config.get(
                "augment_images_noise_level", 0.0
            ),
            augment_images_zoom_factor=tuple(
                merged_config.get("augment_images_zoom_factor", [1.0, 1.0])
            ),
            augment_texts=merged_config.get("augment_texts", False),
        )

        # Determine background color (bars use background_color, objects are black)
        bg_color = merged_config.get("image_background_color", "white")
        bar_object.color = (0, 0, 0)  # Bars are black in legacy system

        return SceneConfig(
            scene_id=f"legacy_scene_{scene_idx}",
            background_color=bg_color,
            objects=[bar_object],
            task=task_config,
            metadata={
                "legacy_class_name": f"{variations.get('image_background_color', 'default')}_bar",
                "variations": variations,
            },
        )

    def _parse_scene_config(self) -> ExperimentSceneConfig:
        """Parse new scene-based configuration format.

        Returns:
            Parsed scene configuration
        """
        data = self._config_data

        # Parse scene configurations
        scene_configs = []
        for scene_data in data["scenes"]:
            scene_config = self._parse_single_scene(scene_data)
            scene_configs.append(scene_config)

        return ExperimentSceneConfig(
            scene_configs=scene_configs,
            variations=data.get("variations", {}),
            embedder_config=data.get("embedder_config"),
            dataset_config=data.get("dataset_config"),
            split_config=data.get("split_config"),
            output_dir=data.get("output_dir", "output"),
            num_tiles_base=data.get("num_tiles_base", 1),
            image_size=tuple(data.get("image_size", [224, 224])),
            metadata=data.get("metadata", {}),
        )

    def _parse_single_scene(self, scene_data: Dict[str, Any]) -> SceneConfig:
        """Parse a single scene configuration.

        Args:
            scene_data: Scene data from JSON

        Returns:
            Parsed scene configuration
        """
        # Parse objects
        objects = []
        for obj_data in scene_data["objects"]:
            obj_config = ObjectConfig(
                object_id=obj_data["object_id"],
                shape_type=obj_data["shape_type"],
                color=obj_data.get("color", "black"),
                size=obj_data.get("size", 1.0),
                shape_params=obj_data.get("shape_params", {}),
                metadata=obj_data.get("metadata", {}),
            )
            objects.append(obj_config)

        # Parse layout if present
        layout = None
        if "layout" in scene_data:
            layout_data = scene_data["layout"]
            layout = LayoutConfig(
                layout_type=layout_data["layout_type"],
                layout_params=layout_data.get("layout_params", {}),
                relationships=layout_data.get("relationships", []),
            )

        # Parse task if present
        task = None
        if "task" in scene_data:
            task_data = scene_data["task"]
            task = TaskConfig(
                task_type=TaskType(task_data["task_type"]),
                num_samples=task_data.get("num_samples", 10),
                task_params=task_data.get("task_params", {}),
                augment_images=task_data.get("augment_images", True),
                augment_images_noise_level=task_data.get(
                    "augment_images_noise_level", 10.0
                ),
                augment_images_zoom_factor=tuple(
                    task_data.get("augment_images_zoom_factor", [1.0, 1.3])
                ),
                augment_texts=task_data.get("augment_texts", False),
            )

        return SceneConfig(
            scene_id=scene_data["scene_id"],
            background_color=scene_data.get("background_color", "white"),
            canvas_size=tuple(scene_data.get("canvas_size", [224, 224])),
            objects=objects,
            layout=layout,
            task=task,
            metadata=scene_data.get("metadata", {}),
        )
