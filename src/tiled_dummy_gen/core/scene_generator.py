"""Scene-based generator for multi-object synthetic data generation."""

import random
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageEnhance
import numpy as np

from tiled_dummy_gen.config.scene_config import (
    SceneConfig,
    ObjectConfig,
    LayoutConfig,
    TaskConfig,
    ExperimentSceneConfig,
    TaskType,
)
from tiled_dummy_gen.shapes.factory import ShapeFactory, default_factory
from tiled_dummy_gen.scene.objects import Scene, SceneObject
from tiled_dummy_gen.scene.layout import (
    SpatialLayout,
    RandomLayout,
    GridLayout,
    RelationalLayout,
)
from tiled_dummy_gen.scene.relationships import RelationshipAnalyzer


class SceneGenerator:
    """Generator for creating scenes with multiple objects and spatial relationships.

    This replaces the old SyntheticDataGenerator with a more flexible,
    composition-based approach that supports:
    - Multiple objects per scene
    - Spatial relationship learning
    - Various ML task types
    - Extensible shape system
    """

    def __init__(
        self, shape_factory: Optional[ShapeFactory] = None, noise_level: float = 10.0
    ):
        """Initialize the scene generator.

        Args:
            shape_factory: Factory for creating shapes (uses default if None)
            noise_level: Default noise level for image augmentation
        """
        self.shape_factory = shape_factory or default_factory
        self.noise_level = noise_level
        self.relationship_analyzer = RelationshipAnalyzer()

        # Color name to RGB mapping (expanded from original)
        self.color_map = {
            "blue": (0, 0, 255),
            "red": (255, 0, 0),
            "green": (34, 139, 34),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "black": (0, 0, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "lime": (0, 255, 0),
            "pink": (255, 192, 203),
            "teal": (0, 128, 128),
            "lavender": (230, 230, 250),
            "brown": (165, 42, 42),
            "beige": (245, 245, 220),
            "maroon": (128, 0, 0),
            "navy": (0, 0, 128),
        }

    def generate_scene(self, scene_config: SceneConfig) -> Tuple[Scene, str]:
        """Generate a scene from configuration.

        Args:
            scene_config: Scene configuration

        Returns:
            Tuple of (Scene object, text description)
        """
        # Create scene with background
        bg_color = self._resolve_color(scene_config.background_color)
        scene = Scene(background_color=bg_color, canvas_size=scene_config.canvas_size)
        scene.metadata.update(scene_config.metadata)

        # Create objects from configuration
        for obj_config in scene_config.objects:
            scene_object = self._create_scene_object(obj_config)
            scene.add_object(scene_object)

        # Apply spatial layout
        if scene_config.layout:
            layout_strategy = self._create_layout_strategy(scene_config.layout)
            layout_strategy.position_objects(scene.objects, scene.canvas_size)
        else:
            # Default random layout
            default_layout = RandomLayout()
            default_layout.position_objects(scene.objects, scene.canvas_size)

        # Generate text description
        text_description = self._generate_scene_description(scene, scene_config)

        return scene, text_description

    def generate_image(
        self, scene: Scene, augment: bool = False, **kwargs
    ) -> Image.Image:
        """Render scene to an image with optional augmentation.

        Args:
            scene: Scene to render
            augment: Whether to apply image augmentations
            **kwargs: Augmentation parameters

        Returns:
            Rendered PIL Image
        """
        # Render the scene
        image = scene.render()

        if augment:
            image = self._apply_augmentations(image, **kwargs)

        return image

    def generate_sample(
        self, scene_config: SceneConfig
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        """Generate a complete sample (image + text + metadata).

        Args:
            scene_config: Scene configuration

        Returns:
            Tuple of (Image, text description, metadata dict)
        """
        # Generate scene
        scene, text_description = self.generate_scene(scene_config)

        # Determine augmentation settings
        augment = False
        aug_params = {}
        if scene_config.task:
            augment = scene_config.task.augment_images
            aug_params = {
                "noise_level": scene_config.task.augment_images_noise_level,
                "zoom_factor_range": scene_config.task.augment_images_zoom_factor,
            }

        # Render image
        image = self.generate_image(scene, augment=augment, **aug_params)

        # Collect metadata
        metadata = {
            "scene_id": scene_config.scene_id,
            "object_count": len(scene_config.objects),
            "canvas_size": scene_config.canvas_size,
            "background_color": scene_config.background_color,
            "objects": [
                {
                    "object_id": obj.object_id,
                    "shape_type": obj.shape_type,
                    "color": obj.color,
                    "size": obj.size,
                    "position": getattr(
                        scene.get_object(obj.object_id), "position", None
                    ),
                }
                for obj in scene_config.objects
            ],
        }

        # Add spatial relationships
        relationships = scene.get_spatial_relationships()
        metadata["spatial_relationships"] = relationships

        # Add task-specific metadata
        if scene_config.task:
            task_metadata = self._generate_task_metadata(scene, scene_config.task)
            metadata["task"] = task_metadata

        return image, text_description, metadata

    def batch_generate(
        self, scene_config: SceneConfig, num_samples: Optional[int] = None
    ) -> List[Tuple[Image.Image, str, Dict[str, Any]]]:
        """Generate multiple samples from the same scene configuration.

        Args:
            scene_config: Scene configuration
            num_samples: Number of samples (uses task config if None)

        Returns:
            List of (image, text, metadata) tuples
        """
        if num_samples is None and scene_config.task:
            num_samples = scene_config.task.num_samples
        elif num_samples is None:
            num_samples = 1

        samples = []
        for i in range(num_samples):
            # Create slightly varied scene for each sample
            varied_config = self._create_varied_config(scene_config, i)
            sample = self.generate_sample(varied_config)
            samples.append(sample)

        return samples

    def _create_scene_object(self, obj_config: ObjectConfig) -> SceneObject:
        """Create a SceneObject from ObjectConfig.

        Args:
            obj_config: Object configuration

        Returns:
            Created SceneObject
        """
        # Resolve color
        color = self._resolve_color(obj_config.color)

        # Create shape using factory
        shape = self.shape_factory.create_shape(
            shape_type=obj_config.shape_type,
            color=color,
            size=obj_config.size,
            **obj_config.shape_params,
        )

        # Create scene object (position will be set by layout strategy)
        scene_object = SceneObject(
            shape=shape,
            position=(0, 0),  # Placeholder, will be set by layout
            object_id=obj_config.object_id,
            metadata=obj_config.metadata.copy(),
        )

        return scene_object

    def _resolve_color(self, color: Any) -> Tuple[int, int, int]:
        """Resolve color specification to RGB tuple.

        Args:
            color: Color name string or RGB tuple

        Returns:
            RGB tuple
        """
        if isinstance(color, str):
            color_lower = color.lower()
            if color_lower in self.color_map:
                return self.color_map[color_lower]
            else:
                # Default to black for unknown colors
                return (0, 0, 0)
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(color)
        else:
            # Invalid color specification
            return (0, 0, 0)

    def _create_layout_strategy(self, layout_config: LayoutConfig) -> SpatialLayout:
        """Create layout strategy from configuration.

        Args:
            layout_config: Layout configuration

        Returns:
            Layout strategy instance
        """
        layout_type = layout_config.layout_type.lower()
        params = layout_config.layout_params

        if layout_type == "random":
            return RandomLayout(**params)
        elif layout_type == "grid":
            return GridLayout(**params)
        elif layout_type == "relational":
            return RelationalLayout(relationships=layout_config.relationships, **params)
        else:
            # Default to random if unknown
            return RandomLayout()

    def _apply_augmentations(
        self,
        image: Image.Image,
        noise_level: float = None,
        zoom_factor_range: Tuple[float, float] = (1.0, 1.0),
        **kwargs,
    ) -> Image.Image:
        """Apply image augmentations.

        Args:
            image: Input image
            noise_level: Gaussian noise standard deviation
            zoom_factor_range: (min_zoom, max_zoom) for random zoom
            **kwargs: Additional augmentation parameters

        Returns:
            Augmented image
        """
        if noise_level is None:
            noise_level = self.noise_level

        # Convert to numpy for noise application
        image_array = np.array(image)

        # Apply Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image_array.shape).astype(np.int16)
            noisy_image = image_array.astype(np.int16) + noise
            image_array = np.clip(noisy_image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_array)

        # Apply zoom augmentation
        min_zoom, max_zoom = zoom_factor_range
        if max_zoom > min_zoom:
            zoom_factor = random.uniform(min_zoom, max_zoom)
            width, height = image.size
            new_size = (int(width * zoom_factor), int(height * zoom_factor))

            # Resize and crop back to original size
            image = image.resize(new_size, Image.BICUBIC)
            left = (new_size[0] - width) // 2
            top = (new_size[1] - height) // 2
            image = image.crop((left, top, left + width, top + height))

        # Apply brightness variation
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = random.uniform(0.8, 1.2)
        image = enhancer.enhance(brightness_factor)

        return image

    def _generate_scene_description(
        self, scene: Scene, scene_config: SceneConfig
    ) -> str:
        """Generate text description for the scene.

        Args:
            scene: Generated scene
            scene_config: Scene configuration

        Returns:
            Text description
        """
        # Use relationship analyzer for spatial descriptions
        description = self.relationship_analyzer.generate_relationship_description(
            scene
        )

        # Add task-specific description variations if configured
        if scene_config.task and scene_config.task.augment_texts:
            # Add some variation to the description
            variations = [
                description,
                description.replace("A scene with", "This image shows"),
                description.replace("A scene with", "The picture contains"),
                f"In this scene, there is {description[13:]}",  # Remove "A scene with "
            ]
            description = random.choice(variations)

        return description

    def _generate_task_metadata(
        self, scene: Scene, task_config: TaskConfig
    ) -> Dict[str, Any]:
        """Generate task-specific metadata.

        Args:
            scene: Generated scene
            task_config: Task configuration

        Returns:
            Task metadata dictionary
        """
        metadata = {
            "task_type": task_config.task_type.value,
            "task_params": task_config.task_params.copy(),
        }

        if task_config.task_type == TaskType.SPATIAL_RELATIONSHIP:
            # Add spatial relationship analysis
            relationships = self.relationship_analyzer.analyze_scene(scene)
            metadata["detected_relationships"] = relationships

            if len(scene.objects) == 2:
                # Binary spatial classification
                primary_relations = (
                    self.relationship_analyzer.get_primary_relationships(scene, 1)
                )
                if primary_relations:
                    metadata["primary_relationship"] = primary_relations[0]

        elif task_config.task_type == TaskType.BINARY_CLASSIFICATION:
            # Generate binary classification labels based on task parameters
            if "target_relation" in task_config.task_params:
                from tiled_dummy_gen.scene.relationships import SpatialRelation

                target_relation = SpatialRelation(
                    task_config.task_params["target_relation"]
                )
                labels = self.relationship_analyzer.create_binary_classification_labels(
                    scene, target_relation
                )
                metadata["classification_labels"] = labels

        return metadata

    def _create_varied_config(
        self, scene_config: SceneConfig, sample_idx: int
    ) -> SceneConfig:
        """Create a slightly varied version of the scene config for sample diversity.

        Args:
            scene_config: Original scene configuration
            sample_idx: Sample index for reproducible variation

        Returns:
            Varied scene configuration
        """
        # For now, return the same config
        # In the future, this could introduce small variations in:
        # - Object positions (for random layouts)
        # - Object properties (slight color/size variations)
        # - Scene metadata

        # Create a copy with unique scene_id
        import copy

        varied_config = copy.deepcopy(scene_config)
        varied_config.scene_id = f"{scene_config.scene_id}_sample_{sample_idx}"

        return varied_config


class SceneBasedPipeline:
    """Pipeline for scene-based synthetic data generation.
    
    This provides the same interface as SyntheticDataPipeline but uses
    the new scene-based configuration system.
    """

    def __init__(self, config: ExperimentSceneConfig):
        """Initialize the scene-based pipeline.
        
        Args:
            config: Experiment scene configuration
        """
        self.config = config
        self.generator = SceneGenerator()
        self._data = []
        
    def run(self):
        """Generate synthetic data without embeddings."""
        import pandas as pd
        from pathlib import Path
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_records = []
        
        for scene_config in self.config.scene_configs:
            # Generate samples for this scene
            samples = self.generator.batch_generate(scene_config)
            
            for i, (image, description, metadata) in enumerate(samples):
                if self.config.num_tiles_base > 1:
                    # Save the original large image
                    full_image_filename = f"{scene_config.scene_id}_sample_{i:04d}_full.png"
                    full_image_path = output_dir / full_image_filename
                    image.save(full_image_path)
                    
                    # Tile the large image and create records for each tile
                    tiles = self._tile_image(image, self.config.num_tiles_base)
                    
                    for tile_idx, (tile_image, tile_x, tile_y) in enumerate(tiles):
                        # Save tile
                        tile_filename = f"{scene_config.scene_id}_sample_{i:04d}_tile_{tile_x}_{tile_y}.png"
                        tile_path = output_dir / tile_filename
                        tile_image.save(tile_path)
                        
                        # Create data record for this tile
                        record = {
                            "filename": tile_filename,
                            "label": scene_config.scene_id,
                            "description": description,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "full_image_filename": full_image_filename,
                            **metadata
                        }
                        data_records.append(record)
                else:
                    # Save full image (single tile mode)
                    filename = f"{scene_config.scene_id}_sample_{i:04d}.png"
                    image_path = output_dir / filename
                    image.save(image_path)
                    
                    # Create data record
                    record = {
                        "filename": filename,
                        "label": scene_config.scene_id,
                        "description": description,
                        "tile_x": 0,
                        "tile_y": 0,
                        **metadata
                    }
                    data_records.append(record)
        
        # Convert to DataFrame
        self._data = pd.DataFrame(data_records)
        
    def run_with_embedding(self):
        """Generate synthetic data and embeddings."""
        # First generate the data
        self.run()
        
        # Then embed images
        self.embed_images()
        
    def embed_images(self):
        """Generate embeddings for existing images."""
        if self._data.empty:
            raise ValueError("No data to embed. Run generate first.")
            
        from tiled_dummy_gen.core.embedder import ImageEmbedder
        from pathlib import Path
        
        # Initialize embedder
        from tiled_dummy_gen.config.parser import EmbedderConfig, PreprocessingConfig
        
        embedder_config_dict = self.config.embedder_config
        preprocessing_dict = embedder_config_dict.get("preprocessing", {})
        
        # Create config objects
        preprocessing_config = PreprocessingConfig(
            resize=preprocessing_dict.get("resize", False),
            resize_size=preprocessing_dict.get("resize_size"),
            normalize=preprocessing_dict.get("normalize", False),
            normalization_mean=preprocessing_dict.get("normalization_mean", [0.485, 0.456, 0.406]),
            normalization_std=preprocessing_dict.get("normalization_std", [0.229, 0.224, 0.225]),
            additional_transforms=preprocessing_dict.get("additional_transforms")
        )
        
        embedder_config = EmbedderConfig(
            embedder_name=embedder_config_dict["embedder_name"],
            device=embedder_config_dict.get("device", "cpu"),
            preprocessing=preprocessing_config
        )
        
        embedder = ImageEmbedder(embedder_config)
        
        # Embed all images
        output_dir = Path(self.config.output_dir)
        
        embeddings_data = []
        for filename in self._data['filename']:
            image_path = output_dir / filename
            image = Image.open(image_path)
            embedding = embedder.embed_image(image)
            
            # Create embedding record
            embedding_record = {'filename': filename}
            for i, value in enumerate(embedding):
                embedding_record[f'embedding_{i}'] = value
            
            embeddings_data.append(embedding_record)
        
        # Create embedding DataFrame and merge
        import pandas as pd
        embedding_df = pd.DataFrame(embeddings_data)
        
        self._data = pd.merge(
            self._data, 
            embedding_df, 
            on='filename', 
            how='left'
        )
        
    def save_data(self, format_type: str = "hdf5"):
        """Save data in specified format.
        
        Args:
            format_type: Output format ("hdf5", "files", "webdataset")
        """
        if self._data.empty:
            raise ValueError("No data to save. Run generate first.")
            
        if format_type == "hdf5":
            from tiled_dummy_gen.export.hdf5_exporter import HDF5Exporter
            from types import SimpleNamespace
            
            # Convert dict configs to objects for compatibility
            dataset_config = SimpleNamespace(**self.config.dataset_config)
            split_config = SimpleNamespace(**self.config.split_config)
            
            exporter = HDF5Exporter(
                output_dir=self.config.output_dir,
                dataset_config=dataset_config,
                split_config=split_config,
                num_tiles_base=self.config.num_tiles_base
            )
            exporter.export(self._data)
            
        elif format_type == "webdataset":
            from tiled_dummy_gen.export.webdataset_exporter import WebDatasetExporter
            from types import SimpleNamespace
            
            # Convert dict configs to objects for compatibility
            dataset_config = SimpleNamespace(**self.config.dataset_config)
            split_config = SimpleNamespace(**self.config.split_config)
            
            exporter = WebDatasetExporter(
                output_dir=self.config.output_dir,
                dataset_config=dataset_config,
                split_config=split_config,
                num_tiles_base=self.config.num_tiles_base
            )
            exporter.export(self._data)
            
        elif format_type == "files":
            # Files are already saved during run()
            pass
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
    def _tile_image(self, image, num_tiles_base):
        """Split a large image into tiles.
        
        Args:
            image: PIL Image to tile
            num_tiles_base: Number of tiles per side (e.g., 3 for 3x3 grid)
            
        Returns:
            List of (tile_image, tile_x, tile_y) tuples
        """
        from PIL import Image
        
        width, height = image.size
        tile_width = width // num_tiles_base
        tile_height = height // num_tiles_base
        
        tiles = []
        
        for tile_y in range(num_tiles_base):
            for tile_x in range(num_tiles_base):
                # Calculate tile boundaries
                left = tile_x * tile_width
                top = tile_y * tile_height
                right = left + tile_width
                bottom = top + tile_height
                
                # Extract tile
                tile = image.crop((left, top, right, bottom))
                
                # Resize to expected embedding size (224x224)
                tile = tile.resize((224, 224), Image.LANCZOS)
                
                tiles.append((tile, tile_x, tile_y))
        
        return tiles
