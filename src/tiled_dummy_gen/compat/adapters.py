"""Adapter classes to maintain backward compatibility with existing APIs."""

from typing import Tuple, List, Optional
from PIL import Image
from tiled_dummy_gen.config.parser import ClassConfig, ExperimentConfig
from tiled_dummy_gen.core.scene_generator import SceneGenerator
from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline
from tiled_dummy_gen.config.scene_config import (
    SceneConfig,
    ObjectConfig,
    TaskConfig,
    TaskType,
)
from tiled_dummy_gen.compat.converters import ClassConfigConverter


class LegacyGeneratorAdapter:
    """Adapter that maintains the SyntheticDataGenerator API while using the new SceneGenerator internally.

    This allows existing code to continue working without modifications while
    benefiting from the improved architecture underneath.
    """

    def __init__(
        self, image_size=(256, 256), noise_level=10, base_image_size=(256, 256)
    ):
        """Initialize adapter with original SyntheticDataGenerator parameters.

        Args:
            image_size: Image dimensions
            noise_level: Noise level for augmentation
            base_image_size: Base size for scaling calculations
        """
        self.image_size = image_size
        self.noise_level = noise_level
        self.base_image_size = base_image_size

        # Initialize new scene generator
        self._scene_generator = SceneGenerator(noise_level=noise_level)
        self._converter = ClassConfigConverter()

        # Maintain original attributes for compatibility
        self.color_map = self._scene_generator.color_map
        self.thickness_map = {"thin": 5, "medium": 20, "thick": 50}
        self.orientations = ["horizontal", "vertical", "diagonal"]

    def generate_image(
        self, config: ClassConfig, image_size: Optional[Tuple[int, int]] = None
    ):
        """Generate image using legacy ClassConfig - maintains original API.

        Args:
            config: Legacy ClassConfig
            image_size: Optional image size override

        Returns:
            PIL Image
        """
        if image_size is None:
            image_size = self.image_size

        # Convert legacy config to new scene config
        scene_config = self._converter.convert_class_config(config, image_size)

        # Generate using new system
        scene, _ = self._scene_generator.generate_scene(scene_config)

        # Apply augmentations if specified
        augment = config.augment_images
        aug_params = {}
        if augment:
            aug_params = {
                "noise_level": config.augment_images_noise_level,
                "zoom_factor_range": config.augment_images_zoom_factor,
            }

        return self._scene_generator.generate_image(
            scene, augment=augment, **aug_params
        )

    def generate_text(self, config: ClassConfig) -> str:
        """Generate text description using legacy ClassConfig.

        Args:
            config: Legacy ClassConfig

        Returns:
            Text description string
        """
        # Convert to scene config and generate
        scene_config = self._converter.convert_class_config(config)
        scene, text_description = self._scene_generator.generate_scene(scene_config)

        # Apply text augmentation if specified
        if config.augment_texts:
            # Create variations similar to original system
            templates = [
                text_description,
                text_description.replace("A scene with", "This image shows"),
                text_description.replace("A scene with", "The picture contains"),
                f"In this scene, there is {text_description[13:]}",  # Remove "A scene with "
            ]
            import random

            text_description = random.choice(templates)

        return text_description

    def generate_sample(
        self, config: ClassConfig, image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Image.Image, str]:
        """Generate image-text pair using legacy ClassConfig.

        Args:
            config: Legacy ClassConfig
            image_size: Optional image size override

        Returns:
            Tuple of (PIL Image, text description)
        """
        image = self.generate_image(config, image_size)
        text = self.generate_text(config)
        return image, text

    def generate_random_sample(
        self,
        class_configs: List[ClassConfig],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Image.Image, str]:
        """Generate random sample from list of ClassConfigs.

        Args:
            class_configs: List of ClassConfig instances
            image_size: Optional image size override

        Returns:
            Tuple of (PIL Image, text description)
        """
        import random

        config = random.choice(class_configs)
        return self.generate_sample(config, image_size)

    def batch_generate(
        self,
        class_configs: List[ClassConfig],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[Image.Image, str]]:
        """Generate batch of samples from ClassConfigs.

        Args:
            class_configs: List of ClassConfig instances
            image_size: Optional image size override

        Returns:
            List of (PIL Image, text description) tuples
        """
        samples = []
        for config in class_configs:
            for _ in range(config.num_samples):
                sample = self.generate_sample(config, image_size)
                samples.append(sample)
        return samples

    # Legacy method aliases for exact compatibility
    def augment_image(
        self, image: Image.Image, zoom_factor_range: Tuple[float, float]
    ) -> Image.Image:
        """Apply image augmentation - maintains original method signature."""
        return self._scene_generator._apply_augmentations(
            image, zoom_factor_range=zoom_factor_range, noise_level=self.noise_level
        )

    def _apply_noise(self, image_array, noise_level):
        """Legacy noise application method for compatibility."""
        import numpy as np

        noise = np.random.normal(0, noise_level, image_array.shape).astype(np.int16)
        noisy_image = image_array.astype(np.int16) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image


class LegacyPipelineAdapter:
    """Adapter for SyntheticDataPipeline that uses new architecture while maintaining API compatibility."""

    def __init__(self, config: ExperimentConfig):
        """Initialize pipeline adapter with legacy ExperimentConfig.

        Args:
            config: Legacy ExperimentConfig
        """
        self.config = config

        # Store legacy attributes for compatibility
        self.output_dir = config.output_dir
        self.class_configs = config.class_configs
        self.embedder_config = config.embedder_config
        self.dataset_config = config.dataset_config
        self.split_config = config.split_config
        self.num_tiles_base = config.num_tiles_base
        self.image_size = (
            config.image_size[0] * config.num_tiles_base,
            config.image_size[1] * config.num_tiles_base,
        )

        # Initialize new scene generator and converter
        self._scene_generator = SceneGenerator()
        self._converter = ClassConfigConverter()

        # Initialize legacy generator for exact compatibility
        self.generator = LegacyGeneratorAdapter(
            image_size=self.image_size,
            noise_level=self._get_average_noise_level(),
            base_image_size=self.image_size,
        )

        # Initialize embedder if config provided
        if self.embedder_config:
            from tiled_dummy_gen.core.embedder import ImageEmbedder

            self.embedder = ImageEmbedder(config=self.embedder_config)
        else:
            self.embedder = None

        # Initialize data manager
        from tiled_dummy_gen.data.manager import DataManager

        self.data_manager = DataManager()

        # Setup directories
        self.setup_directory()

    def _get_average_noise_level(self):
        """Calculate average noise level from class configs."""
        import numpy as np

        noise_levels = [
            config.augment_images_noise_level
            for config in self.class_configs
            if config.augment_images
        ]
        return np.mean(noise_levels) if noise_levels else 10.0

    def setup_directory(self):
        """Setup output directories - maintains original API."""
        import os
        import logging

        logger = logging.getLogger(__name__)

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        logger.info(f"Images directory: {self.images_dir}")

        if self.embedder:
            self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.visualizations_dir, exist_ok=True)
            logger.info(f"Visualizations directory: {self.visualizations_dir}")

    def generate_data(self):
        """Generate synthetic data using legacy API - delegates to new system internally."""
        # Use the original pipeline's generate_data method for full compatibility
        # This is complex because it involves file saving and data management
        # For now, create a temporary SyntheticDataPipeline instance

        from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

        temp_pipeline = SyntheticDataPipeline(config=self.config)
        temp_pipeline.generate_data()

        # Copy the data manager state
        self.data_manager = temp_pipeline.data_manager

    def embed_images(self):
        """Embed images using legacy API."""
        if not self.embedder:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("No embedder initialized. Skipping embedding generation.")
            return

        from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

        temp_pipeline = SyntheticDataPipeline(config=self.config)
        temp_pipeline.data_manager = self.data_manager  # Use existing data
        temp_pipeline.embed_images()

        # Copy back the updated data manager
        self.data_manager = temp_pipeline.data_manager

    def run(self):
        """Run data generation pipeline without embedding."""
        self.generate_data()
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Data generation pipeline completed.")

    def run_with_embedding(self):
        """Run data generation pipeline with embedding."""
        self.generate_data()
        self.embed_images()
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Data generation and embedding pipeline completed.")

    def save_data(self, output_format: str):
        """Save data using legacy API."""
        from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

        temp_pipeline = SyntheticDataPipeline(config=self.config)
        temp_pipeline.data_manager = self.data_manager
        temp_pipeline.save_data(output_format)

    def create_zero_shot_labels(self):
        """Create zero-shot labels using legacy API."""
        from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

        temp_pipeline = SyntheticDataPipeline(config=self.config)
        temp_pipeline.create_zero_shot_labels()
