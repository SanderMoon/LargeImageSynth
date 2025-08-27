import os
import csv
import random
from PIL import Image
from tiled_dummy_gen.core.generator import SyntheticDataGenerator
from tiled_dummy_gen.core.embedder import ImageEmbedder
import pandas as pd
import numpy as np
import torch
import h5py
from typing import List, Tuple, Optional
from tiled_dummy_gen.config.parser import ExperimentConfig
from sklearn.model_selection import train_test_split
import json
import logging

from tiled_dummy_gen.data.manager import DataManager
from tiled_dummy_gen.export.hdf5_exporter import HDF5Exporter
from tiled_dummy_gen.export.file_structure_exporter import FileStructureExporter
from tiled_dummy_gen.export.webdataset_exporter import WebDatasetExporter

logger = logging.getLogger(__name__)


class SyntheticDataPipeline:
    """
    A pipeline to generate synthetic image-text data, embed images, and save them to a specified directory.
    It orchestrates data generation, embedding, and export using dedicated managers and exporters.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the pipeline with generator, dataset, and embedding configurations.

        Parameters:
        - config: Instance of ExperimentConfig containing all necessary configurations.
        """
        self.config = config
        self.output_dir = self.config.output_dir
        self.class_configs = self.config.class_configs
        self.embedder_config = self.config.embedder_config
        self.dataset_config = self.config.dataset_config
        self.split_config = self.config.split_config
        self.num_tiles_base = self.config.num_tiles_base
        self.image_size = (
            self.config.image_size[0] * self.num_tiles_base,
            self.config.image_size[1] * self.num_tiles_base,
        )

        self.data_manager = DataManager()

        # Initialize the SyntheticDataGenerator with generator parameters
        self.generator = SyntheticDataGenerator(
            image_size=self.image_size,  # Dynamic image size from config
            noise_level=self._get_average_noise_level(),
            base_image_size=self.image_size,  # Set base_image_size to match image_size for scaling
        )

        # Initialize the embedder if embedder configuration is provided
        if self.embedder_config:
            self.embedder = ImageEmbedder(config=self.embedder_config)
        else:
            self.embedder = None

        # Setup directory structure
        self.setup_directory()

    def _get_average_noise_level(self):
        """
        Calculates the average noise level from all class configurations that have image augmentation enabled.

        Returns:
        - Float representing the average noise level.
        """
        noise_levels = [
            config.augment_images_noise_level
            for config in self.class_configs
            if config.augment_images
        ]
        return np.mean(noise_levels) if noise_levels else self.generator.noise_level

    def setup_directory(self):
        """
        Sets up the output directory and subdirectories for images and visualizations.
        """
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
        """
        Generates synthetic samples and populates the DataManager.
        Images are saved to disk.
        """
        embedder_size = (
            self.embedder.expected_size if self.embedder else self.generator.image_size
        )
        tile_base = self.num_tiles_base
        tile_size = embedder_size  # Each tile should match the embedder's expected size

        for class_config in self.class_configs:
            subject_id = class_config.name
            for sample_idx in range(1, class_config.num_samples + 1):
                sample_id = f"Sample{sample_idx}"

                if tile_base == 1:
                    image, text = self.generator.generate_sample(class_config)
                    filename = self._generate_filename(subject_id, sample_id)
                    image_path = os.path.join(self.images_dir, filename)
                    image.save(image_path)
                    self.data_manager.add_annotation(
                        {
                            "filename": filename,
                            "description": text,
                            "label": f"{subject_id}_{sample_id}",
                            "tile_x": None,
                            "tile_y": None,
                        }
                    )
                else:
                    large_image_size = (
                        tile_size[0] * tile_base,
                        tile_size[1] * tile_base,
                    )
                    image_size_tuple = (large_image_size[0], large_image_size[1])
                    large_image, text = self.generator.generate_sample(
                        class_config, image_size=image_size_tuple
                    )
                    large_filename = self._generate_filename(
                        subject_id, sample_id, is_large=True
                    )
                    large_image_path = os.path.join(self.images_dir, large_filename)
                    large_image.save(large_image_path)
                    logger.info(f"Saved large image: {large_image_path}")

                    tiles = self._split_into_tiles(large_image, tile_base, tile_size)

                    for x in range(tile_base):
                        for y in range(tile_base):
                            tile_image = tiles[x][y]
                            tile_sample_id = f"{sample_id}_Tile{x+1}_{y+1}"
                            tile_filename = self._generate_filename(
                                subject_id, tile_sample_id, is_large=False
                            )
                            tile_image_path = os.path.join(
                                self.images_dir, tile_filename
                            )
                            tile_image.save(tile_image_path)
                            logger.info(f"Saved tile image: {tile_image_path}")

                            self.data_manager.add_annotation(
                                {
                                    "filename": tile_filename,
                                    "description": text,
                                    "label": f"{subject_id}_{sample_id}",
                                    "tile_x": x + 1,
                                    "tile_y": y + 1,
                                }
                            )
        logger.info("Data generation completed.")

    def _split_into_tiles(
        self, large_image: Image.Image, tile_base: int, tile_size: Tuple[int, int]
    ) -> List[List[Image.Image]]:
        """
        Splits a large image into smaller tiles based on the tile base.
        """
        tiles = []
        for x in range(tile_base):
            row = []
            for y in range(tile_base):
                left = y * tile_size[0]
                upper = x * tile_size[1]
                right = left + tile_size[0]
                lower = upper + tile_size[1]
                tile = large_image.crop((left, upper, right, lower))
                row.append(tile)
            tiles.append(row)
        return tiles

    def _generate_filename(
        self, subject_id: str, sample_id: str, is_large: bool = False
    ) -> str:
        """
        Generates a unique filename for each image or tile.
        """
        if is_large:
            base = f"{subject_id}_{sample_id}_large_{random.randint(1000,9999)}.png"
        else:
            base = f"{subject_id}_{sample_id}.png"

        # Ensure filename uniqueness
        while os.path.exists(os.path.join(self.images_dir, base)):
            if is_large:
                base = f"{subject_id}_{sample_id}_large_{random.randint(1000,9999)}.png"
            else:
                base = f"{subject_id}_{sample_id}_{random.randint(1000,9999)}.png"

        return base

    def embed_images(self):
        """
        Embeds all images using the ImageEmbedder and stores embeddings in DataManager.
        Also saves a visualization for the first image or tile.
        """
        if not self.embedder:
            logger.warning("No embedder initialized. Skipping embedding generation.")
            return

        annotations_df = self.data_manager.annotations_df
        if annotations_df.empty:
            logger.warning("No annotations found in DataManager. Cannot embed images.")
            return

        visualization_saved = False

        for _, annotation in annotations_df.iterrows():
            image_path = os.path.join(self.images_dir, annotation["filename"])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error opening image {image_path}: {e}")
                continue

            embedding = self.embedder.embed_image(image)
            self.data_manager.add_embedding(annotation["filename"], embedding.tolist())

            if not visualization_saved:
                visualization_path = os.path.join(
                    self.visualizations_dir, "preprocessing_visualization.png"
                )
                self.embedder.save_preprocessed_visualization(image, visualization_path)
                visualization_saved = True
        logger.info("Image embedding completed.")

    def run(self):
        """
        Runs the data generation pipeline without embedding.
        """
        self.generate_data()
        logger.info("Data generation pipeline completed.")

    def run_with_embedding(self):
        """
        Runs the data generation pipeline including embedding images.
        """
        self.generate_data()
        self.embed_images()
        logger.info("Data generation and embedding pipeline completed.")

    def save_data(self, output_format: str):
        """
        Saves the generated data using the specified output format.
        """
        data_to_export = self.data_manager.get_merged_data()

        if data_to_export.empty:
            logger.warning("No data to export.")
            return

        if output_format == "hdf5":
            exporter = HDF5Exporter(self.output_dir, self.dataset_config, self.split_config, self.num_tiles_base)
            exporter.export(data_to_export)
        elif output_format == "webdataset":
            exporter = WebDatasetExporter(self.output_dir, self.dataset_config, self.split_config, self.num_tiles_base)
            exporter.export(data_to_export)
        elif output_format == "both":
            hdf5_exporter = HDF5Exporter(self.output_dir, self.dataset_config, self.split_config, self.num_tiles_base)
            hdf5_exporter.export(data_to_export)
            files_exporter = FileStructureExporter(self.output_dir, self.dataset_config, self.split_config, self.num_tiles_base)
            files_exporter.export(data_to_export)
        else:
            logger.error(f"Unsupported output format: {output_format}")

    def create_zero_shot_labels(self):
        """
        Creates zero-shot learning label mappings in JSON format.
        """
        class_id_map = {}
        textual_to_id = {}

        for idx, class_config in enumerate(self.class_configs, start=1):
            for i in range(1, class_config.num_samples + 1):
                sample_id = f"{class_config.image_background_color}_Sample{i}"
                class_id_map[sample_id] = idx
                text_label = f"This image depicts a {class_config.image_background_color} background with a {class_config.image_bar_orientation} bar orientation and a {class_config.image_bar_thickness} bar thickness."
                textual_to_id[idx] = text_label

        textual_to_id_path = os.path.join(
            self.output_dir, "zero_shot_textual_to_id.json"
        )
        sample_to_id_path = os.path.join(
            self.output_dir, "zero_shot_sample_to_id.json"
        )

        with open(textual_to_id_path, "w") as f:
            json.dump(textual_to_id, f, indent=4)

        with open(sample_to_id_path, "w") as f:
            json.dump(class_id_map, f, indent=4)

        logger.info(f"Saved zero-shot textual-to-ID labels to '{textual_to_id_path}'.")
        logger.info(f"Saved zero-shot sample-to-ID labels to '{sample_to_id_path}'.")
