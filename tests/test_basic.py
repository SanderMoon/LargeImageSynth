"""Basic tests for TiledDummyGen package."""

import pytest
from pathlib import Path


def test_package_import():
    """Test that the package can be imported."""
    import tiled_dummy_gen

    assert tiled_dummy_gen.__version__ == "0.1.0"


def test_core_imports():
    """Test that core components can be imported."""
    from tiled_dummy_gen import (
        SyntheticDataGenerator,
        ImageEmbedder,
        SyntheticDataPipeline,
        ConfigParser,
        ExperimentConfig,
    )

    # Basic class existence checks
    assert SyntheticDataGenerator is not None
    assert ImageEmbedder is not None
    assert SyntheticDataPipeline is not None
    assert ConfigParser is not None
    assert ExperimentConfig is not None


def test_cli_import():
    """Test that CLI module can be imported."""
    from tiled_dummy_gen import cli

    assert cli is not None


def test_config_validation():
    """Test configuration validation with example config."""
    from tiled_dummy_gen.config.parser import ConfigParser

    # Create a minimal valid config
    config_data = {
        "class_template": {
            "image_bar_orientation": "horizontal",
            "image_bar_thickness": "medium",
            "num_samples": 1,
            "augment_images": False,
            "augment_images_noise_level": 0.0,
            "augment_images_zoom_factor": [1.0, 1.0],
            "augment_texts": False,
        },
        "variations": {"image_background_color": ["blue"]},
        "embedder_config": {
            "embedder_name": "google/vit-base-patch16-224",
            "device": "cpu",
        },
        "dataset_config": {"hdf5_filename": "test.hdf5"},
        "split_config": {"split": False, "train": 0.7, "val": 0.2, "test": 0.1},
        "output_dir": "/tmp/test",
        "num_tiles_base": 1,
        "image_size": [224, 224],
    }

    # Test that we can create config objects
    from tiled_dummy_gen.config.parser import ExperimentConfig, ClassConfig

    # This should not raise an exception
    class_configs = []
    for bg_color in config_data["variations"]["image_background_color"]:
        class_config = ClassConfig(
            name=f"test_{bg_color}",
            num_samples=1,
            image_background_color=bg_color,
            image_bar_orientation="horizontal",
            image_bar_thickness="medium",
            augment_images=False,
            augment_images_noise_level=0.0,
            augment_images_zoom_factor=(1.0, 1.0),
            augment_texts=False,
        )
        class_configs.append(class_config)

    assert len(class_configs) == 1
    assert class_configs[0].image_background_color == "blue"
