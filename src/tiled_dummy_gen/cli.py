"""Command-line interface for TiledDummyGen.

This module provides command-line tools for running synthetic data generation
pipelines, configuring experiments, and managing datasets.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging

import click

from tiled_dummy_gen.config.parser import ExperimentConfigLoader
from tiled_dummy_gen.core.pipeline import SyntheticDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _initialize_pipeline(config_path: Path, output_dir: Optional[Path], verbose: bool) -> SyntheticDataPipeline:
    """Helper to load config and initialize the pipeline."""
    if verbose:
        # Set logger to debug level for all loggers.
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Verbose mode enabled.")
    
    logger.info(f"Loading configuration from: {config_path}")
    loader = ExperimentConfigLoader(str(config_path))
    experiment_config = loader.get_experiment_config()
    
    if output_dir:
        experiment_config.output_dir = str(output_dir)
        logger.info(f"Output directory overridden to: {output_dir}")
        
    pipeline = SyntheticDataPipeline(config=experiment_config)
    return pipeline


@click.group()
@click.version_option()
def main():
    """TiledDummyGen: Generate synthetic tiled datasets with embeddings."""
    pass


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o", 
    type=click.Path(path_type=Path),
    help="Override output directory from config"
)
@click.option(
    "--embedding/--no-embedding",
    default=True,
    help="Whether to generate embeddings (default: True)"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["hdf5", "files", "webdataset", "both"]),
    default="hdf5",
    help="Output format (default: hdf5)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def run(
    config_path: Path,
    output_dir: Optional[Path],
    embedding: bool,
    output_format: str,
    verbose: bool
):
    """Run the complete synthetic data generation pipeline.
    
    CONFIG_PATH: Path to the JSON configuration file.
    """
    try:
        pipeline = _initialize_pipeline(config_path, output_dir, verbose)
        
        if embedding:
            logger.info("Running pipeline with embedding generation...")
            pipeline.run_with_embedding()
        else:
            logger.info("Running pipeline without embedding generation...")
            pipeline.run()
        
        if output_format in ("hdf5", "both"):
            logger.info(f"Saving data in 'hdf5' format(s)...")
            pipeline.save_data("hdf5")
            
        if output_format in ("files", "both"):
            logger.info(f"Saving data in 'files' format(s)...")
            pipeline.save_data("files")

        if output_format in ("webdataset", "both"):
            logger.info(f"Saving data in 'webdataset' format(s)...")
            pipeline.save_data("webdataset")
        
        click.echo("✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=verbose)
        sys.exit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path), 
    help="Override output directory from config"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def generate(config_path: Path, output_dir: Optional[Path], verbose: bool):
    """Generate synthetic data without embeddings.
    
    CONFIG_PATH: Path to the JSON configuration file.
    """
    try:
        pipeline = _initialize_pipeline(config_path, output_dir, verbose)
        logger.info("Generating synthetic data...")
        pipeline.run()
        click.echo("✅ Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}", exc_info=verbose)
        sys.exit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Override output directory from config"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def embed(config_path: Path, output_dir: Optional[Path], verbose: bool):
    """Generate embeddings for existing images.
    
    CONFIG_PATH: Path to the JSON configuration file.
    """
    try:
        pipeline = _initialize_pipeline(config_path, output_dir, verbose)
        logger.info("Generating embeddings...")
        pipeline.embed_images()
        click.echo("✅ Embedding generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=verbose)
        sys.exit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def validate(config_path: Path, verbose: bool):
    """Validate a configuration file.
    
    CONFIG_PATH: Path to the JSON configuration file to validate.
    """
    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug(f"Verbose mode enabled.")

        logger.info(f"Validating configuration: {config_path}")
        loader = ExperimentConfigLoader(str(config_path))
        experiment_config = loader.get_experiment_config()
        
        click.echo("✅ Configuration is valid!")
        click.echo(f"   - Classes: {len(experiment_config.class_configs)}")
        click.echo(f"   - Output: {experiment_config.output_dir}")
        click.echo(f"   - Embedder: {experiment_config.embedder_config.embedder_name}")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()