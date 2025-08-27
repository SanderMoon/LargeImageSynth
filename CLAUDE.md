# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install with CLI support only
pip install -e ".[cli]"
```

### Testing and Quality Assurance
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/tiled_dummy_gen

# Format code
black src/ tests/

# Lint code  
ruff src/ tests/

# Type checking
mypy src/
```

### CLI Usage
```bash
# Run complete pipeline with embeddings
tiled-dummy-gen run examples/experiments/binary.json --verbose

# Generate synthetic data only (no embeddings)
tiled-dummy-gen generate examples/experiments/binary.json --no-embedding

# Validate configuration file
tiled-dummy-gen validate examples/experiments/binary.json

# Embed existing data
tiled-dummy-gen embed examples/experiments/binary.json
```

## Architecture Overview

### Core Pipeline Flow
1. **Configuration Loading** (`config/parser.py`) - Parses JSON experiment configurations and creates typed config objects
2. **Data Generation** (`core/generator.py`) - Creates synthetic images with configurable properties (bars, colors, orientations)  
3. **Image Embedding** (`core/embedder.py`) - Embeds images using Vision Transformer models (ViT)
4. **Data Management** (`data/manager.py`) - Manages annotations and embeddings in memory
5. **Export** (`export/`) - Exports to multiple formats (HDF5, WebDataset, file structure)

The main orchestrator is `SyntheticDataPipeline` in `core/pipeline.py` which coordinates all components.

### Key Components

**SyntheticDataGenerator**: Creates synthetic images with bars of different orientations (horizontal, vertical, diagonal), thicknesses (thin, medium, thick), and background colors. Supports image augmentation with noise and zoom.

**ImageEmbedder**: Uses pre-trained Vision Transformer models to create embeddings. Configurable preprocessing pipeline with resize, normalization, and custom transforms.

**Export System**: Modular exporters supporting:
- HDF5 format with train/val/test splits
- WebDataset format for large-scale training
- File structure export for traditional ML workflows

### Configuration System

Experiments are defined in JSON files with these key sections:
- `class_template`: Base configuration for synthetic data generation
- `variations`: Creates class variations (e.g., different background colors)
- `embedder_config`: Vision model and preprocessing settings
- `dataset_config`: Output format configuration
- `split_config`: Train/validation/test split ratios

### Tiled Image Support

The system supports generating multi-tile images where:
- `num_tiles_base` controls the grid size (e.g., 3x3 tiles)
- Large images are split into smaller tiles for embedding
- Each tile maintains spatial relationship metadata
- Both individual tiles and large composite images are saved

## Common Development Patterns

### Adding New Export Formats
1. Create new exporter class inheriting from `BaseExporter` in `export/`
2. Implement `export()` method following the interface
3. Register in `SyntheticDataPipeline.save_data()` method

### Adding New Embedding Models  
1. Extend `ImageEmbedder` configuration in `config/parser.py`
2. Add model loading logic in `embedder.py`
3. Update preprocessing pipeline as needed

### Configuration Validation
All configurations use dataclasses with validation in `config/parser.py`. Split ratios must sum to 1.0, and required fields are enforced at runtime.