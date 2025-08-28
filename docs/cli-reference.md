# CLI Reference

## Commands

### `large-image-synth run`

Run the complete synthetic data generation pipeline with embeddings.

```bash
large-image-synth run CONFIG_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Override output directory from config
- `--embedding / --no-embedding` - Enable/disable embedding generation (default: enabled)
- `--format [hdf5|files|webdataset|both]` - Output format (default: hdf5)
- `-v, --verbose` - Enable verbose output

**Examples:**
```bash
# Scene-based generation (recommended)
large-image-synth run examples/experiments/spatial_stars_example.json

# Legacy bar-based generation  
large-image-synth run experiments/binary.json

# Override output directory
large-image-synth run experiments/spatial_stars_example.json -o /custom/output

# Generate multiple export formats
large-image-synth run experiments/spatial_stars_example.json --format both

# Skip embedding generation
large-image-synth run experiments/spatial_stars_example.json --no-embedding
```

### `large-image-synth generate`

Generate synthetic data without embeddings (faster).

```bash
large-image-synth generate CONFIG_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Override output directory
- `-v, --verbose` - Enable verbose output

**Example:**
```bash
large-image-synth generate experiments/multiclass.json -v
```

### `large-image-synth embed`

Generate embeddings for existing images.

```bash
large-image-synth embed CONFIG_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Override output directory
- `-v, --verbose` - Enable verbose output

**Example:**
```bash
# Generate embeddings after running generate command
large-image-synth embed experiments/binary.json
```

### `large-image-synth validate`

Validate a configuration file without running the pipeline.

```bash
large-image-synth validate CONFIG_PATH
```

**Example:**
```bash
large-image-synth validate experiments/binary.json
# Output:
# ✅ Configuration is valid!
#    - Classes: 2
#    - Output: ../data/synthetic_data_binary_class
#    - Embedder: google/vit-base-patch16-224
```

## Common Workflow

1. **Validate configuration:**
   ```bash
   large-image-synth validate config.json
   ```

2. **Generate data with embeddings:**
   ```bash
   large-image-synth run config.json --verbose
   ```

3. **Or generate in steps:**
   ```bash
   # Step 1: Generate synthetic data
   large-image-synth generate config.json
   
   # Step 2: Generate embeddings
   large-image-synth embed config.json
   ```

## Output Formats

### HDF5 Format (`--format hdf5`)
- Single `.hdf5` file with hierarchical structure
- Suitable for ML frameworks expecting HDF5 datasets
- Includes embeddings, positions, and metadata

### File Structure Format (`--format files`)  
- Organized directory structure with `.pth` files
- Compatible with PyTorch data loaders
- Separate feature and tile information files

### Both Formats (`--format both`)
- Generates both HDF5 and file structure outputs
- Useful for different downstream applications