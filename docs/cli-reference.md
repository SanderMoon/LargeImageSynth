# CLI Reference

## Commands

### `tiled-dummy-gen run`

Run the complete synthetic data generation pipeline with embeddings.

```bash
tiled-dummy-gen run CONFIG_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Override output directory from config
- `--embedding / --no-embedding` - Enable/disable embedding generation (default: enabled)
- `--format [hdf5|files|both]` - Output format (default: hdf5)
- `-v, --verbose` - Enable verbose output

**Examples:**
```bash
# Basic usage
tiled-dummy-gen run experiments/binary.json

# Override output directory
tiled-dummy-gen run experiments/binary.json -o /custom/output

# Generate both HDF5 and file structure formats
tiled-dummy-gen run experiments/binary.json --format both

# Skip embedding generation
tiled-dummy-gen run experiments/binary.json --no-embedding
```

### `tiled-dummy-gen generate`

Generate synthetic data without embeddings (faster).

```bash
tiled-dummy-gen generate CONFIG_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Override output directory
- `-v, --verbose` - Enable verbose output

**Example:**
```bash
tiled-dummy-gen generate experiments/multiclass.json -v
```

### `tiled-dummy-gen embed`

Generate embeddings for existing images.

```bash
tiled-dummy-gen embed CONFIG_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Override output directory
- `-v, --verbose` - Enable verbose output

**Example:**
```bash
# Generate embeddings after running generate command
tiled-dummy-gen embed experiments/binary.json
```

### `tiled-dummy-gen validate`

Validate a configuration file without running the pipeline.

```bash
tiled-dummy-gen validate CONFIG_PATH
```

**Example:**
```bash
tiled-dummy-gen validate experiments/binary.json
# Output:
# ✅ Configuration is valid!
#    - Classes: 2
#    - Output: ../data/synthetic_data_binary_class
#    - Embedder: google/vit-base-patch16-224
```

## Common Workflow

1. **Validate configuration:**
   ```bash
   tiled-dummy-gen validate config.json
   ```

2. **Generate data with embeddings:**
   ```bash
   tiled-dummy-gen run config.json --verbose
   ```

3. **Or generate in steps:**
   ```bash
   # Step 1: Generate synthetic data
   tiled-dummy-gen generate config.json
   
   # Step 2: Generate embeddings
   tiled-dummy-gen embed config.json
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