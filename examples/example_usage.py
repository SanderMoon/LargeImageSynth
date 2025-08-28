#!/usr/bin/env python3
"""Example usage of TiledDummyGen library.

This script demonstrates how to use the TiledDummyGen library to generate
synthetic datasets with different configurations.
"""

import os
from pathlib import Path

from tiled_dummy_gen import ConfigParser, SyntheticDataPipeline


def run_binary_example():
    """Run binary classification example."""
    print("🔄 Running binary classification example...")

    # Path to configuration file
    config_path = Path(__file__).parent / "experiments" / "binary.json"

    # Parse configuration
    parser = ConfigParser(str(config_path))
    experiment_config = parser.get_experiment_config()

    # Initialize pipeline
    pipeline = SyntheticDataPipeline(config=experiment_config)

    # Run pipeline with embeddings
    pipeline.run_with_embedding()

    # Save to HDF5
    hdf5_path = os.path.join(
        experiment_config.output_dir, experiment_config.dataset_config.hdf5_filename
    )
    pipeline.save_to_hdf5(hdf5_path)

    print(
        f"✅ Binary example completed! Output saved to: {experiment_config.output_dir}"
    )


def run_multiclass_example():
    """Run multiclass example."""
    print("🔄 Running multiclass example...")

    # Path to configuration file
    config_path = Path(__file__).parent / "experiments" / "multiclass.json"

    # Parse configuration
    parser = ConfigParser(str(config_path))
    experiment_config = parser.get_experiment_config()

    # Initialize pipeline
    pipeline = SyntheticDataPipeline(config=experiment_config)

    # Run pipeline with embeddings
    pipeline.run_with_embedding()

    # Save to HDF5
    hdf5_path = os.path.join(
        experiment_config.output_dir, experiment_config.dataset_config.hdf5_filename
    )
    pipeline.save_to_hdf5(hdf5_path)

    print(
        f"✅ Multiclass example completed! Output saved to: {experiment_config.output_dir}"
    )


def visualize_embeddings(
    annotations_path: str, embeddings_path: str, num_samples: int = 32
):
    """Visualize embeddings using PCA.

    Args:
        annotations_path: Path to annotations CSV file
        embeddings_path: Path to embeddings CSV file
        num_samples: Number of samples to visualize
    """
    try:
        import pandas as pd
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        print(f"📊 Visualizing embeddings from {embeddings_path}...")

        # Load data
        annotations = pd.read_csv(annotations_path)
        embeddings = pd.read_csv(embeddings_path)

        # Merge on filename
        data = pd.merge(annotations, embeddings, on="filename")

        # Sample data if needed
        if len(data) > num_samples:
            data = data.sample(n=num_samples, random_state=42)

        # Extract embedding vectors
        embedding_cols = [
            col for col in embeddings.columns if col.startswith("embedding_")
        ]
        X = data[embedding_cols].values

        # Reduce dimensions using PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        # Encode labels
        label_categories = data["label"].astype("category")
        label_codes = label_categories.cat.codes
        unique_labels = label_categories.cat.categories.tolist()

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=label_codes,
            cmap="viridis",
            edgecolor="k",
            alpha=0.7,
            s=50,
        )

        # Create legend
        handles, _ = scatter.legend_elements(prop="colors")
        plt.legend(
            handles,
            unique_labels,
            title="Classes",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.title("PCA Visualization of Image Embeddings")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig("embedding_visualization.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("📈 Visualization saved as embedding_visualization.png")

    except ImportError as e:
        print(f"⚠️  Visualization requires additional packages: {e}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def main():
    """Main function to run examples."""
    print("🚀 TiledDummyGen Examples")
    print("=" * 40)

    # Run examples
    try:
        run_binary_example()
        print()
        run_multiclass_example()

        # Try visualization if possible
        print()
        print("📊 Attempting to visualize results...")

        # Use the binary example output for visualization
        binary_config_path = Path(__file__).parent / "experiments" / "binary.json"
        parser = ConfigParser(str(binary_config_path))
        config = parser.get_experiment_config()

        annotations_path = os.path.join(config.output_dir, "annotations.csv")
        embeddings_path = os.path.join(
            config.output_dir, "synthetic_dataset_embeddings.csv"
        )

        if os.path.exists(annotations_path) and os.path.exists(embeddings_path):
            visualize_embeddings(annotations_path, embeddings_path)
        else:
            print("⚠️  Annotation or embedding files not found for visualization")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
