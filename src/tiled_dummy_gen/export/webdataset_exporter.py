import os
import pandas as pd
import torch
import json
import logging
import webdataset as wds
from typing import Any

from tiled_dummy_gen.export.base_exporter import Exporter

logger = logging.getLogger(__name__)


class WebDatasetExporter(Exporter):
    """
    Exports data to the WebDataset format (TAR archives).
    """

    def __init__(
        self,
        output_dir: str,
        dataset_config: Any,
        split_config: Any,
        num_tiles_base: int,
    ):
        super().__init__(output_dir, dataset_config, split_config, num_tiles_base)
        self.output_filename = self.dataset_config.hdf5_filename.replace(
            ".hdf5", ".tar"
        )
        self.output_path = os.path.join(self.output_dir, self.output_filename)
        logger.info(f"WebDatasetExporter initialized. Output path: {self.output_path}")

    def export(self, data_df: pd.DataFrame):
        """
        Exports the merged DataFrame to a WebDataset TAR archive.
        """
        if data_df.empty:
            logger.warning("DataFrame is empty. Nothing to save to WebDataset.")
            return

        logger.info(f"Saving data to WebDataset format: {self.output_path}")

        with wds.TarWriter(self.output_path) as sink:
            for index, row in data_df.iterrows():
                sample_key = os.path.splitext(row["filename"])[0]

                # Get embedding columns
                embedding_cols = [
                    col for col in data_df.columns if col.startswith("embedding_")
                ]
                feature_tensor = torch.tensor(
                    row[embedding_cols].values.astype(float).tolist()
                )

                metadata = {
                    "label": row["label"],
                    "description": row["description"],
                    "position": (row["tile_x"], row["tile_y"]),
                }

                sample = {
                    "__key__": sample_key,
                    "pth": feature_tensor,
                    "json": json.dumps(metadata, indent=4),
                }

                sink.write(sample)

        logger.info(f"Data successfully saved to WebDataset: {self.output_path}")
