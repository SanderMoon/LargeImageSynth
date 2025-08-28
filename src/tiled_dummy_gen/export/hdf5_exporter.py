import os
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import logging
from typing import Any, List
import random

from tiled_dummy_gen.export.base_exporter import Exporter

logger = logging.getLogger(__name__)


class HDF5Exporter(Exporter):
    """
    Exports data to an HDF5 file following a predefined hierarchy.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_config: Any,
        split_config: Any,
        num_tiles_base: int,
    ):
        super().__init__(output_dir, dataset_config, split_config, num_tiles_base)
        self.hdf5_filename = self.dataset_config.hdf5_filename
        self.hdf5_path = os.path.join(self.output_dir, self.hdf5_filename)
        logger.info(f"HDF5Exporter initialized. Output path: {self.hdf5_path}")

    def export(self, data_df: pd.DataFrame):
        """
        Exports the merged DataFrame to an HDF5 file.
        """
        if data_df.empty:
            logger.warning("DataFrame is empty. Nothing to save to HDF5.")
            return

        logger.info(f"Saving data to HDF5 format: {self.hdf5_path}")

        # Parse subject ID and sample ID from label
        def parse_label(label):
            parts = label.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Label '{label}' does not match the expected format 'PXXXXXX_SampleY'"
                )
            subject_id = parts[0]
            sample_id = "_".join(parts[1:])  # In case sample_id contains underscores
            return subject_id, sample_id

        sample_ids_txt_list = []  # For txt file (strings)
        sample_paths_list = []  # For HDF5 paths

        with h5py.File(self.hdf5_path, "w") as hdf5_file:
            sample_index_grp = hdf5_file.create_group("sample-index")
            subjects_grp = hdf5_file.create_group("subjects")

            grouped = data_df.groupby("label")

            for label, group in grouped:
                try:
                    subject_id, sample_id = parse_label(label)
                except ValueError as ve:
                    logger.warning(f"Skipping label '{label}': {ve}")
                    continue

                sample_grp_path = f"subjects/{subject_id}/{sample_id}"

                subject_grp = subjects_grp.require_group(subject_id)
                sample_grp = subject_grp.require_group(sample_id)
                hipt_features_grp = sample_grp.require_group("hipt_features_comp_1")

                embedding_cols = [
                    col for col in data_df.columns if col.startswith("embedding_")
                ]
                embeddings = group[embedding_cols].values.astype(np.float64)
                n_samples = embeddings.shape[0]
                embedding_dim = embeddings.shape[1]

                embeddings_reshaped = embeddings.reshape(n_samples, 1, embedding_dim)

                if self.num_tiles_base == 1:
                    positions_reshaped = np.array(
                        [[1, 1, 1] for _ in range(n_samples)], dtype="int64"
                    )
                else:
                    positions = group[["tile_x", "tile_y"]].values
                    positions_reshaped = np.hstack(
                        (np.ones((n_samples, 1), dtype="int64"), positions)
                    )

                positions_reshaped = positions_reshaped.reshape(n_samples, 1, 3)

                tile_keys = np.array(
                    [f"tile_{random.randint(1000, 9999)}" for _ in range(n_samples)],
                    dtype="S13",
                )

                hipt_features_grp.create_dataset(
                    "features", data=embeddings_reshaped, dtype="float64"
                )
                hipt_features_grp.create_dataset(
                    "positions", data=positions_reshaped, dtype="int64"
                )
                hipt_features_grp.create_dataset(
                    "tile_keys", data=tile_keys, dtype="S13"
                )

                descriptions = group["description"].tolist()
                concatenated_text = descriptions[0]
                text_key = self.dataset_config.text_key
                dt = h5py.string_dtype(encoding="utf-8")
                sample_grp.create_dataset(text_key, data=concatenated_text, dtype=dt)

                sample_ids_txt_list.append(sample_id)
                sample_paths_list.append(sample_grp_path)

            # After processing all samples, create datasets under 'sample-index'
            if sample_ids_txt_list:
                sample_ids_bytes = [cid.encode("utf-8") for cid in sample_ids_txt_list]
                sample_paths_bytes = [
                    cpath.encode("utf-8") for cpath in sample_paths_list
                ]

                max_sample_id_length = (
                    max(len(cid) for cid in sample_ids_bytes) if sample_ids_bytes else 0
                )
                max_sample_path_length = (
                    max(len(cpath) for cpath in sample_paths_bytes)
                    if sample_paths_bytes
                    else 0
                )

                sample_ids_np = np.array(
                    sample_ids_bytes, dtype=f"S{max_sample_id_length}"
                )
                sample_paths_np = np.array(
                    sample_paths_bytes, dtype=f"S{max_sample_path_length}"
                )

                sample_index_grp.create_dataset("sample_ids", data=sample_ids_np)
                sample_index_grp.create_dataset("sample_paths", data=sample_paths_np)

                logger.info(
                    f"Created 'sample-index' with {len(sample_ids_txt_list)} samples."
                )
            else:
                logger.warning("No samples were added to 'sample-index'.")

        if self.split_config.split:
            self._save_split_files(sample_ids_txt_list)
        logger.info(f"Data successfully saved to HDF5: {self.hdf5_path}")

    def _save_split_files(self, sample_ids_list: List[str]):
        """
        Saves train, validation, and test split files.
        """
        if not sample_ids_list:
            logger.warning("No sample IDs to save to text files for splits.")
            return

        train_ratio = self.split_config.train
        val_ratio = self.split_config.val
        test_ratio = self.split_config.test

        train_ids, temp_ids = train_test_split(
            sample_ids_list, test_size=(1 - train_ratio), random_state=42
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42
        )

        splits = {"train": train_ids, "val": val_ids, "test": test_ids}

        for split, ids in splits.items():
            sample_ids_joined = ",".join(ids)
            sample_ids_path = os.path.join(self.output_dir, f"sample_ids_{split}.txt")
            with open(sample_ids_path, "w") as f:
                f.write(sample_ids_joined)
            logger.info(f"Saved {split} sample IDs to '{sample_ids_path}'.")
