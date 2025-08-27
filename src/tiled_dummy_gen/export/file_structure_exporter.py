import os
import pandas as pd
import numpy as np
import torch
import json
import logging
from typing import Any, List

from .base_exporter import Exporter

logger = logging.getLogger(__name__)

class FileStructureExporter(Exporter):
    """
    Exports data to a structured file format compatible with PyTorch data loaders.
    """
    def __init__(self, output_dir: str, dataset_config: Any, split_config: Any, num_tiles_base: int):
        super().__init__(output_dir, dataset_config, split_config, num_tiles_base)
        logger.info("FileStructureExporter initialized.")

    def export(self, data_df: pd.DataFrame):
        """
        Exports the merged DataFrame to a structured file format.
        """
        if data_df.empty:
            logger.warning("DataFrame is empty. Nothing to save to file structure.")
            return

        logger.info(f"Saving data to file structure in: {self.output_dir}")

        def parse_label(label):
            parts = label.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Label '{label}' does not match the expected format 'PXXXXXX_SampleY'"
                )
            subject_id = parts[0]
            sample_id = "_".join(parts[1:])
            return subject_id, sample_id

        data_df["subject_id", "sample_id"] = data_df["label"].apply(lambda x: pd.Series(parse_label(x)))

        text_annotations = {}
        grouped_descriptions = data_df.groupby(["subject_id", "sample_id"])["description"].first().reset_index()
        for idx, row in grouped_descriptions.iterrows():
            subject_id = row["subject_id"]
            sample_id = row["sample_id"]
            description = row["description"]
            if subject_id not in text_annotations:
                text_annotations[subject_id] = {}
            text_annotations[subject_id][sample_id] = description

        data_df["sample_id_col"] = data_df["sample_id"]
        sample_ids = data_df["sample_id_col"].unique()
        sample_index_map = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        data_df["sample_index"] = data_df["sample_id_col"].map(sample_index_map)

        data_df["slide_index"] = 0
        data_df["cross_section_index"] = 0
        data_df["tile_index"] = data_df.groupby("sample_id_col").cumcount()

        embedding_cols = [col for col in data_df.columns if col.startswith("embedding_")]

        if "tile_x" not in data_df.columns or "tile_y" not in data_df.columns:
            logger.warning("Tile positions 'tile_x' and 'tile_y' are missing from data. Cannot save full file structure.")
            return

        batch_size = 100
        num_batches = (len(sample_ids) + batch_size - 1) // batch_size
        sample_ids_txt_list = []

        for batch_num in range(num_batches):
            batch_sample_ids = sample_ids[batch_num * batch_size: (batch_num + 1) * batch_size]
            batch_dir = os.path.join(self.output_dir, f"data_{batch_num}")
            os.makedirs(batch_dir, exist_ok=True)
            extracted_features_dir = os.path.join(batch_dir, "extracted_features")
            os.makedirs(extracted_features_dir, exist_ok=True)

            feature_info_path = os.path.join(batch_dir, "feature_information.txt")
            tile_info_path = os.path.join(batch_dir, "tile_information.txt")

            with open(feature_info_path, "w") as feature_info_file, \
                 open(tile_info_path, "w") as tile_info_file:
                for sample_id in batch_sample_ids:
                    sample_ids_txt_list.append(sample_id)
                    sample_data = data_df[data_df["sample_id_col"] == sample_id]
                    sample_index = sample_data["sample_index"].iloc[0]

                    wsi_names = specimen_data["filename"].unique().tolist()
                    wsi_names = [f"{wsi_name}" for wsi_name in wsi_names]
                    feature_info_file.write(json.dumps(wsi_names) + "\n")

                    sample_info = {
                        "sample_index": int(sample_index),
                        "subject": sample_data["subject_id"].iloc[0],
                        "sample": sample_id,
                        "size": float(sample_data.shape[0]),
                    }
                    feature_info_file.write(json.dumps(sample_info) + "\n")

                    pth_filename = f"{sample_index}.pth"
                    feature_info_file.write(f'"{pth_filename}"\n')

                    pth_file_path = os.path.join(extracted_features_dir, pth_filename)
                    pth_data = {0: {}}

                    for _, row in sample_data.iterrows():
                        key_tuple = (
                            int(row["sample_index"]),
                            int(row["slide_index"]),
                            int(row["cross_section_index"]),
                            int(row["tile_index"]),
                        )
                        feature_array = list(row[embedding_cols].values.astype(np.float32))
                        feature_tensor = [feature_array]
                        position_tensor = [(1, int(row["tile_x"])), int(row["tile_y"])]

                        pth_data[0][key_tuple] = {
                            "feature": feature_tensor,
                            "position": position_tensor,
                        }
                    torch.save(pth_data, pth_file_path)

                    tile_info = {
                        "sample_index": int(sample_index),
                        "tile_indices": sample_data["tile_index"].tolist(),
                    }
                    tile_info_file.write(json.dumps(tile_info) + "\n")

            logger.info(f"Saved data for batch {batch_num} to '{batch_dir}'.")

        text_annotations_path = os.path.join(self.output_dir, "text_annotations.json")
        with open(text_annotations_path, "w") as json_file:
            json.dump(text_annotations, json_file, indent=4)
        logger.info(f"Saved text annotations to '{text_annotations_path}'.")

        self._save_split_files(sample_ids_txt_list)
        logger.info("Data successfully saved to file structure.")

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

        train_ids, temp_ids = train_test_split(sample_ids_list, test_size=(1 - train_ratio), random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

        splits = {"train": train_ids, "val": val_ids, "test": test_ids}

        for split, ids in splits.items():
            sample_ids_joined = ",".join(ids)
            sample_ids_path = os.path.join(self.output_dir, f"sample_ids_{split}.txt")
            with open(sample_ids_path, "w") as f:
                f.write(sample_ids_joined)
            logger.info(f"Saved {split} sample IDs to '{sample_ids_path}'.")
