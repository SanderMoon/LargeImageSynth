from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Exporter(ABC):
    """
    Abstract base class for data exporters.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_config: Any,
        split_config: Any,
        num_tiles_base: int,
    ):
        self.output_dir = output_dir
        self.dataset_config = dataset_config
        self.split_config = split_config
        self.num_tiles_base = num_tiles_base
        logger.info(f"Exporter initialized for output directory: {output_dir}")

    @abstractmethod
    def export(self, data_df: pd.DataFrame):
        """
        Abstract method to export data.
        Must be implemented by concrete exporter classes.
        """
        pass

