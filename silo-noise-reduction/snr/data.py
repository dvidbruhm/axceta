from pathlib import Path

import pandas as pd

from config.config import logger


def load_silo_data(file_path: Path) -> pd.DataFrame:
    """Load silo data from csv file"""
    logger.info(f"Loading data from {file_path}...")
    data = pd.DataFrame(pd.read_csv(file_path, low_memory=False))
    data["AcquisitionTime"] = pd.to_datetime(data["AcquisitionTime"])
    data["DistanceFO"] /= 1000
    logger.info("Loaded.")
    return data


def load_dist_to_volume_data(file_path: Path) -> pd.DataFrame:
    """Load distance to volume conversion data for every silo"""
    logger.info(f"Loading conversion data from {file_path}")
    data = pd.DataFrame(pd.read_csv(file_path))
    logger.info("Loaded.")
    return data
