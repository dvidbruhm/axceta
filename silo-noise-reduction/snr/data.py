from pathlib import Path

import pandas as pd

from config.config import logger
from snr.utils import dist_to_volume


def load_silo_data(file_path: Path) -> pd.DataFrame:
    """Load silo data from csv file"""
    logger.info(f"Loading data from {file_path}...")
    data = pd.DataFrame(pd.read_csv(file_path, low_memory=False))
    data["AcquisitionTime"] = pd.to_datetime(data["AcquisitionTime"])
    data["DistanceFO"] /= 1000.0

    # TEMP
    data = data[data["LocationName"].str.contains("Axceta") == False]
    data = data[data["LocationName"].str.contains("Ghost") == False]
    data = data[data["LocationName"].str.contains("Manual") == False]
    data = data[data["LocationName"].str.contains("Massi") == False]
    data = data[data["LocationName"].str.contains("Germec-002B") == False]
    data = data[data["LocationName"].str.contains("Germec-001B") == False]
    data = data[data["LocationName"].str.contains("Lafontaine-001B") == False]
    print(data["DistanceFO"])

    logger.info("Loaded.")
    return data


def load_dist_to_volume_data(file_path: Path) -> pd.DataFrame:
    """Load distance to volume conversion data for every silo"""
    logger.info(f"Loading conversion data from {file_path}")
    data = pd.DataFrame(pd.read_csv(file_path))
    logger.info("Loaded.")
    return data


def add_percent_filled_to_data(file_path: Path, conversion_data: pd.DataFrame) -> None:
    logger.info("Computing volume of grain in silos...")
    data = load_silo_data(file_path)
    locations = data["LocationName"].unique()
    full_data = pd.DataFrame()
    for l in ["Jacobs-001"]:
        print("---------", l)
        silo_data = data[data["LocationName"] == l].copy()
        silo_data["perc_filled"] = data["DistanceFO"].apply(dist_to_volume, args=(l, conversion_data))
        full_data = pd.concat([full_data, silo_data])
        save_path = Path("data", f"silo-data-2-with-percent-{l}.csv")
        logger.info(f"Saving to file {save_path}")
        full_data.to_csv(save_path)
