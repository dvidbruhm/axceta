from pathlib import Path

import pandas as pd
import numpy as np

from config.config import logger
from snr.utils import dist_to_volume


def load_silo_data(file_path: Path, scale_dist: bool = False) -> pd.DataFrame:
    """Load silo data from csv file"""
    logger.info(f"Loading data from {file_path}...")
    data = pd.DataFrame(pd.read_csv(file_path, low_memory=False))
    data["AcquisitionTime"] = pd.to_datetime(data["AcquisitionTime"])
    if scale_dist:
        data["DistanceFO"] /= 1000.0
        data["DistanceCDM"] /= 1000.0

    # TEMP
    data = data[data["LocationName"].str.contains("Axceta") == False]
    data = data[data["LocationName"].str.contains("Ghost") == False]
    data = data[data["LocationName"].str.contains("Manual") == False]
    data = data[data["LocationName"].str.contains("Massi") == False]
    data = data[data["LocationName"].str.contains("Germec-002B") == False]
    data = data[data["LocationName"].str.contains("Germec-001B") == False]
    data = data[data["LocationName"].str.contains("Lafontaine-001B") == False]

    data = data[data["UltrasonicEchoSpread"] > 1]

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
    data = load_silo_data(file_path, scale_dist=True)
    locations = data["LocationName"].unique()
    full_data = pd.DataFrame()
    for l in locations:
        logger.info(f"------- Processing file {l} --------")
        silo_data = data[data["LocationName"] == l].copy()
        vols = np.zeros_like(silo_data["DistanceFO"].values)
        weights = np.zeros_like(silo_data["DistanceFO"].values)

        for i, (dist_fo, dist_cdm) in enumerate(silo_data[["DistanceFO", "DistanceCDM"]].values):
            dist = dist_fo
            weight_cdm = 0
            if dist_fo > 2.0:
                dist = dist_cdm
                weight_cdm = 1
            vols[i] = dist_to_volume(dist, l, conversion_data)
            weights[i] = weight_cdm
        silo_data["perc_filled"] = vols
        silo_data["cdm_weight"] = weights

        full_data = pd.concat([full_data, silo_data])
    save_path = Path("data", f"silo-data-3-with-percent.csv")
    logger.info(f"Saving to file {save_path}")
    full_data.to_csv(save_path)
