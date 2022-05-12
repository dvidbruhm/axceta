import json

import pandas as pd

from config.config import logger


def get_silo_values(data: pd.DataFrame) -> dict:
    """
    metadata = data.drop_duplicates(subset=["metadata"])[["LocationName", "metadata"]].dropna()
    print(len(metadata))
    print(metadata["LocationName"])
    for location, md in metadata.values:
        md = json.loads(md)
        logger.info(f"{location}, {md}")
    """
    return {}


def dist_to_volume(dist: float, silo_name: str, conversion_data: pd.DataFrame) -> pd.DataFrame:
    print(conversion_data.columns)
    silo_data = conversion_data.loc[conversion_data["LocationName"] == silo_name]

    h1, h2, h3 = silo_data["H1"].values[0], silo_data["H2"].values[0], silo_data["H3"].values[0]
    print(h1, h2, h3)

    return pd.DataFrame()
