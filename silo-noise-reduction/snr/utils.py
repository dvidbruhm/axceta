import json

import numpy as np
import pandas as pd
import math

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


def dist_to_volume(dist: float, silo_name: str, conversion_data: pd.DataFrame) -> np.ndarray:
    #print(conversion_data.columns)
    silo_data = conversion_data.loc[conversion_data["LocationName"] == silo_name]


    #print(silo_data)
    #print(silo_name)
    #print(conversion_data)
    h1, h2, h3 = silo_data["H1"].values[0], silo_data["H2"].values[0], silo_data["H3"].values[0]
    #print(h1, h2, h3)
    diam, angle = silo_data["Diametre"].values[0], silo_data["Angle (degré)"].values[0]
    offset = silo_data["Offset du device"].values[0]
    #print(diam, angle, offset)
    max_d = h3 - h1 - offset
    r1 = diam / 2
    cone_height = h2 - h1
    r2 = r1 - (cone_height * math.tan(math.radians(angle)))
    #print(r1, r2, cone_height)
    vol_cone = (1/3) * math.pi * (r1**2 + r2**2 + r1*r2) * cone_height
    h_cyl = h3 - h2
    vol_cyl = math.pi * r1**2 * h_cyl
    #print(vol_cone, h_cyl, vol_cyl)
    vol_tot = vol_cyl + vol_cone
    dist_tot = dist + offset
    new_r1 = cone_height - (dist_tot - h_cyl) * math.tan(math.radians(angle)) + r2 if dist_tot > h_cyl else 0
    density = silo_data["densité de moulé (kg/hl)"].values[0]

    if dist_tot <= h_cyl:
        vol_hecto = (math.pi * r1**2 * (h_cyl - dist_tot) + vol_cone) * 10
    else:
        vol_hecto = (1/3) * math.pi * (new_r1**2 + r2**2 + new_r1*r2) * (h3 - h1 - dist_tot) * 10

    perc_fill = vol_hecto / vol_tot * 10

    #print(vol_tot, dist_tot, new_r1)
    #print(density, vol_hecto, perc_fill)

    return perc_fill
