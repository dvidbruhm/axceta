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
    #print("H1, H2, H3 : ", h1, h2, h3)
    diam, angle = silo_data["Diametre"].values[0], silo_data["Angle (degré)"].values[0]
    offset = silo_data["Offset du device"].values[0]
    #print("Diameter, angle, offset : ", diam, angle, offset)
    max_d = h3 - h1 - offset
    r1 = diam / 2
    cone_height = h2 - h1
    r2 = r1 - (cone_height * math.tan(math.radians(angle)))
    if "CDPQA" in silo_name:
        r2 = 0.4445
    #print("R1, R2, Cone Height : ", r1, r2, cone_height)
    vol_cone = (1/3) * math.pi * (r1**2 + r2**2 + r1*r2) * cone_height
    h_cyl = h3 - h2
    vol_cyl = math.pi * r1**2 * h_cyl
    #print("VolumeCone, HeightCylinder, VolumeCylinder : ", vol_cone, h_cyl, vol_cyl)
    vol_tot = vol_cyl + vol_cone
    dist_tot = dist + offset
    new_r1 = (cone_height - (dist_tot - h_cyl)) * math.tan(math.radians(angle)) + r2 if dist_tot > h_cyl else 0
    #print("Total Volume, Total distance, New R1 : ", vol_tot, dist_tot, new_r1)
    density = silo_data["densité de moulé (kg/hl)"].values[0]
    if dist_tot <= h_cyl:
        vol_hecto = (math.pi * r1 * r1 * (h_cyl - dist_tot) + vol_cone) * 10
    else:
        vol_hecto = (1/3) * math.pi * (new_r1**2 + r2**2 + new_r1*r2) * (h3 - h1 - dist_tot) * 10

    perc_fill = vol_hecto / vol_tot * 10
    #print("Volume Hecto, Percent : ", vol_hecto, perc_fill)

    #print(vol_tot, dist_tot, new_r1)
    #print(density, vol_hecto, perc_fill)

    return perc_fill


def batch_dist_to_vol(data: pd.DataFrame, conversion_data: pd.DataFrame, silo_name: str) -> pd.DataFrame:

    silo_data = data[data["LocationName"] == silo_name]

    vols = np.zeros_like(silo_data["DistanceFO"].values)
    vols_fo = np.zeros_like(silo_data["DistanceFO"].values)
    vols_cdm = np.zeros_like(silo_data["DistanceFO"].values)
    weights = np.zeros_like(silo_data["DistanceFO"].values)

    for i, (dist_fo, dist_cdm) in enumerate(silo_data[["DistanceFO", "DistanceCDM"]].values):
        vols_fo[i] = dist_to_volume(dist_fo, silo_name, conversion_data)
        vols_cdm[i] = dist_to_volume(dist_cdm, silo_name, conversion_data)
        weight_cdm = 0
        if dist_fo > 2.0:
            weight_cdm = 1
        vols[i] = vols_fo[i] * (1 - weight_cdm) + vols_cdm[i] * weight_cdm
        weights[i] = weight_cdm

    silo_data["perc_filled"] = vols
    silo_data["perc_filled_fo"] = vols_fo
    silo_data["perc_filled_cdm"] = vols_cdm
    silo_data["cdm_weight"] = weights
    return silo_data

if __name__ == "__main__":
    """ Only for testing """
    from pathlib import Path
    import snr.data as data
    conversion_data = data.load_dist_to_volume_data(file_path=Path("data", "dist_to_volume.csv"))
    silo_name = "CDPQA-024"
    vol = dist_to_volume(2.500, silo_name, conversion_data)
    silo_data = conversion_data.loc[conversion_data["LocationName"] == silo_name]
    #for c in silo_data.columns:
    #    print(c, " \t\t ", silo_data[c].values[0])
    print(vol)
