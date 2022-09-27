import numpy as np
import pandas as pd

import json
from scipy.signal import find_peaks

def raw_data_quality(data: np.ndarray) -> float:
    """
    Computes a quality index for a raw ultrasound reading

    Parameters
    ----------
    data : np.ndarray
        Raw ultrasound data
    plot : bool
        If true, plot the input ultrasound data to visualise the quality

    Returns
    -------
    float quality:
        Quality index on a 1-5 scale. 5 means good quality, 1 means poor quality.
    """
    min_peak_height = 40

    # Remove main bang
    data = data[3000:]
    
    maxs, _ = find_peaks(data, height=min_peak_height, width=20)

    if len(maxs) == 0:
        return 1

    max_peak_i = np.argmax(data[maxs])
    maxs_without_biggest = maxs[maxs != maxs[max_peak_i]]

    hist, _ = np.histogram(data[maxs], bins=[0, 80, 120, 160, 200, 256], density=False)
    #hist = hist / np.max(hist)
    global_max = 2 * np.max(data) / 255

    if hist[4] > 0:
        quality = 5
        if hist[4] > 3:
            quality -= 1
    elif hist[3] > 0:
        quality = 4
        if hist[3] > 4:
            quality -= 1
    elif hist[2] > 0:
        quality = 3
        if hist[2] > 5:
            quality -= 1
    elif hist[1] > 0:
        quality = 2
        if hist[1] > 6:
            quality -= 1
    else:
        quality = 1

    quality = quality * global_max

    return quality

# TO REMOVE
df = pd.DataFrame(pd.read_csv("ADX_plugin/test_batch.csv"))
# ---------

df["UltrasoundData_data"] = df.apply(lambda row: np.array(json.loads(row["UltrasoundData_data"])), axis=1)

result = df

# Assign quality index as new column in the output DataFrame
result["raw_data_quality"] = result.apply(lambda x: raw_data_quality(x["UltrasoundData_data"]), axis=1)
