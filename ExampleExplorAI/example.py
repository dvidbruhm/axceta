import math
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

import algos.algos as algos
import algos.utils as utils
import main


RAW_FOLDER = "raw/"
OUTPUT_FOLDER = "processed/"
PARQUET_OUTPUT = "parquet/"


def read_csv(file):
    df = pd.read_csv(file, converters={"AcquisitionTime": pd.to_datetime, "raw_data": utils.str_raw_data_to_list, "metadata": json.loads})
    sd = []
    for i in range(len(df)):
        metadata = df["metadata"][i]
        metadata = {key: float(val) for key, val in metadata.items()}
        sd.append(metadata)
    df["siloDimensions"] = sd

    df = df.drop(["quality", "signalArea", "maxSignal", "mainBangEnd", "metadata"], axis=1)
    return df


def compute_algos(df):
    # For each entry in the dataset, compute our algorithms: see docs/algos_description.pdf for more info
    df["mainBangEnd"] = df.apply(lambda row: algos.detect_main_bang_v2(row["raw_data"], row["samplingFrequency"]), axis=1)
    df["sensorWavefront"] = df.apply(lambda row: algos.wavefront_empty_and_full_detection(row["raw_data"], 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), axis=1)
    df["signalQuality"] = df.apply(lambda row: algos.signal_quality(row["raw_data"])["quality"], axis=1)
    # df["sensorDistance"] = df.apply(lambda row: utils.tof_to_dist(row["sensorWavefront"], row["temperature"], row["samplingFrequency"]), axis=1)
    # df["sensorWeight"] = df.apply(lambda row: utils.dist_to_volume(row["sensorDistance"], row["siloDimensions"]), axis=1)
    return df


def rename(df):
    df["siloHeight"] = df["maxBinDistance"]
    df["trueWeight"] = df["LcValue_t"]
    df["trueWavefront"] = df["Lc_ToF"] / df["decimationFactor"]
    df = df.drop(["LcValue_t", "Lc_ToF", "maxBinIndex", "maxBinDistance"], axis=1)
    return df


if __name__ == "__main__":
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    dfs = []
    for file_name in glob(RAW_FOLDER + "*.csv"):
        print(f"Processing: {file_name}")
        df = read_csv(file_name)
        df = compute_algos(df)
        df = rename(df)
        dfs.append(df)
        print(f"current df len: {len(df)}")
    total_df = pd.concat(dfs).reset_index(drop=True)
    print(f"{len(total_df)}")

    total_df.to_csv(OUTPUT_FOLDER + "dataset.csv", index=False)

    # filtered_df = df.loc[df.groupby("batchId")["signalQuality"].idxmax()].sort_values(by=["AcquisitionTime"])
    # plt.plot(filtered_df["AcquisitionTime"], filtered_df["trueWavefront"], label="true")
    # plt.plot(filtered_df["AcquisitionTime"], filtered_df["sensorWavefront"], label="sensor")
    # plt.legend()
    # plt.show()
    #
    # exit()
