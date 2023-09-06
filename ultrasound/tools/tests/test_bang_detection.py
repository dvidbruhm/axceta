import pandas as pd
import tools.utils as utils
import matplotlib.pyplot as plt
import tools.ultrasound_algos as ua
import json
import polars as pl
import numpy as np
from scipy.signal import find_peaks


density = 70

def select_best_wavefront_in_batch(df, batch_column, wavefront_column):
    df2 = df.group_by("batchId").agg(pl.map_groups(exprs=["AcquisitionTime", "wavefront"], function=select_wavefront_and_time).alias("groupby"))


def select_wavefront_and_time(list_of_series):
    i = list_of_series[1].arg_max()
    return list_of_series[0][i], list_of_series[1][i]


def test_bang_detection_polars():
    df = pl.read_csv("data/random/small_silo_full.csv", infer_schema_length=10000, dtypes={"AcquisitionTime": pl.Datetime})
    select_best_wavefront_in_batch(df, "batchId", "wavefront")
    plt.plot(df.select(pl.col("AcquisitionTime")), df.select(pl.col("wavefront")))

    df2 = df.group_by("batchId", maintain_order=True).agg([pl.col("wavefront").max()])
    print(df)
    df = df2.join(df, on=["batchId", "wavefront"])
    print(df)
    print(df2)

    df3 = df.group_by("batchId").agg(pl.map_groups(exprs=["AcquisitionTime", "wavefront"], function=select_wavefront_and_time))
    plt.plot(df2.select(pl.col("AcquisitionTime")), df2.select(pl.col("wavefront")))
    plt.show()


def main_bang_end_peaks(data):
    data = np.array(data)
    peaks, props = find_peaks(abs(data - 255), prominence=0)
    max_prom_index = np.argmax(props["prominences"])
    return peaks[max_prom_index]


def test_bang_detection():
    df = pd.read_csv("data/random/quality_bad_2.csv")
    raw = np.array(utils.str_raw_data_to_list(df["raw_data"][0]))
    bang_end_v2 = ua.detect_main_bang_v2(raw, 15, 1000, 10000, 20000)
    print(sum(raw) / len(raw))
    print(ua.signal_quality(raw[bang_end_v2:]))
    plt.plot(raw)
    plt.axvline(bang_end_v2)
    plt.show()

    df = pd.read_csv("data/random/Paquette7-2.csv")

    # df = utils.select_silo(df, "Avinor-1484A")

    batch_ids = df["batchId"].unique()
    wfs = []
    weights = []
    cloud_wfs = []
    cloud_weights = []
    bang_ends = []
    qualities = []
    for j, batch_id in enumerate(batch_ids):
        print(f"Batch {j}")
        batch = df[df["batchId"] == batch_id].reset_index()
        batch_wfs = []
        batch_qualities = []
        for i in range(len(batch)):
            row = batch.loc[i]
            raw = np.array(utils.str_raw_data_to_list(row["raw_data"]))
            raw2 = raw.copy()
            raw2[0:20] = 0
            bang_end_v1 = ua.detect_main_bang_end(raw, row["pulseCount"], row["samplingFrequency"])
            bang_end_v2 = ua.detect_main_bang_v2(raw, sum(raw) / len(raw), 1000, 10000, 20000)
            bang_end_v3 = main_bang_end_peaks(raw2)
            bang_ends.append(bang_end_v2)
            wf = ua.wavefront(raw, 0, 0.5, 20, row["pulseCount"], 20000, bang_end=bang_end_v3)
            batch_wfs.append(wf)
            batch_qualities.append(ua.signal_quality(raw[bang_end_v2:])["c2"])
            if i == 0:
                silo_data = json.loads(row["metadata"])
                for k in silo_data.keys():
                    silo_data[k] = float(silo_data[k])
                temp = row["temperature"]
            if j in [5, 15, 25, 35, 45, 55, 65, 100, 150, 160, 170, 180, 200, 210] and True:
                raw = np.array(raw)
                peaks, props = find_peaks(abs(raw - 255), prominence=0)
                max_prom_index = np.argmax(props["prominences"])
                plt.subplot(len(batch), 1, i + 1)
                plt.plot(raw)
                plt.plot(peaks[max_prom_index], raw[peaks[max_prom_index]], "x")
                plt.axvline(bang_end_v1, color="green")
                plt.axvline(bang_end_v2, color="red")
                plt.axvline(bang_end_v3, linestyle="--")
                plt.text(600, 100, f"{round(batch_qualities[-1], 2)}")
                plt.text(600, 200, f"{round(sum(raw) / len(raw), 2)}")
        plt.show()

        wfs.append(batch_wfs[np.argmax(batch_qualities)])
        cloud_wfs.append(max(batch["WavefrontIndex"].values))

        qualities.append(max(batch_qualities))

        dist = utils.tof_to_dist(max(batch_wfs) * 25, temp)
        weights.append(utils.dist_to_volume_agco(dist, silo_data) * density / 100)

        cloud_dist = utils.tof_to_dist(max(batch["WavefrontIndex"].values) * 25, temp)
        cloud_weights.append(utils.dist_to_volume_agco(cloud_dist, silo_data) * density / 100)

        if j in []:
            plt.show()

    print()
    tim = pd.to_datetime(df.groupby("batchId").last().reset_index(drop=True)["AcquisitionTime"]).sort_values(ascending=True)
    print(len(cloud_wfs))
    plt.plot(tim, cloud_wfs, label="cloud")
    plt.plot(tim, wfs, label="local")
    plt.plot(tim, qualities)
    plt.legend()
    plt.show()

    
    df = df[["raw_data", "samplingFrequency"]]
    df["raw_data"] = df.apply(lambda x: str(x["raw_data"]).replace(",", ";"), axis=1)
    df["minThreshold"] = 15
    df["windowSize"] = 1000
    df["maxBangLen"] = 10000
    df["bangEnd"] = bang_ends
    df.to_csv("data/tests/BangEndTestData.csv", index=False)
    print(df)


if __name__ == "__main__":
    # test_bang_detection_polars()
    test_bang_detection()
