import pandas as pd
import tools.ultrasound_algos as ua
import numpy as np
import matplotlib.pyplot as plt
import tools.utils as utils
import statistics

import tools.batch_algos as ba


def half_max_peak_width(data):
    m = max(data)
    max_index = max(range(len(data)), key=lambda i: data[i])
    start, end = 0, 0
    for i in range(max_index, len(data)):
        if data[i] <= m / 2:
            end = i
            break
    for i in range(max_index, 0, -1):
        if data[i] <= m / 2:
            start = i
            break
    width = abs(end - start)
    return width


def display_batch(batch):
    sorted_batch = batch.sort_values(by=["quality"], ascending=False)
    for i in range(len(batch)):
        row = sorted_batch.iloc[i]
        raw_data = row["raw_data"]
        plt.subplot(len(batch), 1, i + 1)
        plt.plot(raw_data, label="Signal")
        plt.axvline(row["wavefront"], linestyle="--", label="wf", color="green")
        plt.axvline(row["bang_end"], label="Bang end", color="red")
        plt.axvline(row["maxBinIndex"], color="black")
        plt.ylabel(f'{row["config"]}')

        x, y = range(len(raw_data[row["bang_end"] :])), raw_data[row["bang_end"] :]
        gmodel = Model(gaussian)
        amp, cen, wid = max(y), np.argmax(y), 10
        result = gmodel.fit(y, x=x, amp=amp, cen=cen, wid=wid)
        plt.plot([i + row["bang_end"] for i in x], result.best_fit, "--", color="orange")

        plt.text(0.5 * len(raw_data), 0.5 * max(raw_data), row["q1"])
        plt.text(0.5 * len(raw_data), 0.4 * max(raw_data), f"{row['q2']} - {result.params['amp'].value} - {result.params['wid'].value}")
        plt.text(0.5 * len(raw_data), 0.3 * max(raw_data), row["q3"])
        plt.legend()

    plt.show()


from numpy import exp, pi, sqrt
from lmfit import Model
import time


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2 * pi) * wid)) * exp(-((x - cen) ** 2) / (2 * wid**2))


def quality3(data, i, sample_rate=20000):
    data = data[:-3]
    bang_end = ua.detect_main_bang_v2(data, sample_rate)
    x, y = np.array(range(len(data[bang_end:]))), np.array(data[bang_end:])

    peaks, proms = ua.custom_find_peaks(y, min_prom=5)
    if len(peaks) == 0:
        return 0

    quality = max(y) / ((max(proms) - np.mean(proms)) + len(peaks))
    return quality * 100

    print(peaks, proms)
    plt.plot(x, y)
    plt.plot(x[peaks], y[peaks], "o")
    plt.show()
    exit()


def quality2(data, i, sample_rate=20000):
    data = data[:-3]
    bang_end = ua.detect_main_bang_v2(data, sample_rate)
    x, y = range(len(data[bang_end:])), data[bang_end:]

    gmodel = Model(gaussian)
    amp, cen, wid = max(y), np.argmax(y), 10
    result = gmodel.fit(y, x=x, amp=amp, cen=cen, wid=wid)
    if np.mean(result.best_fit) < 0.1:
        return 0

    # error
    a = result.best_fit > result.best_fit.max() * 0.2
    # try:
    first, last = np.nonzero(a)[0][0], np.nonzero(a)[0][-1]
    # except:
    #     plt.plot(x, y, label="data")
    #     plt.plot(x, result.init_fit, "--", label="init")
    #     plt.plot(x, result.best_fit, "--", label="best")
    #     plt.plot(x, +result.best_fit)
    #     plt.legend()
    #     plt.show()

    diff = y - result.best_fit
    err_list = sum([0 if v < 0 else v for v in diff[first:last]])
    err = 100 * max(y) / ((err_list / 10) + (result.params["wid"].value)) if err_list > 0 else np.inf

    # plt.plot(x, y, label="data")
    # plt.plot(x, result.init_fit, "--", label="init")
    # plt.plot(x, result.best_fit, "--", label="best")
    # plt.plot(x, +result.best_fit)
    # plt.axvline(first)
    # plt.axvline(last)
    # plt.legend()
    # plt.show()

    return err


def test_qualities():
    df = pd.read_csv("data/random/test_quality3.csv", converters={"AcquisitionTime": pd.to_datetime, "raw_data": utils.str_raw_data_to_list}, nrows=1000)
    df["bang_end"] = df.apply(lambda row: ua.detect_main_bang_v2(row["raw_data"], row["samplingFrequency"]), axis=1)
    df["wavefront"] = df.apply(lambda row: ua.wavefront_empty_and_full_detection(row["raw_data"], 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), axis=1)
    df["q1"] = df.apply(lambda row: ua.signal_quality(row["raw_data"])["quality"], axis=1)
    # df["q2"] = df.apply(lambda row: quality2(row["raw_data"], row.name), axis=1)
    # df["q3"] = df.apply(lambda row: quality3(row["raw_data"], row.name), axis=1)
    #
    df_q1 = df.loc[df.groupby("batchId")["q1"].idxmax()].sort_values(by=["AcquisitionTime"]).reset_index()
    # df_q2 = df.loc[df.groupby("batchId")["q2"].idxmax()].sort_values(by=["AcquisitionTime"]).reset_index()
    # df_q3 = df.loc[df.groupby("batchId")["q3"].idxmax()].sort_values(by=["AcquisitionTime"]).reset_index()
    plt.plot(df_q1["wavefront"], label="q1")
    # plt.plot(df_q2["wavefront"], label="q2")
    # plt.plot(df_q3["wavefront"], label="q3", color="green")
    # plt.plot(df_q3[df_q3["q3"] > 75]["wavefront"], "x", label="q3 > 80", color="green")
    # plt.plot(df_q2["q2"], "--", alpha=0.2, label="qual2")
    # plt.plot(df_q3["q3"], "--", alpha=0.2, label="qual3")
    plt.legend()
    plt.show()

    for i, id in enumerate(df["batchId"].unique()):
        if i in list(range(40, 60)):
            batch = df[df["batchId"] == id].reset_index()
            ba.is_silo_empty(batch.to_dict(orient="records"))
            # display_batch(batch)


if __name__ == "__main__":
    test_qualities()
