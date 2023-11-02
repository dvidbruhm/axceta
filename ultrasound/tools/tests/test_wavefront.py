import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import tools.ultrasound_algos as ua
import tools.utils as utils
import us.utils
import sys
import tools.smoothing as sm

sys.path.insert(0, "/home/david/Projects/experiments-api/app/experiments/")
import algos


def test_wavefront_full_silo():
    df = pd.read_csv("data/random/export-4.csv")
    df = df.dropna()

    conv = 20000 / 1e6
    df["wf"] = df.apply(
        lambda row: ua.wavefront_empty_and_full_detection(utils.str_raw_data_to_list(row["raw_data"]), 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"] * conv),
        axis=1,
    )
    df["wf_exp"] = df.apply(
        lambda row: algos.wavefront_empty_and_full_detection(utils.str_raw_data_to_list(row["raw_data"]), 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"] * conv),
        axis=1,
    )
    df["quality"] = df.apply(
        lambda row: ua.signal_quality(utils.str_raw_data_to_list(row["raw_data"]))["c2"],
        axis=1,
    )
    df["quality_exp"] = df.apply(
        lambda row: algos.signal_quality(utils.str_raw_data_to_list(row["raw_data"]))["quality"],
        axis=1,
    )
    batch_ids = df["batchId"].unique()

    prod_wfs = []
    new_wfs = []
    exp_wfs = []
    for i, id in enumerate(batch_ids):
        batch = df[df["batchId"] == id].reset_index()
        prod_wfs.append(batch["WavefrontIndex_prod"][batch["WaveformQuality_prod"].argmax()])
        new_wfs.append(batch["wf"][batch["quality"].argmax()])
        exp_wfs.append(batch["wf_exp"][batch["quality_exp"].argmax()])

        if i in [825]:
            for i in range(len(batch)):
                plt.subplot(len(batch), 1, i + 1)
                row = batch.iloc[i]
                raw = utils.str_raw_data_to_list(row["raw_data"])
                new_be = ua.detect_main_bang_v2(raw, row["samplingFrequency"])
                plt.plot(raw)
                plt.axvline(row["wf"], color="orange", label="new wf")
                plt.axvline(row["wf_exp"], color="red", label="exp wf")
                plt.axvline(row["WavefrontIndex_prod"], color="blue", linestyle="--", label="prod wf")
                plt.axvline(new_be, color="yellow", linestyle="--", label="new be")
                plt.title(f"exp: {row['quality_exp']} / prod: {row['WaveformQuality_prod']} / new: {row['quality']}")
                plt.legend()
            plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(np.array(prod_wfs), label="prod", color="blue")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(np.array(new_wfs), label="new", color="orange")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(exp_wfs, label="exp", color="red")
    plt.legend()
    plt.show()
    exit()

    for i in range(len(df)):
        # plt.subplot(len(df), 1, i + 1)
        row = df.iloc[i]
        raw = utils.str_raw_data_to_list(row["raw_data"])
        new_be = ua.detect_main_bang_v2(raw, row["samplingFrequency"])
        wf = ua.wavefront_empty_and_full_detection(raw, 0.5, row["pulseCount"], row["samplingFrequency"], 5000)
        # plt.title(ua.signal_quality(raw, row["samplingFrequency"])["c2"])
        # plt.plot(raw)
        # plt.axvline(wf)
        # plt.axvline(be, color="orange")
    plt.show()


def test_no_wavefront_with_nice_peak_error():
    data = pd.read_csv("data/random/paquette_error1.csv")
    raw_signal = data["raw_data"].values
    temp = data.loc[0, "temperature"]
    wf = ua.wavefront(raw_signal, temp, 0.5, 20, 31, sample_rate=500000 / 25)
    plt.plot(raw_signal)
    plt.axvline(wf)
    plt.show()
    exit()


def test_wavefront_with_window():
    silo_name = "Paquette5-16"
    silo_data = us.utils.get_silo_data("data/silo_data.csv", silo_name)

    df = pd.read_csv("data/random/ax90.csv", converters={"AcquisitionTime": pd.to_datetime, "raw_data": json.loads})
    normal_wfs = []
    new_wfs = []
    new_wfs2 = []
    for i in range(len(df)):
        temp = df.loc[i, "temperature"]
        pc = df.loc[i, "pulseCount"]
        density = 70
        sampling_rate = int(500000 / 25)
        raw_data = df.loc[i, "raw_data"]
        normal_wf = ua.wavefront(raw_data, temp, 0.5, 10, pc, sampling_rate)
        normal_wfs.append(normal_wf)
        new_wf = ua.wavefront_with_window(raw_data, new_wfs[max(0, i - 48) : i], df["AcquisitionTime"].values[max(0, i - 48) : i], 0.5, pc, temp, silo_data, density, 1, sampling_rate)
        new_wfs.append(new_wf)
        new_wf2 = ua.wavefront_with_window(
            raw_data,
            new_wfs2[max(0, i - 48) : i],
            df["AcquisitionTime"].values[max(0, i - 48) : i],
            0.5,
            pc,
            temp,
            silo_data,
            density,
            1,
            sampling_rate,
            alpha=0.5,
            max_days_before=5,
        )
        new_wfs2.append(new_wf2)

        if i in [1500]:
            plt.plot(raw_data)
            plt.axvline(normal_wf, color="green")
            plt.axvline(new_wf, color="blue")
            plt.axvline(new_wf2, color="yellow")
            plt.show()

    nb_plots = 3
    plt.subplot(nb_plots, 1, 1)
    plt.plot(normal_wfs, label="Normal wavefront")
    plt.legend(loc="best")
    plt.subplot(nb_plots, 1, 2)
    plt.plot(df["WavefrontIndex"].values, label="Cloud wavefront")
    plt.legend(loc="best")
    plt.subplot(nb_plots, 1, 3)
    plt.plot(normal_wfs, label="Normal wavefront", alpha=0.5)
    plt.plot(new_wfs2, label="New wavefront")
    # plt.plot(new_wfs2, label="New wavefront2")
    plt.legend(loc="best")
    plt.show()

    exit()


if __name__ == "__main__":
    test_wavefront_with_window()
    exit()
    test_wavefront_full_silo()
    test_no_wavefront_with_nice_peak_error()
    # Test main bang length vs pulse count
    data2 = pd.read_csv("data/test/raw_data_with_pulse.csv", converters={"UltrasoundData": lambda x: json.loads(x)["data"]})

    le = [1000, 1300, 1500, 1700]
    for i, pulse in enumerate(sorted(data2["Pulse"].unique())):
        fil_data = data2[data2["Pulse"] == pulse].reset_index()
        for j in range(1, len(fil_data)):
            plt.subplot(len(fil_data), 1, j)
            plt.plot(fil_data.loc[j, "UltrasoundData"])
            plt.axvline(le[i])
            plt.title(f"current: {data2.loc[j, 'CurrentLimit']}")
        plt.xlabel(f"Pulse : {pulse}")
        plt.show()

    exit()

    data = pd.read_csv("data/test/export-20230124.csv", converters={"RawData": json.loads})
    print(len(data))

    n = len(data)
    for i in range(n):
        raw_data = data.loc[i, "RawData"]
        wf_index = ua.wavefront(raw_data, data.loc[i, "Temperature"], 0.5, 1.5)
        bang_index = ua.detect_main_bang_end(raw_data)
        print(wf_index, bang_index)

        plt.subplot(n, 1, i + 1)
        plt.axvline(bang_index, color="orange")
        plt.axvline(wf_index, color="green")
        plt.plot(raw_data)
    # plt.show()
