import matplotlib.pyplot as plt
import pandas as pd
import json
import tools.ultrasound_algos as ua
import us.utils as utils


def test_no_wavefront_with_nice_peak_error():
    data = pd.read_csv("data/random/paquette_error1.csv")
    raw_signal = data["raw_data"].values
    temp = data.loc[0, "temperature"]
    wf = ua.wavefront(raw_signal, temp, 0.5, 20, 31, sample_rate=500000/25)
    plt.plot(raw_signal)
    plt.axvline(wf)
    plt.show()
    exit()


def test_wavefront_with_window():
    silo_name = "Paquette5-16"
    silo_data = utils.get_silo_data("data/silo_data.csv", silo_name)

    df = pd.read_csv("data/paquette/Paquette5-16 _06-28_to_07-20.csv", converters={"AcquisitionTime": pd.to_datetime, "rawdata": json.loads})
    normal_wfs = []
    new_wfs = []
    new_wfs2 = []
    for i in range(len(df)):
        temp = df.loc[i, "temperature"]
        pc = df.loc[i, "pulseCount"]
        density = df.loc[i, "feedDensity"]
        sampling_rate = int(500000 / 25)
        raw_data = df.loc[i, "rawdata"]
        normal_wf = ua.wavefront(raw_data, temp, 0.5, 10, pc, sampling_rate)
        normal_wfs.append(normal_wf)
        new_wf = ua.wavefront_with_window(raw_data, new_wfs[max(0, i - 48):i], df["AcquisitionTime"].values[max(0, i - 48):i], 0.5, pc, temp, silo_data, density, 2, sampling_rate)
        new_wfs.append(new_wf)
        new_wf2 = ua.wavefront_with_window(raw_data, new_wfs2[max(0, i - 48):i], df["AcquisitionTime"].values[max(0, i - 48):i],
                                           0.5, pc, temp, silo_data, density, 1, sampling_rate, alpha=0.0, max_days_before=5)
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
    plt.plot(new_wfs, label="New wavefront")
    plt.plot(normal_wfs, label="Normal wavefront")
    # plt.plot(new_wfs2, label="New wavefront2")
    plt.legend(loc="best")
    plt.show()

    exit()


if __name__ == "__main__":
    test_wavefront_with_window()
    test_no_wavefront_with_nice_peak_error()
    # Test main bang length vs pulse count
    data2 = pd.read_csv("data/test/raw_data_with_pulse.csv",
                        converters={"UltrasoundData": lambda x: json.loads(x)["data"]})

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

    data = pd.read_csv("data/test/export-20230124.csv",
                       converters={"RawData": json.loads})
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
