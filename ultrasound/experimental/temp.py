from numpy import matrixlib
import pandas as pd
import json
import numpy as np
import tools.ultrasound_algos as ua
import matplotlib.pyplot as plt
import scipy.signal as signal
import tools.utils as utils



df = pd.read_csv("data/paquette/Paquette2-6.csv")
df2 = df[["temperature", "pulseCount", "samplingFrequency", "raw_data", "maxBinIndex"]]

wfs = []
max_bin_signals = []
for i in range(len(df2)):
    row = df2.iloc[i]
    raw_data = utils.str_raw_data_to_list(row["raw_data"])
    print(int(6000 * row["samplingFrequency"] / 1e6))
    max_bin_signal = sum(raw_data[row["maxBinIndex"]:row["maxBinIndex"] + int(6000 * row["samplingFrequency"] / 1e6)]) / max(raw_data[ua.detect_main_bang_v2(raw_data, 15, 1000, 10000, 20000):])
    max_bin_signals.append(max_bin_signal)
    if max_bin_signal < 25:
        wf = ua.wavefront(raw_data, row["temperature"], 0.5, 20, row["pulseCount"], row["samplingFrequency"])
    else:
        wf = row["maxBinIndex"]
    wfs.append(wf)

    if i in [522, 523, 524]:
        plt.plot(raw_data)
        plt.axvline(ua.detect_main_bang_v2(raw_data, 15, 1000, 10000, 20000))
        plt.axvline(wf, color="green")
        plt.axvline(df["wavefront"][i], color="red")
        plt.axvline(row["maxBinIndex"], linestyle="--")
        plt.show()
df2["wavefront"] = wfs
df2["threshold"] = 0.5


plt.plot(df["wavefront"], label="v1")
plt.plot(df2["wavefront"], label="v2")
plt.plot(max_bin_signals)
plt.legend()
plt.show()

df2 = df2[["raw_data", "threshold", "pulseCount", "samplingFrequency", "maxBinIndex", "wavefront"]]
df2["raw_data"] = df2.apply(lambda x: str(x["raw_data"]).replace(",", ";"), axis=1)
df2.to_csv("data/tests/DistanceComputerEmptySiloTestData.csv", index=False)
exit()





df1 = pd.read_csv("./data/dashboard/silos/Mpass7.csv")
df1 = df1[["pulseCount", "rawdata"]].iloc[0:100].reset_index()
df1["samplingRate"] = [500000] * len(df1)
df1["threshold"] = [0.5] * len(df1)
df1["cutoffFreq"] = [150] * len(df1)
df1["noDataThreshold"] = [20] * len(df1)
df1 = df1[["rawdata", "threshold", "pulseCount",
           "samplingRate", "cutoffFreq", "noDataThreshold"]]

df = pd.read_csv("./data/random/errors1.csv")
df = df[["pulseCount", "raw_data", "samplingFrequency"]].iloc[0:].reset_index()
df["samplingRate"] = df["samplingFrequency"]
df["rawdata"] = df["raw_data"]

df["threshold"] = [0.5] * len(df)
df["cutoffFreq"] = [150] * len(df)
df["noDataThreshold"] = [20] * len(df)
df = df[["rawdata", "threshold", "pulseCount",
         "samplingRate", "cutoffFreq", "noDataThreshold"]]

df = pd.concat([df, df1])
wfs = []
low_wfs = []
# dflow = pd.read_csv("experimental/lowpass.csv", header=None,
#                    index_col=False).sort_values(by=[0]).reset_index()
for i in range(len(df)):
    raw = json.loads(df["rawdata"].iloc[i])
    sr = df["samplingRate"].iloc[i]
    thresh = df["threshold"].iloc[i]
    cutoff = df["cutoffFreq"].iloc[i]
    nodata = df["noDataThreshold"].iloc[i]
    pc = df["pulseCount"].iloc[i]
    wf = ua.wavefront(raw, 0, 0.5, 20, pc, sr)
    low_wf = ua.lowpass_wavefront(raw, 0, 0.5, pc, 20, sr, cutoff, nodata)
    wfs.append(wf)
    low_wfs.append(low_wf)

    """
    print(i, dflow[0][i])
    plt.plot(raw, label="raw")
    plt.plot(json.loads(dflow[1][i]), label="c#")
    cutoff_freq = 150
    sample_rate = sr
    b, a = signal.butter(2, cutoff_freq / (sample_rate / 2),
                         'lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, raw)
    plt.plot(lowpass, label="scipy")
    plt.axvline(wf, color="red")
    plt.axvline(low_wf, color="green")
    plt.legend()
    plt.show()
    """

df["rawdata"] = df.apply(lambda x: str(x["rawdata"]).replace(",", ";"), axis=1)
df["wavefront"] = wfs
df["lowpassWavefront"] = low_wfs
df["index"] = list(range(len(df)))
df = df[["index", "rawdata", "threshold", "pulseCount", "samplingRate",
         "cutoffFreq", "noDataThreshold", "wavefront", "lowpassWavefront"]]
print(len(df))
df.to_csv("experimental/test_data.csv", index=False)
