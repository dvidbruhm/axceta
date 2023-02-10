import matplotlib.pyplot as plt
import pandas as pd
import json
import tools.wavefront as wf

if __name__ == "__main__":
    # Test main bang length vs pulse count
    data2 = pd.read_csv("data/test/raw_data_with_pulse.csv", converters={"UltrasoundData": lambda x: json.loads(x)["data"]})
    print(len(data2))
    print(data2.columns)
    print(data2["Pulse"].unique())

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
        wf_index = wf.wavefront(raw_data, data.loc[i, "Temperature"], 0.5, 1.5)
        bang_index = wf.detect_main_bang_end(raw_data)
        print(wf_index, bang_index)

        plt.subplot(n, 1, i + 1)
        plt.axvline(bang_index, color="orange")
        plt.axvline(wf_index, color="green")
        plt.plot(raw_data)
    # plt.show()
