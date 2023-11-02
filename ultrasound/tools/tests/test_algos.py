import pandas as pd
import tools.utils as utils
import tools.ultrasound_algos as ua
import matplotlib.pyplot as plt


def best_quality(x):
    print(x)
    exit()
    return x


if __name__ == "__main__":
    df = pd.read_csv("data/random/bugAlgo(4silos).csv")
    df = df[df["LocationName"] == "elmire-2-6322"].reset_index()

    df["raw_data"] = df.apply(lambda x: utils.str_raw_data_to_list(x["raw_data"]), axis=1)
    df["quality"] = df.apply(lambda x: ua.signal_quality(x["raw_data"], x["samplingFrequency"])["c2"], axis=1)

    df = df.loc[df.groupby("batchId")["quality"].idxmax()].sort_values(by="AcquisitionTime").reset_index()
    df["wf"] = df.apply(lambda x: ua.wavefront_empty_and_full_detection(x["raw_data"], 0.5, x["pulseCount"], x["samplingFrequency"], x["maxBinIndex"]), axis=1)
    print(df)

    plt.plot(df["latest_Algo"] / 50, label="cloud algo")
    plt.plot(df["wf"], label="new algo")
    plt.legend()
    plt.show()

    inds = [40, 41]
    import numpy as np
    for i, ind in enumerate(inds, start=1):
        row = df.loc[ind]
        plt.subplot(len(inds), 1, i)
        plt.plot(row["raw_data"])
        be = ua.detect_main_bang_v2(row["raw_data"], 0, 1000, 10000, row["samplingFrequency"])
        plt.fill_between(range(len(row["raw_data"])), row["raw_data"], where=np.array(range(len(row["raw_data"]))) > be, color="red", alpha=0.15)
        plt.axvline(be, color="orange")
        plt.axvline(row["maxBinIndex"], color="black", linestyle="--")
        plt.axvline(row["wf"], color="green")
        plt.axvline(row["latest_Algo"] / 50, color="pink")
    plt.show()
