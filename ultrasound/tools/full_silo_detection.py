import matplotlib.pyplot as plt
import pandas as pd
import tools.ultrasound_algos as ua
import tools.utils as utils
import numpy as np


def process_data(data):
    pass


def full_silo_detection():
    # df = pd.read_csv("data/random/small_silo_full.csv")
    file = "autre série ou on. ades gros prob de ToF.csv"
    df = pd.read_csv(f"data/random/{file}")
    df["raw_data"] = df.apply(lambda x: utils.str_raw_data_to_list(x["raw_data"]), axis=1)

    df["bang_end"] = df.apply(lambda x: ua.detect_main_bang_v2(x["raw_data"], 0, 1000, 10000, x["samplingFrequency"]), axis=1)
    df["wavefront2"] = df.apply(lambda x: ua.wavefront_empty_detection(x["raw_data"], 0.5, x["pulseCount"], x["samplingFrequency"], x["maxBinIndex"]), axis=1)
    df["wavefront3"] = df.apply(
        lambda x: ua.wavefront_empty_and_full_detection(
            x["raw_data"], 0.5, x["pulseCount"], x["samplingFrequency"], x["maxBinIndex"], temperature=x["temperature"], window_in_meters=1
        ),
        axis=1,
    )
    df["signal_before_bang_end"] = df.apply(lambda x: sum(x["raw_data"][: x["bang_end"]]), axis=1)
    df["signal_after_bang_end"] = df.apply(lambda x: sum(x["raw_data"][x["bang_end"] :]), axis=1)
    df["ratio"] = df.apply(lambda x: x["signal_after_bang_end"] / x["signal_before_bang_end"], axis=1)
    df["quality"] = df.apply(lambda x: ua.signal_quality(x["raw_data"], x["samplingFrequency"])["c2"], axis=1)

    inds = [3]
    batch_data = {"index": [], "wf": [], "ratio": []}
    best_df = []
    for j, batch_id in enumerate(df["batchId"].unique()):
        batch = df[df["batchId"] == batch_id].reset_index()
        sum_data = np.array(batch.loc[0]["raw_data"])
        for i in range(len(batch)):
            row = batch.loc[i]
            print("-------------------------------------------")
            print(j, i, row["quality"], ua.wavefront_empty_and_full_detection(row["raw_data"], 0.5, 0, row["samplingFrequency"], row["maxBinIndex"]))
            if i > 0:
                sum_data += np.array(row["raw_data"])
            if j not in inds:
                continue
            plt.subplot(len(batch) + 1, 1, i + 1)
            plt.plot(row["raw_data"])
            plt.axvline(row["bang_end"])
            # plt.axvline(row["WavefrontIndex"]+1, color="yellow")
            plt.axvline(row["maxBinIndex"], color="red")
            plt.axvline(row["LC_ToF"] * row["samplingFrequency"] / 1e6, color="green")
            plt.axvline(row["wavefront2"], color="blue")
            plt.axvline(row["wavefront3"], color="orange")

            plt.text(600, 100, f"{row['signal_after_bang_end']}")
            plt.text(0, 100, f"{row['signal_before_bang_end']}")
            plt.text(300, 100, f"ratio:{row['signal_after_bang_end'] / row['signal_before_bang_end']}")
            plt.text(300, 230, f"quality:{row['quality']}")

        best_row = batch.loc[batch["quality"].argmax()]
        best_df.append(best_row)

        # sum_data = ((sum_data - sum_data.min()) / (sum_data.max() - sum_data.min())) * 255
        sum_data[sum_data >= 255] = 255
        be = ua.detect_main_bang_v2(sum_data, 0, 1000, 10000, batch.loc[0]["samplingFrequency"])
        wf = ua.wavefront_empty_and_full_detection(sum_data, 0.5, batch.loc[0]["pulseCount"], batch.loc[0]["samplingFrequency"], batch.loc[0]["maxBinIndex"])
        ratio = sum(sum_data[be:]) / sum(sum_data[:be])

        batch_data["wf"].append(wf)
        batch_data["ratio"].append(ratio)
        if len(batch_data["index"]) > 0:
            batch_data["index"].append(len(batch) + batch_data["index"][-1])
        else:
            batch_data["index"].append(0)

        if j not in inds:
            continue
        plt.subplot(len(batch) + 1, 1, len(batch) + 1)
        plt.plot(sum_data)
        plt.axvline(wf, color="orange")
        plt.axvline(be, color="blue")
        plt.axvline(batch.loc[0]["LC_ToF"] * batch.loc[0]["samplingFrequency"] / 1e6, color="green")
        plt.text(300, 100, f"{ratio}")
        plt.text(300, 230, f"quality:{ua.signal_quality(sum_data)['c2']}")
        plt.show()

    best_df = pd.DataFrame(best_df).reset_index()
    # plt.plot(best_df["wavefront2"], label="wavefront ancien")
    # plt.plot(best_df["ratio"] * 1000, "--", label="ratio")
    plt.plot(best_df["WavefrontIndex"], label="wavefront cloud", color="orange")
    plt.plot(best_df["wavefront3"], label="wavefront nouveau", color="blue")
    # plt.plot(df["signal_before_bang_end"] / max(df["signal_before_bang_end"]), label="signal_before")
    # plt.plot(df["signal_after_bang_end"] / max(df["signal_after_bang_end"]), label="signal_after")
    # plt.plot(best_df["ratio"] * 100, label="ratio")
    plt.plot(best_df["LC_ToF"] * df["samplingFrequency"] / 1e6, "--", label="Loadcell")
    # plt.plot(batch_data["wf"], "-x")
    # plt.plot(np.array(batch_data["ratio"]) * 1000, "-o")
    # plt.axhline(200, linestyle="--")
    plt.legend()
    plt.title(f"{file}")
    plt.show()


if __name__ == "__main__":
    full_silo_detection()
