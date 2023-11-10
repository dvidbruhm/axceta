import pandas as pd
import matplotlib.pyplot as plt
import tools.utils as utils
import tools.prod_algos as pa


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

        plt.text(0.5 * len(raw_data), 0.5 * max(raw_data), row["quality"])
        plt.legend()

    plt.show()


def main():
    file = "data/tests_silo_vide/export-2.csv"
    df = pd.read_csv(file, converters={"raw_data": utils.str_raw_data_to_list, "AcquisitionTime": pd.to_datetime})
    df["bang_end"] = df.apply(lambda row: pa.detect_main_bang_v2(row["raw_data"], row["samplingFrequency"]), axis=1)
    df["wavefront"] = df.apply(lambda row: pa.wavefront_empty_and_full_detection(row["raw_data"], 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), axis=1)
    df["quality"] = df.apply(lambda row: pa.signal_quality(row["raw_data"])[4], axis=1)

    df_q = df.loc[df.groupby("batchId")["quality"].idxmax()].sort_values(by=["AcquisitionTime"]).reset_index()

    plt.plot(df_q["AcquisitionTime"], df_q["wavefront"])
    plt.show()

    for i, id in enumerate(df["batchId"].unique()):
        if i in [20]:
            batch = df[df["batchId"] == id].reset_index()
            display_batch(batch)


if __name__ == "__main__":
    main()
