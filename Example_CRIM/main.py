import matplotlib.pyplot as plt
import pandas as pd
import algos.algos as algos
import algos.utils as utils

# Can be anything in:
# [
#   'Avinor-1483A' 'Avinor-1483B' 'Avinor-1484A' 'Avinor-1484B'
#   'Avinor-1485A' 'Avinor-1485B' 'Avinor-1486A' 'Avinor-1486B'
#   'Avinor-1489A' 'Avinor-1489B'
# ]
SILO_NAME = "Avinor-1483A"


def compute_all_algos(data_file):
    """Function to compute our algos on a sample dataset that contains the data for a silo.

    The useful data for our algo contains:
        - raw_data: the measured ultrasound signal
        - pulseCount: the pulse associated to the config of the device for this specific measure
        - samplingFrequency: the frequency of the raw_data (by default 20 Khz, which means each point in the raw_data represents 0.05 micro second)
    """

    # Read the input sample dataset as a csv file for a specific silo, and convert
    # the time and raw_data to appropriate data types
    df = pd.read_csv(data_file, converters={"AcquisitionTime": pd.to_datetime})
    # Select a single silo to view
    df = df[df["LocationName"] == SILO_NAME]
    df["raw_data"] = df.apply(lambda row: utils.str_raw_data_to_list(row["raw_data"]), axis=1)

    # For each entry in the dataset, compute our algorithms: see docs/algos_description.pdf for more info
    df["bang_end"] = df.apply(lambda row: algos.detect_main_bang_v2(row["raw_data"], row["samplingFrequency"]), axis=1)
    df["wavefront"] = df.apply(lambda row: algos.wavefront_empty_and_full_detection(row["raw_data"], 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), axis=1)
    df["quality"] = df.apply(lambda row: algos.signal_quality(row["raw_data"])["quality"], axis=1)

    # For each "batch" of data, keep only the one with the best computed quality
    # (a batch is a group of measures done with different device configurations, weak to strong ultrasound bang)
    filtered_df = df.loc[df.groupby("batchId")["quality"].idxmax()].sort_values(by=["AcquisitionTime"])
    return df, filtered_df


def visualize_batches(data_file):
    """Function to visualise some batches, the following code is not relevant for the algorithms"""
    df, filtered_df = compute_all_algos(data_file)

    # *********************************************
    # Can change the ids to visualize other batches
    # *********************************************
    batch_ids = df["batchId"].unique()[[800, 900, 1000, 1100]]

    for id in batch_ids:
        batch = df[df["batchId"] == id].reset_index().sort_values(by=["quality"], ascending=False)
        plt.subplot(len(batch) + 1, 1, 1)
        plt.plot(filtered_df["AcquisitionTime"], filtered_df["wavefront"], label="Our sensor")
        plt.plot(filtered_df["AcquisitionTime"], filtered_df["trueWavefront"], label="Truth value")
        plt.axvline(filtered_df[filtered_df["batchId"] == id]["AcquisitionTime"], color="green", linestyle="--", label="Current visualized batch")
        plt.ylabel("Wavefront\nover time")
        plt.legend(loc="upper right")

        for j, i in enumerate(batch.index):
            plt.subplot(len(batch) + 1, 1, j + 2)
            row = batch.loc[i]
            plt.plot(row["raw_data"], color="green", label="Ultrasound")
            plt.axvline(row["wavefront"], color="pink", linestyle="--", label="Wavefront")
            plt.axvline(row["bang_end"], color="orange", linestyle="--", label="End of bang")
            plt.ylabel(f"Config #{row.name}\nQuality: {round(row['quality'], 1)}")
            plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    visualize_batches("data/dataset.csv")
