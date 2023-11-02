import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tools.ultrasound_algos as algos
import tools.utils as utils
import us.utils
import tools.smoothing as sm


def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y - y_bar) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
    return r2


def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_reg = m * x + c
    r2 = r_squared(y, y_reg)
    return y_reg, r2, m, c


def run_func_on_prev_data(df, nb_prev_data, func, params):
    pass


def find_last_fill(values, threshold=200, nb_meaned_values=5):
    last_fill = 0
    if len(values) < nb_meaned_values * 2:
        return last_fill

    for i in range(len(values) - 1 - nb_meaned_values, -1 + nb_meaned_values, -1):
        p, n = values[i - nb_meaned_values : i], values[i : i + nb_meaned_values]
        diff = np.mean(p) - np.mean(n)
        if diff >= threshold:
            last_fill = i
            break
    return last_fill


def best_prev_data(filt_df):
    filt_df["wavefront"] = sm.smooth_all(filt_df["AcquisitionTime"].values, filt_df["wavefront"].values, 850, spike_filter_num_change=4)

    f, t = 0, len(filt_df)
    pred_ys = []
    for i in range(f, t):
        current_wf = filt_df["wavefront"].values[i]
        last_fill = find_last_fill(filt_df["wavefront"].values[:i], nb_meaned_values=1, threshold=50)

        prev_data = filt_df.loc[last_fill:i]

        # take n last best quality
        best_qual_data = prev_data[-40:].nlargest(12, "quality").sort_values(by="AcquisitionTime")
        #
        # best_qual_data = prev_data[prev_data["quality"] > 45][-5:]
        if len(best_qual_data) < 5 or best_qual_data["quality"].mean() < 40:
            pred_ys.append(current_wf)
            continue

        current_x = filt_df["AcquisitionTime"].values[i]
        x, y = best_qual_data["AcquisitionTime"].values, best_qual_data["wavefront"].values
        time = x.astype(np.float64)
        normed_time = (time - np.min(time)) / (np.max(time) - np.min(time))
        current_time = (current_x.astype(np.float64) - np.min(time)) / (np.max(time) - np.min(time))

        y_reg, r2, m, c = linear_regression(normed_time, y)
        if m <= 0:
            pred_ys.append(current_wf)
            continue
        y_current = m * current_time + c
        pred_ys.append(y_current)

    plt.plot(filt_df["AcquisitionTime"], filt_df["wavefront"])

    plt.plot(filt_df["AcquisitionTime"].values[f:t], pred_ys, ".")

    fills = []
    for i in range(0, len(filt_df)):
        fills.append(find_last_fill(filt_df["wavefront"].values[:i], nb_meaned_values=1, threshold=50))
    ufills = np.unique(fills)
    print(ufills)
    for i in ufills:
        plt.axvline(filt_df["AcquisitionTime"].values[i], color="green", linestyle="--")
    # plt.axvline(filt_df.loc[last_fill]["AcquisitionTime"], linestyle="--", color="green")
    # plt.plot(prev_data["AcquisitionTime"], prev_data["wavefront"], ".")
    # plt.plot(best_qual_data["AcquisitionTime"], best_qual_data["wavefront"], "x")
    #
    # plt.plot(x, y_reg, linestyle="--")
    # plt.plot([current_x], [m * current_time + c], "o")

    plt.show()
    exit()


def wf(data):
    data = np.array(data)
    bang_end = algos.detect_main_bang_v2(data, 20000)
    m = max(data[bang_end:])
    w = np.argmax(data[bang_end:] >= m) + bang_end
    return w


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
    silo_data = us.utils.get_silo_data("data/silo_data.csv", "Paquette3-8")

    df["raw_data"] = df.apply(lambda row: utils.str_raw_data_to_list(row["raw_data"]), axis=1)

    # For each entry in the dataset, compute our algorithms: see docs/algos_description.pdf for more info
    df["bang_end"] = df.apply(lambda row: algos.detect_main_bang_v2(row["raw_data"], row["samplingFrequency"]), axis=1)
    df["wavefront"] = df.apply(lambda row: algos.wavefront_empty_and_full_detection(row["raw_data"], 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), axis=1)
    df["quality"] = df.apply(lambda row: algos.signal_quality(row["raw_data"])["quality"], axis=1)

    # For each "batch" of data, keep only the one with the best computed quality
    # (a batch is a group of measures done with different device configurations, weak to strong ultrasound bang)
    filtered_df = df.loc[df.groupby("batchId")["quality"].idxmax()].sort_values(by=["AcquisitionTime"]).reset_index()

    # NEW SIMPLE WF
    df["wf"] = df.apply(lambda row: wf(row["raw_data"]), axis=1)
    batch_ids = df["batchId"].unique()
    selected = []
    for j, id in enumerate(batch_ids):
        batch = df[df["batchId"] == id].reset_index()
        if j == 0:
            s_row = batch[batch["quality"] == batch["quality"].max()].squeeze()
            selected.append(s_row)
            continue

        current_sel_row = None
        current_best_amax_distance = np.inf
        for i in range(len(batch)):
            row = batch.loc[i]
            amax = np.argmax(row["raw_data"][row["bang_end"] :]) + row["bang_end"]
            prev_amax = np.argmax(selected[-1]["raw_data"][selected[-1]["bang_end"] :]) + selected[-1]["bang_end"]
            amax_dist = abs(prev_amax - amax)
            if amax_dist < current_best_amax_distance:
                current_sel_row = row
                current_best_amax_distance = amax_dist

        if current_best_amax_distance > 200:
            s_row = batch[batch["quality"] == batch["quality"].max()].squeeze()
            selected.append(s_row)
            continue

        selected.append(current_sel_row)
    selected_df = pd.DataFrame(selected)
    selected_df["wf"] = selected_df.apply(lambda row: wf(row["raw_data"]) if row["maxBinIndex"] > wf(row["raw_data"]) else row["maxBinIndex"], axis=1)
    selected_df["wavefront"] = selected_df["wf"]

    # NEW BASED ON BEST PREV
    best_prev_data(filtered_df)

    return df, filtered_df, selected_df


def visualize_batches(df, filtered_df, sel_df, batches):
    """Function to visualise some batches, the following code is not relevant for the algorithms"""
    batch_ids = df["batchId"].unique()[batches]

    for id in batch_ids:
        batch = df[df["batchId"] == id].reset_index().sort_values(by=["quality"], ascending=False)
        plt.subplot(len(batch) + 1, 1, 1)
        plt.plot(filtered_df["AcquisitionTime"], filtered_df["wavefront"], label="Our sensor")
        plt.plot(sel_df["AcquisitionTime"], sel_df["wf"], label="Simple wavefront")
        # plt.plot(filtered_df["AcquisitionTime"], filtered_df["LC_ToF"] / 50, label="Truth value")
        plt.axvline(filtered_df[filtered_df["batchId"] == id]["AcquisitionTime"], color="green", linestyle="--", label="Current visualized batch")
        plt.ylabel("Wavefront\nover time")
        plt.legend(loc="upper right")

        for j, i in enumerate(batch.index):
            plt.subplot(len(batch) + 1, 1, j + 2)
            row = batch.loc[i]
            plt.plot(row["raw_data"], color="green", label="Ultrasound")
            plt.axvline(row["wavefront"], color="pink", linestyle="--", label="Wavefront")
            plt.axvline(row["wf"], color="purple", linestyle="--", label="New wavefront")
            plt.axvline(row["bang_end"], color="orange", linestyle="--", label="End of bang")
            plt.ylabel(f"Config #{row.name}\nQuality: {round(row['quality'], 1)}")
            plt.legend(loc="upper right")
        plt.show()


def main():
    # df, filt_df, sel_df = compute_all_algos("data/long_series/paquette3-8.csv")
    df, filt_df, sel_df = compute_all_algos("data/long_series/beaudry-1-5.csv")

    # plt.plot(filt_df["AcquisitionTime"], filt_df["wavefront"], label="current algo", alpha=0.5)
    plt.plot(filt_df["WavefrontIndex"], label="cloud algo", alpha=0.3, color="gray")
    # plt.plot(filt_df["AcquisitionTime"], filt_df["wf"], label="simple algo", linestyle="--")
    plt.plot(range(len(sel_df)), sel_df["wf"], label="simple algo2", linestyle="--")
    plt.legend()
    plt.show()

    visualize_batches(df, filt_df, sel_df, [700, 710, 720])


if __name__ == "__main__":
    main()
