import pandas as pd
import tools.ultrasound_algos as ua
import numpy as np
import matplotlib.pyplot as plt
import tools.utils as utils
import statistics


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



def signal_quality(data):
    # Standard devitation
    # max
    # Area Under Curve
    # mean
    #
    # max / mean?
    # max / Area under Curve?
    # Half peak width of max peak?

    conv = 20000 / 1e6
    stats = {}
    stats["m"] = round(max(data), 2)
    normalized_data = [d / stats["m"] for d in data]
    stats["mean"] = sum(normalized_data) / len(normalized_data)
    stats["area"] = sum(normalized_data) * conv # normalized area under curve
    print(len(normalized_data))
    print(sum(np.ones_like(np.array(normalized_data))) * conv)
    stats["stdev"] = statistics.pstdev(normalized_data)
    stats["peak_width"] = half_max_peak_width(data)

    stats["c1"] = round(max(normalized_data) / stats["mean"], 2)
    stats["c2"] = ((1 / 25.5) * stats["m"] - stats["area"] * 5) + 50

    # if max(data) < 15:
    #     stats["c1"], stats["c2"] = 0, 0

    # print(stats["m"], stats["mean"], stats["area"], stats["stdev"], stats["c2"])
    # plt.plot(data)
    # plt.show()
    return stats



def process_batch(batch):
    wfs = []
    cloud_qualities = []
    qualities = []
    cloud_wfs = []
    bang_ends = []
    stats = []
    after_bin_maxs = []
    for i in range(len(batch)):
        row = batch.iloc[i]
        try:
            int(row["config"])
        except ValueError:
            pass
        else:
            wfs = []
            break
        raw_data = utils.str_raw_data_to_list(row["raw_data"])
        wf = ua.wavefront(raw_data, row["temperature"], 0.5, 20,
                          row["pulseCount"], sample_rate=row["samplingFrequency"])
        wfs.append(wf)
        signal_quality(raw_data[ua.detect_main_bang_v2(raw_data, 15, 1000, 10000, 20000):])

        cloud_quality = row["maxSignal"] / row["signalArea"] * 100
        bang_ends.append(ua.detect_main_bang_v2(raw_data, 15, 1000, 10000, 20000))
        bang_ends.append(ua.detect_main_bang_v2(raw_data, 15, 1000, 10000, 20000))
        cloud_qualities.append(row["WaveformQuality"])
        cloud_wfs.append(row["wavefront"])

        bang_end = ua.detect_main_bang_v2(raw_data, 15, 1000, 10000, 20000)
        stats.append(signal_quality(raw_data[bang_end:]))
        q = stats[-1]["c2"]
        qualities.append(q)

        after_bin = sum(raw_data[row["maxBinIndex"]:]) / len(raw_data[row["maxBinIndex"]:]) * 10
        after_bin_maxs.append(after_bin)

    if not len(wfs):
        return None, None, None

    batch["local_wf"] = wfs
    # batch["bang_end"] = bang_ends
    batch["quality"] = qualities
    batch["stats"] = stats
    best_quality_index = np.argmax(qualities)
    new_best_wf = wfs[best_quality_index]
    best_cloud_index = np.argmax(cloud_qualities)
    cloud_best_wf = cloud_wfs[best_cloud_index]

    return batch, cloud_best_wf, new_best_wf, after_bin_maxs[best_quality_index]


def display_batch(batch):
    sorted_batch = batch.sort_values(by=["quality"], ascending=False)
    for i in range(len(batch)):
        row = sorted_batch.iloc[i]
        raw_data = utils.str_raw_data_to_list(row["raw_data"])
        plt.subplot(len(batch), 1, i + 1)
        plt.plot(raw_data, label="Signal")
        plt.axvline(row["local_wf"], linestyle="--", label="Local wf", color="blue")
        plt.axvline(row["wavefront"], linestyle="--", label="Cloud wf", color="green")
        # plt.axvline(row["bang_end"], label="Bang end", color="red")
        plt.axvline(row["maxBinIndex"], color="black")
        plt.ylabel(f'{row["config"]}')
        plt.text(0.5 * len(raw_data), 0.5 * max(raw_data), row["quality"])
        plt.text(0.5 * len(raw_data), 0.4 * max(raw_data), row["WaveformQuality"])
        plt.text(0.5 * len(raw_data), 0.3 * max(raw_data), row["stats"]["area"])
        plt.legend()
    plt.show()


def process_dataset(file):
    df = pd.read_csv(file)
    #df = utils.select_silo(df, "Avinor-1484A")
    print(df.columns)
    if "latest_Algo" in df.columns:
        df["wavefront"] = df["WavefrontIndex"]

    batch_ids = df["batchId"].unique()
    cloud_wfs = []
    new_wfs = []
    after_bins = []
    #lc = []
    for i, batch_id in enumerate(batch_ids):
        print(f"Batch {i} - {batch_id}")
        batch = df[df["batchId"] == batch_id]
        batch, cloud_wf, new_wf, after_bin = process_batch(batch)
        if cloud_wf and new_wf:
            if i > 0:
                display_batch(batch)
            cloud_wfs.append(cloud_wf)
            new_wfs.append(new_wf)
            after_bins.append(after_bin)
            #lc.append(batch["LC_ToF"].iloc[0] / 50)

    plt.plot(cloud_wfs, label="cloud")
    plt.plot(new_wfs, label="Wavefront")
    # plt.plot(np.array(after_bins), label="After bin max signal")
    # wf_bin = [wf if ab < 100 else max(new_wfs) for wf, ab in zip(new_wfs, after_bins)]
    # plt.plot(wf_bin, label="Wf bin")
    # plt.plot(lc, label="Loadcell wavefront")
    # plt.axhline(300, linestyle="--", color="black")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    d = [64, 0, 0, 0, 23, 87, 152, 202, 236, 223, 146, 147, 175, 200, 216, 222, 221, 216, 217, 224, 234, 245, 254, 255, 255, 255, 255, 250, 230, 195, 157, 126, 103, 104, 122, 137, 134, 109, 85, 109, 151, 190, 221, 243, 255, 255, 255, 255, 255, 255, 249, 223, 191, 163, 139, 124, 113, 98, 86, 72, 54, 43, 49, 64, 76, 78, 77, 73, 81, 92, 99, 96, 81, 54, 44, 55, 73, 88, 96, 95, 85, 69, 52, 37, 32, 33, 37, 41, 42, 42, 40, 40, 41, 43, 45, 46, 45, 44, 43, 42, 42, 41, 41, 40, 40, 39, 37, 34, 31, 29, 28, 27, 27, 27, 27, 28, 29, 31, 32, 33, 34, 34, 35, 35, 35, 35, 35, 34, 33, 32, 31, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 12, 11, 10, 9, 8, 8, 8, 7, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 6, 18, 19, 29, 41, 32, 24, 31, 27, 35, 43, 49, 44, 43, 48, 48, 52, 45, 36, 35, 28, 23, 40, 53, 49, 45, 33, 40, 34, 41, 35, 34, 31, 27, 28, 27, 33, 41, 35, 37, 42, 34, 25, 18, 12, 9, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 1, 1, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2]
    d = d[ua.detect_main_bang_v2(d, 15, 1000, 10000, 20000):]
    print(signal_quality(d))
    process_dataset("data/random/Paquette7-2.csv")
    # import glob
    # for f in glob.glob("data/tests/Paquette*.csv"):
    #     plt.title(f.split("/")[-1])
    #     process_dataset(f)
