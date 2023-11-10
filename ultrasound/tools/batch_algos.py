import matplotlib.pyplot as plt
import numpy as np


def is_silo_empty(batch):
    nb_sample = len(batch)
    tot_data = np.zeros(len(batch[0]["raw_data"]))
    all_data = np.zeros((nb_sample, len(batch[0]["raw_data"])))
    for i in range(len(batch)):
        row = batch[i]
        raw = row["raw_data"]
        tot_data += np.array(raw)
        all_data[i] = np.array(raw)

        signal_after_max_bin = sum(raw[row["maxBinIndex"] :]) / max(raw[row["bang_end"] :])
        print(signal_after_max_bin)

        plt.subplot(len(batch) + 1, 1, i + 1)
        plt.plot(raw)
        plt.axvline(row["maxBinIndex"], color="black")

    max_data = np.max(all_data, axis=0)
    min_data = np.min(all_data, axis=0)
    print(all_data)
    print(all_data.shape)
    print(np.max(all_data, axis=0))
    plt.subplot(len(batch) + 1, 1, len(batch) + 1)
    plt.plot(tot_data / nb_sample)
    plt.plot(max_data)
    plt.plot(min_data)
    plt.show()
