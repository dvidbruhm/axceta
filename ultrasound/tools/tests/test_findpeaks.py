import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math


def custom_find_peaks(data):
    # Find first derivative of signal
    diffs = np.zeros(len(data) - 1, dtype=np.int64)
    for i in range(0, len(data) - 1):
        diffs[i] = data[i + 1] - data[i]

    # Find points where the signal changes inflection
    is_diff_pos = []
    current_is_diff_pos = False
    for d in diffs:
        if d != 0:
            current_is_diff_pos = True if d > 0 else False
            break

    for d in diffs:
        if d > 0:
            current_is_diff_pos = True
        if d < 0:
            current_is_diff_pos = False
        is_diff_pos.append(current_is_diff_pos)

    peaks = []
    for i in range(0, len(is_diff_pos) - 1):
        if not is_diff_pos[i + 1] and is_diff_pos[i]:
            peak_value = data[i + 1]
            j = i + 1
            while data[j] == peak_value:
                j -= 1
            peaks.append(math.ceil((j + i + 1) / 2))

    # Find each peak prominences
    proms = []
    for p in peaks:
        # check left
        peak_value = data[p]
        i = p
        while peak_value >= data[i]:
            i -= 1
            if i < 0:
                i = 0
                break
        left = i

        # check right
        i = p
        while peak_value >= data[i]:
            i += 1
            if i > len(data) - 1:
                i = len(data) - 1
                break
        right = i
        prom = min(peak_value - min(data[left:p]), peak_value - min(data[p:right]))
        proms.append(prom)

    return peaks, proms


import sys

np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv("data/random/1.csv")
raw = df["raw_data"].values[range(0, len(df), 4)]

print(custom_find_peaks(abs(raw - 255)))

peaks, prom = find_peaks(abs(raw - 255), prominence=0)
cpeaks, cproms = custom_find_peaks(abs(raw - 255))

print(peaks == cpeaks)
print(prom["prominences"] == cproms)
print(repr(raw).replace(" ", "").replace("\n", ""), file=open("find_peaktest.txt", "w"))

plt.plot(raw)
plt.plot(peaks, raw[peaks], "o")
plt.plot(cpeaks, raw[cpeaks], "x")
plt.show()
