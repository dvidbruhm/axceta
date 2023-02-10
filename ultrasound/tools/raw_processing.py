from scipy import signal
import numpy as np


def process_raw_signal(data, freq=500000):
    data = np.array(data)
    freq2 = freq / 2

    window = signal.windows.tukey(len(data), 0.2)  # catalys value: 0.2
    window_data = window * data

    b, a = signal.butter(1, 1000 / freq2, "low")
    lowpass_filtered = signal.lfilter(b, a, abs(window_data))
    smoothed = signal.savgol_filter(lowpass_filtered, 101, 2)

    return smoothed
