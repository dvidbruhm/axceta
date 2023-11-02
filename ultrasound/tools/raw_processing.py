from scipy import signal
import numpy as np


def process_raw_signal(data, freq=500000):
    data = np.array(data)
    freq2 = freq / 2

    # Window
    window = signal.windows.tukey(len(data), 0.05)
    # data = window * data

    # Bandpass
    low, high = 20000, 30000
    bandpass_freqs = [low / freq2, high / freq2]
    sos = signal.butter(2, bandpass_freqs, 'bandpass', output="sos", analog=False)
    bandpass = signal.sosfilt(sos, data)

    # Rectified
    rectified = np.abs(bandpass)

    # Peak hold
    peaks, _ = signal.find_peaks(bandpass)
    peak_hold = np.interp(range(len(bandpass)), peaks, bandpass[peaks])

    # Low pass
    cutoff_freq = 2000
    sos = signal.butter(2, cutoff_freq / freq2, 'lowpass', output="sos", analog=False)
    lowpass = signal.sosfilt(sos, peak_hold)

    filtered = lowpass

    return filtered


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv("data/random/log-1-lc200-p20-data-2023-04-26 12 13 46.csv")
    raw_data = df["raw_data"].values
    plt.plot(raw_data)
    cutoff_freq = 200
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, raw_data)
    plt.plot(lowpass)
    plt.show()

    N = len(raw_data)
    yf = np.fft.fft(abs(raw_data))[: N // 2]
    xf = np.fft.fftfreq(N)[: N // 2]
    plt.plot(xf * 500000, abs(yf))
    plt.show()
    exit()

    df = pd.read_csv("data/test/2023_02_22_113006_V3.csv")

    processed = process_raw_signal(df["ultrasons_data"].values, 500000)
    plt.plot(processed)
    plt.show()
