import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import tools.raw_processing as rp
from scipy.signal import chirp, hilbert
from scipy import signal

if __name__ == "__main__":
    freq = 1 / 2e-6
    freq2 = freq / 2
    main_freq = 27500

    data = pd.read_csv("data/test/v3_examples/2023_02_06_144504_V3.csv")
    data["ultrasons_data"] = data["ultrasons_data"] - np.mean(data["ultrasons_data"])
    sig = data["ultrasons_data"].values[2500:]

    # TODO: ask why this window?
    window = signal.windows.tukey(len(sig), 0.2)  # catalys value: 0.2
    window_sig = window * sig
    bandpass_freqs = [(main_freq - 2000) / freq2, (main_freq + 2000) / freq2]
    b, a = signal.butter(2, bandpass_freqs, 'bandpass', analog=False)
    bandpass_sig = signal.lfilter(b, a, window_sig)
    b, a = signal.butter(4, 5000 / freq2, 'low')
    lowpass_sig = signal.lfilter(b, a, abs(bandpass_sig))

    analytic_signal = hilbert(bandpass_sig)
    hilbert_sig = np.abs(analytic_signal)

    b, a = signal.butter(1, 1000 / freq2, "low")
    v2_lowpass_sig = signal.lfilter(b, a, abs(window_sig))
    smoothed_v2_sig = signal.savgol_filter(v2_lowpass_sig, 101, 2)

    smoothed_v2_sig = rp.process_raw_signal(sig)
    plt.plot(sig, "gray")
    plt.plot(smoothed_v2_sig, "--", color="orange", linewidth=2, label="smoothed v2")
    plt.plot([int(x) for x in smoothed_v2_sig])
    plt.show()

    plt.subplot(5, 1, 1)
    plt.plot(sig, label="original signal")
    plt.legend(loc="best")

    plt.subplot(5, 1, 2)
    plt.plot(window_sig, label="truncated signal")
    plt.legend(loc="best")

    plt.subplot(5, 1, 3)
    plt.plot(bandpass_sig, label="bandpass filtered signal")
    plt.plot(lowpass_sig, "y--", linewidth=2, label="original envelope")
    plt.plot(hilbert_sig, "r--", linewidth=2, label="hilbert envelope")
    plt.legend(loc="best")

    plt.subplot(5, 1, 4)
    plt.plot(abs(window_sig))
    #plt.plot(v2_lowpass_sig, label="v2 catalys filter")
    plt.plot(smoothed_v2_sig, "--", label="smoothed v2")
    plt.legend(loc="best")

    plt.subplot(5, 1, 5)
    plt.plot((sig - np.mean(sig)) / np.max(sig - np.mean(sig)), label="original signal")
    plt.plot(lowpass_sig / np.max(hilbert_sig), "y--", linewidth=2, label="original envelope")
    plt.plot(hilbert_sig / np.max(hilbert_sig), "r--", linewidth=2, label="hilbert envelope")
    plt.plot(v2_lowpass_sig / np.max(v2_lowpass_sig), "g--", linewidth=2, label="v2 lowpass catalys")
    plt.legend(loc="best")
    plt.show()

    N = len(sig)
    T = freq2
    yf = np.fft.fft(abs(sig))[: N//2]
    xf = np.fft.fftfreq(N)[: N//2]

    threshold = 0.5 * max(abs(yf))
    mask = abs(yf) > threshold
    peaks = xf[mask] * freq
    print(peaks)

    plt.plot(xf * freq, abs(yf))
    plt.axvline(main_freq, color="red")
    plt.show()

    exit()

    envelope = se.signal_envelope(sig)

    plt.plot(sig)
    plt.plot(envelope)
    plt.show()

    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))
    print(signal.shape)

    plt.plot(signal)
    plt.plot(np.abs(hilbert(signal)), "r")
    plt.show()
