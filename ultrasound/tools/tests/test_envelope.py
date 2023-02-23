import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import tools.raw_processing as rp
from scipy.signal import chirp, hilbert
from scipy import signal
from scipy.fft import fft, fftfreq


if __name__ == "__main__":
    freq = 1 / 2e-6
    freq2 = freq / 2
    main_freq = 27500

    data = pd.read_csv("data/test/v3_examples/2023_02_06_144504_V3.csv")
    data["ultrasons_data"] = data["ultrasons_data"] - np.mean(data["ultrasons_data"])
    sig = data["ultrasons_data"].values[:]

    # TODO: ask why this window?
    window = signal.windows.tukey(len(sig), 0.05)  # catalys value: 0.2
    window_sig = window * sig
    sig = window_sig
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
    plt.clf()  # plt.show()

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
    plt.clf()  # plt.show()

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
    plt.clf()  # plt.show()

    # TEST N2
    n_plots = 8
    plt.subplot(n_plots, 1, 1)
    plt.plot(sig, label="raw")
    plt.legend()

    plt.subplot(n_plots, 1, 2)
    N = len(sig)
    fft_y = 2.0 / N * np.abs(fft(sig)[:N // 2])
    fft_x = fftfreq(N, 1 / freq)[:N // 2]
    plt.plot(fft_x, fft_y, label="fft")
    plt.legend()

    plt.subplot(n_plots, 1, 3)
    low, high = 20000, 30000
    bandpass_freqs = [low / freq2, high / freq2]
    sos = signal.butter(2, bandpass_freqs, 'bandpass', output="sos", analog=False)
    bandpass_sig = signal.sosfilt(sos, sig)
    plt.plot(bandpass_sig, label="bandpass")
    plt.legend()

    bandpass_sig = np.abs(bandpass_sig)
    plt.subplot(n_plots, 1, 4)
    plt.plot(bandpass_sig, label="rectified")
    plt.legend()

    plt.subplot(n_plots, 1, 5)
    peaks, _ = signal.find_peaks(bandpass_sig)
    peak_hold_sig = np.interp(range(len(bandpass_sig)), peaks, bandpass_sig[peaks])
    plt.plot(peak_hold_sig, label="peak hold")
    plt.legend()

    plt.subplot(n_plots, 1, 6)
    N = len(peak_hold_sig)
    fft_y = 2.0 / N * np.abs(fft(peak_hold_sig)[:N // 2])
    fft_x = fftfreq(N, 1 / freq)[:N // 2]
    plt.plot(fft_x, fft_y, label="fft")
    plt.legend()

    plt.subplot(n_plots, 1, 7)
    cutoff_freq = 2000
    sos = signal.butter(2, cutoff_freq / freq2, 'lowpass', output="sos", analog=False)
    lowpass_sig = signal.sosfilt(sos, peak_hold_sig)
    plt.plot(lowpass_sig, label="lowpass")
    plt.legend()

    plt.subplot(n_plots, 1, 8)
    plt.plot(sig, label="raw sig")
    plt.plot(lowpass_sig, "--", linewidth=2, color="orange", label="final sig")
    plt.plot(smoothed_v2_sig, "--", linewidth=2, color="green", label="current")
    plt.legend()

    plt.show()

    plt.plot(sig, label="raw sig", alpha=0.2, color="gray")
    plt.plot(lowpass_sig, "--", linewidth=2, color="orange", label="final sig")
    plt.plot(rp.process_raw_signal(sig))

    """
    low, high = 300, 350
    bandpass_freqs = [low / freq2, high / freq2]
    sos = signal.butter(2, bandpass_freqs, 'bandstop', output="sos", analog=False)
    bandpass_sig = signal.sosfilt(sos, lowpass_sig)
    plt.plot(bandpass_sig)
    """
    plt.legend()

    plt.show()
