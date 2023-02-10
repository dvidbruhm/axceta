import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize

from scipy.signal import find_peaks, find_peaks_cwt


def quality_color(quality):
    if quality < 1.5:
        return "red"
    if quality < 3.5:
        return "orange"
    return "green"


def raw_data_quality(data: np.ndarray, plot: bool = False) -> float:
    """
    Computes a quality index for a raw ultrasound reading

    Parameters
    ----------
    data : np.ndarray
        Raw ultrasound data
    plot : bool
        If true, plot the input ultrasound data to visualise the quality

    Returns
    -------
    float quality:
        Quality index on a 1-5 scale. 5 means good quality, 1 means poor quality.
    """
    min_peak_height = 40

    # Remove main bang
    data = data[3000:]

    maxs, _ = find_peaks(data, height=min_peak_height, width=20)

    if len(maxs) == 0:
        return 1

    max_peak_i = np.argmax(data[maxs])
    maxs_without_biggest = maxs[maxs != maxs[max_peak_i]]

    hist, _ = np.histogram(data[maxs], bins=[0, 80, 120, 160, 200, 256], density=True)
    hist = hist / np.max(hist)

    if plot:
        plt.plot(data, color="blue")
        plt.plot(maxs, data[maxs], "o")
        plt.plot(maxs[max_peak_i], data[maxs][max_peak_i], "x")
        plt.ylim([0, 255])
        print(hist)

    if hist[4] > 0:
        quality = 5
        if hist[4] > 3:
            quality -= 1
    elif hist[3] > 0:
        quality = 4
        if hist[3] > 4:
            quality -= 1
    elif hist[2] > 0:
        quality = 3
        if hist[2] > 5:
            quality -= 1
    elif hist[1] > 0:
        quality = 2
        if hist[1] > 6:
            quality -= 1
    else:
        quality = 1

    return quality


def find_local_minimas(data):
    """Find local minimums in 1D array

    Parameters
    ----------
    data : numpy.ndarray
        1D array of data

    Returns
    -------
    min_indices : numpy.ndarray
        indices of local minimums
    """
    data = np.array(data)
    x = np.r_[data[0] + 1, data, data[-1] + 1]
    ups, = np.where(x[:-1] < x[1:])
    downs, = np.where(x[:-1] > x[1:])
    minend = ups[np.unique(np.searchsorted(ups, downs))]
    minbeg = downs[::-1][np.unique(np.searchsorted(-downs[::-1], -ups[::-1]))][::-1]
    min_indices = ((minbeg + minend) / 2).astype(int)
    return min_indices


def detect_main_bang_end(data, max_mainbang_index=3000, min_plateau_len=500, min_raw_data_len=10000) -> int:
    """Automatically detect the end of the main bang in a raw ultrasound reading

    Parameters
    ----------
    data : list or numpy.ndarray
        raw data of the ultrasound
    max_mainbang_index : int, optional
        max possible length of the main bang, by default 3000
    min_plateau_len : int, optional
        minimum length for which to consider a plateau in the main bang, by default 500
    min_raw_data_len : int, optional
        minimum possible length of the raw ultrasound data to verify if the data is valid, by default 10000

    Returns
    -------
    int
        index of the end of the main bang
    """

    # Return -1 as an error if the raw data is not valid
    if len(data) < min_raw_data_len:
        return -1
    data = np.array(data)

    # Find max value of the bang
    max_value = max(data[:max_mainbang_index])
    threshold = max_value / 2

    # Find all the max "plateaus" in the bang, as there can be multiple
    # and we want to make sure we find the last one to start from there
    # to find the end of the bang (and not find a false end of the bang
    # which could be between two plateaus)
    max_count = 0
    max_counts = []
    for i in range(1, max_mainbang_index):
        d = data[i]
        prev_d = data[i - 1]

        if d > max_value - 1:
            max_count += 1
        else:
            if max_count != 0:
                max_counts.append((i, max_count))
            max_count = 0

    # Find the end of the last max plateau
    last_max_index = 0
    for i, max_count in max_counts:
        if max_count > min_plateau_len:
            last_max_index = i

    # Find the first minima following the last plateau
    mins = find_local_minimas(data)

    first_min_index = 0
    for m in mins:
        if m > last_max_index:
            first_min_index = m
            break

    # Find the first time the data goes under a certain threshold
    # following the last plateau
    threshold_index = max_mainbang_index
    for i in range(last_max_index, max_mainbang_index):
        d = data[i]
        prev_d = data[i - 1]
        if max_count > min_plateau_len:
            if prev_d > d and d < threshold:
                threshold_index = i
                break

    # We are keeping only the index of the first minima following
    # the last max plateau as it seems to be the more robust approach
    bang_index_end = first_min_index
    # bang_index_end = min(threshold_index, first_min_index)

    return bang_index_end


def wavefront(data, temperature, threshold, window_in_meters, freq=500000):
    """Finds the wavefront index in a raw ultrasound data

    Parameters
    ----------
    data : List[int]
        Raw ultrasound data
    temperature : float
        Temperature of the device in Celsius
    threshold : float ([0 - 1])
        Threshold as a fraction of the max value to find the wavefront, between 0 and 1
    window_in_meters : float
        Window around the max value in which to find the wavefront, in meters
    freq : int, optional
        Frequency of the signal in Hz, by default 500000

    Returns
    -------
    int
        Index of the wavefront
    """
    def argmax(x):
        return max(range(len(x)), key=lambda i: x[i])

    # Error handling
    if math.isnan(temperature) or len(data) < 10000:
        return -1

    # Index at which to cut the main bang
    bang_index = detect_main_bang_end(data)

    # Compute the threshold and the speed of sound based on the temperature
    threshold = threshold * max(data[bang_index:])
    sound_speed = 20.02 * math.sqrt(temperature + 273.15)

    # Compute the window to check for the threshold around the max index,
    # which depends on the frequency of the signal
    window_in_samples = int(window_in_meters / sound_speed * freq) * 2

    # Find index of max value
    max_index = argmax(data[bang_index:]) + bang_index

    # Start and end index of the window in which to check for the threshold
    start = max(bang_index, int(max_index - window_in_samples / 2))
    end = int(max_index + window_in_samples / 2)

    # Find the index at which the data is at the threshold inside the window
    wf_index = -1
    for i, d in enumerate(data[start:end], start=start):
        if d > threshold:
            wf_index = i
            break
    if False:
        print(sound_speed)
        print(max_index)
        print(window_in_samples)
        print(start, end)
        print(wf_index)
        plt.plot(data)
        plt.axhline(threshold)
        plt.axvline(start, color="black", linestyle="--")
        plt.axvline(end, color="black", linestyle="--")
        plt.axvline(wf_index, color="green")
        plt.axvline(max_index, color="yellow", linestyle="--")
        plt.show()
    return wf_index


def denoise(data: np.ndarray, threshold: float) -> np.ndarray:
    data[data < np.max(data) * threshold] = 0
    return data


def denoise_abs(data: np.ndarray, threshold: float) -> np.ndarray:
    data[data < threshold] = 0
    return data


def algo_v1(data: np.ndarray, plot: bool = False) -> float:

    data_untouched = data.copy()

    time_delay = 4000
    #data = data / np.max(data[time_delay:])
    data = data[time_delay:]
    min_peak_height = 0.5 * np.max(data)
    denoised = denoise(data.copy(), 0.1)

    """
    if plot:
        pd_data = pd.Series(data)
        noise_threshold = NoiseThresholdv1().process(data)
        main_bang = MainBangDetectorv1().process(pd_data, noise_threshold)

        plt.plot(denoised, color="blue")
        plt.axvline(main_bang["main_bang_start"], color="black", label="Main bang start")
        plt.axvline(main_bang["main_bang_end"], color="black", label="Main bang end")
        plt.axhline(y=min_peak_height, color="black", linestyle="--")
    """

    maxs, _ = find_peaks(denoised, height=min_peak_height, width=20)
    max_peak_i = np.argmax(denoised[maxs])

    if plot:
        plt.plot(denoised, color="blue")
        max_peak = denoised[maxs][max_peak_i]
        plt.plot(maxs[max_peak_i], max_peak, "o")

        plt.plot(maxs, denoised[maxs], "x")

    tof = np.average(maxs, weights=range(len(maxs), 0, -1))

    #tof = denoised.dot(range(len(denoised))) / np.sum(denoised)

    tof_interp = np.interp(tof, [5000, 30000], [-1, 1])
    tof += 2000 * tof_interp

    tof = (tof + time_delay) * 2
    return tof


@dataclass
class Window:
    start_index: int
    width: int


def find_best_window(data: np.ndarray, width: int) -> Window:
    start_index = 2500
    resolution = 50

    highest_sum = 0
    best_window = Window(start_index=start_index, width=width)

    for current_index in range(start_index, len(data) - width, resolution):
        data_window = data[current_index:current_index + width]
        current_sum = data_window.sum()
        if current_sum > highest_sum:
            highest_sum = current_sum
            best_window = Window(start_index=current_index, width=width)

    return best_window


def compute_cdm(data: np.ndarray, start_index: int) -> int:
    nt = NoiseThresholdv1()
    threshold = nt.process(data)
    data = denoise_abs(data.copy(), threshold)
    indices = np.arange(start_index, len(data) + start_index)
    data = data*data
    cdm = 0 if np.sum(data) == 0 else np.average(indices, weights=data)
    return int(cdm)


def compute_cdm_in_window(data: np.ndarray, start_index: int, window_size: int) -> Tuple[int, Window]:
    win = find_best_window(data[start_index:], window_size)
    cdm = compute_cdm(data[win.start_index:win.start_index + win.width], start_index=win.start_index)
    return cdm, win


def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y - y_bar)**2).sum()
    ss_res = ((y - y_hat)**2).sum()
    return 1 - (ss_res / ss_tot)


def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_reg = m * x + c
    r2 = r_squared(y, y_reg)
    return y_reg, r2


def auto_regression_smoothing(time, values, min_r2_score=0.8, max_regression_len=48, regression_weight=0.9):
    assert len(values) >= max_regression_len, "There must be more or equal previous data than the maximum regression length."

    values = np.array(values)
    time = np.array(time)

    time = time.astype(np.float)
    time = (time - np.min(time)) / (np.max(time) - np.min(time))

    best_r2 = 0
    best_time = []
    best_values = []
    best_values_reg = []
    for i in range(len(values) - max_regression_len, len(values) - 5, 3):
        x_slice = time[i:]
        y_slice = values[i:]

        y_reg, r2 = linear_regression(x_slice, y_slice)
        if r2 > best_r2:
            best_r2 = r2
            best_time = x_slice
            best_values = y_slice
            best_values_reg = y_reg

    smoothed_value = best_values[-1]
    if best_r2 > min_r2_score:
        smoothed_value = best_values_reg[-1] * regression_weight + best_values[-1] * (1 - regression_weight)
    smoothed_value = max(0, smoothed_value)

    return smoothed_value


def split_silo_fill(x: np.ndarray, y: np.ndarray, threshold=10):
    smallest_y = np.inf
    split_points = [0]
    for i in range(len(x)):
        current_y = y[i]
        current_x = x[i]

        if current_y < smallest_y:
            smallest_y = current_y

        if current_y > smallest_y + threshold:
            split_points.append(i)
            smallest_y = np.inf
    if len(split_points) > 0 and split_points[-1] != len(x) - 1:
        split_points.append(len(x) - 1)
    xs = []
    ys = []
    for i in range(1, len(split_points)):
        temp_x = x[split_points[i-1]:min(split_points[i], len(x))]
        temp_y = y[split_points[i-1]:min(split_points[i], len(x))]
        xs.append(np.array(temp_x))
        ys.append(np.array(temp_y))
    print(split_points)
    xs[-1] = np.append(xs[-1], xs[-1][-1])
    ys[-1] = np.append(ys[-1], ys[-1][-1])

    #plt.plot(x, y)
    # for xt, yt in zip(xs, ys):
    #    plt.plot(xt, yt)
    #plt.scatter([x[i] for i in split_points], [y[i] for i in split_points])
    # plt.show()

    return xs, ys
