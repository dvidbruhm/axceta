import pandas as pd
from scipy import interpolate
import scipy.signal as signal
from scipy.signal import find_peaks
import math
import numpy as np
from torchmetrics.image import sam
import tools.fill_prediction_agco as fpa
import us.utils as utils


def argmax(x):
    """Find the index of the max in a list of values"""
    return max(range(len(x)), key=lambda i: x[i])


def find_next_minimum(data, start_index):
    """Find the next local minimum in a raw ultrasound signal

    Parameters
    ----------
    data : list or np.ndarray
        raw ultrasound signal
    start_index : int
        index to start the search for the next minimum in data

    Returns
    -------
    (int, int)
        (index of the minimum, value of the minimum)
    """
    current_min_value = data[start_index]
    current_min_index = start_index

    # Iterate on the signal until the signal stops decreasing
    for i in range(start_index, len(data)):
        val = data[i]
        if val < current_min_value:
            current_min_value = val
            current_min_index = i
        elif val > current_min_value:
            break

    return current_min_index, current_min_value


def find_next_minimum_below_threshold(data, start_index, threshold, max_index):
    """Finds the next minimum in a dataset that is below a certain threshold

    Parameters
    ----------
    data : list or np.ndarray
        raw ultrasound signal
    start_index : int
        index to start the search for the next minimum in data
    threshold : int
        the minimum needs to be below this threshold
    max_index : int
        the max value for the minimum index

    Returns
    -------
    int
        index of the minimum
    """
    if start_index <= 0:
        raise Exception("Start index must be > 0.")

    valid = False
    current_index = start_index
    # Continue to find the next minimum until the value of the minimum
    # is below a threshold
    while not valid:
        diff = data[current_index - 1] - data[current_index]
        if diff > 0:
            min_index, min_value = find_next_minimum(data, current_index)
            current_index = min_index
        else:
            if data[current_index] < 10:
                min_index = current_index
                valid = True
            current_index += 1
            continue

        if min_value < threshold:
            valid = True

        current_index += 1
        if current_index > max_index:
            min_index = max_index
            break

    return min_index


def is_signal_close_to_bang(data, main_bang_end_estimation, start_index, threshold):
    """Check if the main peak of the signal is close to the main bang

    Parameters
    ----------
    data : list[int]
        Raw ultrasound signal
    main_bang_end_estimation : int
        Estimated end of the main bang
    start_index : int
        Start to check for the main peak at this index
    threshold : float
        Peak is considered the main peak if above this threshold

    Returns
    -------
    bool
        Is the main peak close to the bang bang?
    """
    data_start = data[main_bang_end_estimation:start_index]
    data_end = data[start_index:]
    start_peak_value = max(data_start)
    end_peak_index, end_peak_value = argmax(data_end), max(data_end)

    if end_peak_value > threshold and end_peak_value > start_peak_value / 2:
        return False
    return True


def is_silo_full(data, bang_end, ratio=0.05):  # 0.18 if we remove -> / len()
    signal_before_bang = sum(data[:bang_end]) / len(data[:bang_end])
    signal_after_bang = sum(data[bang_end:]) / len(data[bang_end:])
    return signal_after_bang / signal_before_bang < ratio or bang_end > 200


def detect_full_silo_bang_end(data, current_bang_end):
    bang_start_index = next(i for i, d in enumerate(data) if d > 200 or d >= max(data))
    neg_data = abs(np.array(data[bang_start_index:current_bang_end]) - 255)
    peaks, proms = custom_find_peaks(neg_data)
    if len(peaks) == 0:
        return np.argmax(neg_data), 0, 127
    max_prom_index = np.argmax(proms)
    new_bang_end = peaks[max_prom_index] + bang_start_index
    return new_bang_end, bang_start_index, max(proms)


def detect_main_bang_v2(data, sample_rate, window_size=2000, max_bang_len=20000):
    # data = data[:-3]
    conversion_factor = sample_rate / 1e6
    window_size = int(conversion_factor * window_size)
    max_bang_len = conversion_factor * max_bang_len
    step = window_size // 10

    current_threshold = max((sum(data) / len(data)) / 2, 5)
    bang_end = -1

    s = 0
    for i in range(window_size, len(data) - 1, step):
        window = data[i - window_size : i + 1]
        s = sum(window)
        l = len(window)
        if (s / l) < current_threshold or i >= max_bang_len:
            bang_end = i
            break

    return bang_end


def detect_main_bang_end(data, pulse_count, sample_rate=500000, max_bang_len=8000) -> int:
    """Automatically detect the end of the main bang in a raw ultrasound reading

    Parameters
    ----------
    data : list or numpy.ndarray
        raw data of the ultrasound
    max_mainbang_index : int, optional
        max possible length of the main bang, by default 6000

    Returns
    -------
    int
        index of the end of the main bang
    """

    conversion_factor = sample_rate / 1e6
    max_bang_len = int(conversion_factor * max_bang_len)

    # Value pairs of experimental values to convert from PulseCount -> approx. of bang end
    # Tuple(PulseCount, Approximation of bang end)
    pulse_count_to_index = [
        (5, int(2200 * conversion_factor)),
        (10, int(2600 * conversion_factor)),
        (20, int(3000 * conversion_factor)),
        (31, int(4000 * conversion_factor)),
    ]

    # Find closest pulse count from dict
    closest_pulse_count = -1
    min_difference = float("inf")
    for pc, i in pulse_count_to_index:
        diff = abs(pulse_count - pc)
        if diff < min_difference:
            closest_pulse_count = pc
            min_difference = diff

    # Approximation of bang index from pulse count
    bang_index = [pc[1] for pc in pulse_count_to_index if pc[0] == closest_pulse_count][0]

    # Find the first minima following the approximation of bang index
    # to find the true end of the bang
    max_value_in_bang = max(data[:max_bang_len])
    # minima has to be less than this threshold to be considered as bang end
    first_min_threshold = 0.7
    first_min_index = find_next_minimum_below_threshold(data, bang_index, first_min_threshold * max_value_in_bang, max_bang_len)

    # Check if the signal is close to main bang or not
    if first_min_index < max_bang_len:
        threshold = max(data[first_min_index:]) * 0.4
        is_signal_close = is_signal_close_to_bang(data, first_min_index, max_bang_len, threshold)
        if not is_signal_close:
            if data[max_bang_len] < threshold:
                return max_bang_len
            else:
                val = data[max_bang_len]
                i = max_bang_len
                while val > threshold:
                    val = data[i]
                    i -= 1
                    if i == 0:
                        return max_bang_len
                return i

    # If the bang end is further than the max bang len, return the max bang len
    return min(first_min_index, max_bang_len)


def get_area_under_curve(data):
    max_value = max(data)
    mean_value = sum(data) / len(data)
    data_no_noise = [0 if d < mean_value else d for d in data]
    area_under_curve = sum(data_no_noise) / max_value
    return area_under_curve


def auto_gain_detection(data, bang_end, sample_rate=500000, signal_range=(0, 255), max_area_under_curve=5000, min_signal_height=0.5):
    """Analyses a raw ultrasound signal to determine if there need to be more or less gain applied

    Parameters
    ----------
    data : list of np.ndarray
        raw ultrasound signal
    data_range : tuple, optional
        min and max data range, by default (0, 255)
    bang_end : int, optional
        index of the end of the bang, by default 2500

    Returns
    -------
    int
        1 -> Gain needs to be a stronger
        0 -> Gain is OK
        -1 -> Gain needs to be a lower
    """
    conversion_factor = sample_rate / 1e6
    max_area_under_curve = conversion_factor * max_area_under_curve

    max_value = max(data[bang_end:])
    mean_value = sum(data[bang_end:]) / len(data[bang_end:])

    # Remove small noise lower than the mean
    data[data < mean_value] = 0

    # Compute the area under the curve
    area_under_curve = sum(data[bang_end:]) / max_value

    # The max of the signal is too low, need more signal
    if max_value < signal_range[1] * min_signal_height:
        return 1

    # The max of the signal is too high, or there is too much signal (the area
    # under the curve is too high), need less signal
    if area_under_curve > max_area_under_curve or max_value == signal_range[1]:
        return -1

    # Signal is ok: 1. max of the signal is ok, and
    #               2. area under the curve is not too high
    return 0


def wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate=500000, bang_end=None):
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
    pulse_count : int
        Pulse count parameter of the PGA
    sample_rate : int, optional
        Sample rate of the signal in Hz, by default 500000

    Returns
    -------
    int
        Index of the wavefront, returns -1 if no wavefront index is found
    """

    # Error handling
    if math.isnan(temperature):
        return -1

    data = data[:-3]
    # Find end of bang
    if not bang_end:
        bang_end = detect_main_bang_v2(data, sample_rate)

    index_cutoff_value = int(0.95 * len(data))

    # Compute the threshold and the speed of sound based on the temperature
    threshold = threshold * max(data[bang_end:])
    sound_speed = 20.02 * math.sqrt(temperature + 273.15)

    # Compute the window to check for the threshold around the max index,
    # which depends on the frequency of the signal
    window_in_samples = int(window_in_meters / sound_speed * sample_rate) * 2

    # Find index of max value
    max_index = argmax(data[bang_end:]) + bang_end

    # Start and end index of the window in which to check for the threshold
    start = max(bang_end, int(max_index - window_in_samples))
    end = int(max_index) + 1

    # Find the index at which the data is at the threshold inside the window
    wf_index = -1
    for i, d in enumerate(data[start:end], start=start):
        if d > threshold:
            wf_index = i
            break

    # Interpolate
    if data[wf_index - 1] < threshold < data[wf_index]:
        diff = data[wf_index] - data[wf_index - 1]
        threshold_diff = threshold - data[wf_index - 1]
        if diff > 0:
            wf_index_interpolated = wf_index + (threshold_diff / diff) - 1
        else:
            wf_index_interpolated = wf_index
    else:
        wf_index_interpolated = wf_index

    return wf_index_interpolated


def wavefront_empty_and_full_detection(data, threshold, pulse_count, sample_rate, max_bin_index, bang_end=None, temperature=0, window_in_meters=100):
    data = data[:-3]
    # Find end of bang
    if not bang_end:
        bang_end = detect_main_bang_v2(data, sample_rate)

    start_index = int(max_bin_index) if max_bin_index <= len(data) - 1 else len(data) - 1
    end_index = int(max_bin_index + (6000 * sample_rate / 1e6))
    if end_index >= len(data) - 1:
        end_index = len(data) - 1
    signal_after_max_bin = sum(data[start_index:end_index]) / max(data[bang_end:])

    # Empty silo detection
    max_index = np.argmax(data[bang_end:]) + bang_end
    if signal_after_max_bin >= 20 or max_index >= max_bin_index:
        return max_bin_index

    # Full silo detection

    if is_silo_full(data, bang_end):
        new_bang_end, bang_start_index, _ = detect_full_silo_bang_end(data, bang_end)
        wf = wavefront(data[bang_start_index:bang_end], temperature, threshold, window_in_meters, pulse_count, sample_rate, bang_end=new_bang_end - bang_start_index)
        return wf

    return wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate, bang_end=bang_end)


def wavefront_empty_detection(data, threshold, pulse_count, sample_rate, max_bin_index, bang_end=None):
    # Find end of bang
    if not bang_end:
        bang_end = detect_main_bang_v2(data, sample_rate)

    start_index = int(max_bin_index) if max_bin_index <= len(data) - 1 else len(data) - 1
    end_index = int(max_bin_index + (6000 * sample_rate / 1e6))
    if end_index >= len(data) - 1:
        end_index = len(data) - 1
    signal_after_max_bin = sum(data[start_index:end_index]) / max(data[bang_end:])

    # Empty silo detection
    max_index = np.argmax(data[bang_end:]) + bang_end
    if signal_after_max_bin >= 25 or max_index >= max_bin_index:
        return max_bin_index

    return wavefront(data, 0, threshold, 20, pulse_count, sample_rate)


def wavefront_with_window(
    signal,
    prev_values,
    prev_times,
    threshold,
    pulse_count,
    temperature,
    silo_data,
    density,
    window_width_meters=1,
    sample_rate=20000,
    max_days_before=2,
    alpha=0,
):
    bang_end = detect_main_bang_v2(signal, sample_rate)
    # signal = np.array(signal)
    # signal[0:bang_end] = 0
    print(prev_values)
    current_normal_wf = wavefront(signal, temperature, threshold, 20, pulse_count, sample_rate)
    if len(prev_values) > 1 and current_normal_wf - prev_values[-1] < -400:
        print("allo")
        return current_normal_wf
    fill_index = 0
    for i in range(len(prev_times) - 1, 0, -1):
        diff = prev_values[i] - prev_values[i - 1]
        if diff < -200:
            fill_index = i
            break
    prev_values = prev_values[fill_index:]
    prev_times = prev_times[fill_index:]
    if (len(prev_values) != len(prev_times)) or (len(prev_values) < 5):
        print("allo2")
        return current_normal_wf

    signal = np.array(signal)
    df = pd.DataFrame(data={"values": prev_values}, index=prev_times)
    df = df[df.index > df.index[-1] - pd.Timedelta(days=max_days_before)]
    mean_consum = fpa.get_mean_consumption(df["values"].values, max_pos_delta=-200)
    next_fill_pred = np.mean(df["values"].values[-3:]) + mean_consum

    # Transform window width in samples
    sound_speed = 20.02 * math.sqrt(temperature + 273.15)
    window_in_samples = int(sample_rate * (window_width_meters / sound_speed))

    # Get predicted next tof
    # dist = utils.weight_to_dist(next_fill_pred, silo_data, density, temperature)
    # predicted_tof = int(sample_rate * (dist / sound_speed))

    predicted_tof = max(0, min(next_fill_pred, len(signal)))

    # start_index = int(predicted_tof - window_in_samples / 2)
    start_index = int(prev_values[-1] - window_in_samples / 2)
    start_index = max(start_index, bang_end)
    end_index = start_index + window_in_samples
    predicted_signal = np.copy(signal[max(0, start_index) : min(end_index, len(signal))])

    # Find the index at which the data is at the threshold inside the window
    abs_threshold = threshold * np.max(predicted_signal)
    wf_index = -1
    for i, d in enumerate(predicted_signal, start=start_index):
        if d > abs_threshold:
            wf_index = i
            break
    else:
        wf_index = end_index

    # import matplotlib.pyplot as plt
    # plt.subplot(2, 1, 1)
    # plt.plot(prev_values)
    # plt.plot([len(prev_values) - 1, len(prev_values)], [prev_values[-1], wf_index], "--")
    # plt.subplot(2, 1, 2)
    # plt.plot(signal)
    # plt.plot(range(start_index, end_index), predicted_signal, ".")
    # plt.axvline(wf_index)
    # plt.axvline(start_index, color="red")
    # plt.axvline(end_index, color="red")
    # plt.show()
    # Interpolate

    if signal[wf_index - 1] < abs_threshold < signal[wf_index]:
        diff = signal[wf_index] - signal[wf_index - 1]
        threshold_diff = abs_threshold - signal[wf_index - 1]
        if diff > 0:
            wf_index_interpolated = wf_index + (threshold_diff / diff) - 1
        else:
            wf_index_interpolated = wf_index
    else:
        wf_index_interpolated = wf_index

    return (wf_index_interpolated * (1 - alpha)) + (predicted_tof * alpha)


def find_width_at_threshold(data, threshold, bang_end, get_total_width=False):
    thresh = np.ones_like(data) * threshold
    diff = thresh - data
    cross = (np.sign(diff * np.roll(diff, 1)) < 1).astype(float)[1:]
    inds = np.where(cross > 0.5)[0]
    if len(inds) % 2 == 1:
        inds = inds[1:]
    if not get_total_width:
        width = 0
        for i in range(0, len(inds), 2):
            width += inds[i + 1] - inds[i]
    else:
        width = inds[-1] - inds[0]
    return width


def wavefront_with_params(data, temperature, threshold, window_in_meters, pulse_count, sample_rate=500000, cutoff_freq=150, no_data_threshold=20):
    conversion_factor = sample_rate / 1e6
    # Compute wavefront
    bang_end = detect_main_bang_end(data, pulse_count, sample_rate)
    wf = lowpass_wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate, cutoff_freq, no_data_threshold)

    # Compute params
    max_value = max(data[bang_end:])
    total_width_25 = find_width_at_threshold(data[bang_end:], max_value * 0.25, bang_end, get_total_width=True)
    total_width_50 = find_width_at_threshold(data[bang_end:], max_value * 0.5, bang_end, get_total_width=True)
    total_width_75 = find_width_at_threshold(data[bang_end:], max_value * 0.75, bang_end, get_total_width=True)
    above_25 = find_width_at_threshold(data[bang_end:], max_value * 0.25, bang_end, get_total_width=False)
    above_50 = find_width_at_threshold(data[bang_end:], max_value * 0.5, bang_end, get_total_width=False)
    above_75 = find_width_at_threshold(data[bang_end:], max_value * 0.75, bang_end, get_total_width=False)
    area_under_curve = get_area_under_curve(data[bang_end:])

    # Assign params to return
    params = {}
    params["max"] = max_value
    params["area_under_curve"] = round(area_under_curve, 2) / conversion_factor
    params["total_width_25"] = total_width_25 / conversion_factor
    params["total_width_50"] = total_width_50 / conversion_factor
    params["total_width_75"] = total_width_75 / conversion_factor
    params["nb_data_above_25"] = above_25 / conversion_factor
    params["nb_data_above_50"] = above_50 / conversion_factor
    params["nb_data_above_75"] = above_75 / conversion_factor

    return wf, params


def lowpass_wavefront(data, temperature, threshold, pulse_count, window_in_meters=5, sample_rate=500000, cutoff_freq=150, no_data_threshold=20):
    """Computes the wavefront on a lowpass of the raw signal, and chooses the best wavefront

    Parameters
    ----------
    data : List[int]
        Raw ultrasound data
    temperature : float
        Temperature of the device in Celsius
    threshold : float ([0 - 1])
        Threshold as a fraction of the max value to find the wavefront, between 0 and 1
    pulse_count : int
        Pulse count parameter of the PGA
    window_in_meters : float
        Window around the max value in which to find the wavefront, in meters
    sample_rate : int, optional
        Sample rate of the signal in Hz, by default 500000
    cutoff_freq : float, optional
        Cutoff frequency of the lowpass filter
    no_data_threshold : float, optional
        Noise threshold at which we consider there is true signal

    Returns
    -------
    int
        Index of the wavefront, returns -1 if no wavefront index is found
    """
    data = data[:-3]

    # Compute the normal wavefront
    wf = wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate=sample_rate)

    # Compute the lowpass filter
    b, a = signal.butter(2, cutoff_freq / (sample_rate / 2), "lowpass", analog=False)
    lowpass = signal.filtfilt(b, a, data)
    lowpass_bang_end = detect_main_bang_end(lowpass, pulse_count, sample_rate=sample_rate)
    lowpass_wf = wavefront(lowpass, temperature, threshold, window_in_meters, pulse_count, sample_rate=sample_rate)

    # Check if there is signal after the main bang and choose the wavefront accordingly
    best_wf = lowpass_wf
    if max(data[lowpass_bang_end:]) < no_data_threshold:
        best_wf = wf
    return best_wf


def before_wavefront(data, temperature, threshold, window_in_meters, pulse_count, threshold_before, sample_rate=500000):
    bang_end = detect_main_bang_end(data, pulse_count, sample_rate)
    wf = wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate)
    threshold_before_abs = threshold_before * max(data[bang_end:])
    i = int(wf)
    current_val = data[i]
    while current_val > threshold_before_abs:
        current_val = data[i]
        i -= 1
    return i


import statistics


def signal_quality(data, sampling_rate=20000):
    data = data[:-3]
    bang_end = detect_main_bang_v2(data, sampling_rate)
    quality_mod = 10
    # if is_silo_full:
    #     bang_end, _, max_prom = detect_full_silo_bang_end(data, bang_end)
    #     quality_mod = (max_prom / 255) * 20
    conv = sampling_rate / 1e6
    stats = {}
    stats["m"] = round(max(data[bang_end:]), 2)
    normalized_data = [d / stats["m"] for d in data[bang_end:]]
    stats["mean"] = sum(normalized_data) / len(normalized_data)
    stats["area"] = sum(normalized_data) * conv  # normalized area under curve
    stats["stdev"] = statistics.pstdev(normalized_data)

    stats["c1"] = round(max(normalized_data) / stats["mean"], 2)
    stats["c2"] = ((1 / 25.5) * stats["m"] - stats["area"] * 5 - (max((bang_end / (10000 * conv)) - 0.5, 0) * 20)) + 40 + quality_mod
    stats["quality"] = stats["c2"]

    return stats


def center_of_mass(data, pulse_count, sample_rate=500000):
    # Find end of bang
    bang_end = detect_main_bang_end(data, pulse_count, sample_rate)

    # Remove noise floor
    mean_value = sum(data[bang_end:]) / len(data[bang_end:])
    mean_value = mean_value if mean_value > 15 else 15
    data = data.copy()
    data[bang_end:] = [0 if d < mean_value else d for d in data[bang_end:]]
    data = [d**2 for d in data]
    indices = list(range(0, len(data)))

    # Compute center of mass
    s = sum(data[bang_end:])
    if s > 0:
        center = sum(x * w for w, x in zip(data[bang_end:], indices[bang_end:])) / sum(data[bang_end:])
    else:
        return len(data)
    return center


def fill_nan(A):
    """
    interpolate to fill nan values
    """
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False, kind="linear")
    B = np.where(np.isfinite(A), A, f(inds))
    return B


def enveloppe(data, pulse_count, sample_rate=500000):
    # Find end of bang
    data = np.array(data)
    bang_end = detect_main_bang_end(data, pulse_count, sample_rate)
    peaks, _ = signal.find_peaks(data, height=0)
    new_data = np.zeros(len(data))
    for i in range(len(data)):
        if i <= bang_end or i in peaks:
            new_data[i] = data[i]
            continue

        new_data[i] = np.nan
    new_data = fill_nan(new_data)
    return new_data


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    from rich import print
    from rich.progress import track
    import tools.smoothing as rs

    print("Loading data...")
    signal = pd.read_csv("data/random/errors1.csv", converters={"AcquisitionTime": pd.to_datetime})
    print("Done.")
    print(signal.columns)
    for i in range(len(signal)):
        raw = json.loads(signal.iloc[i]["raw_data"])
        pc = signal.iloc[i]["pulseCount"]
        lowpass_wavefront(raw, 0, 0.5, pc, sample_rate=20000)
        wf, params = wavefront_with_params(raw, 0, 0.5, 10, 10, sample_rate=20000)
    exit()
    centers = []
    centers_dist = []
    centers_weight = []
    for i in track(
        range(len(signal)),
    ):
        raw_data = json.loads(signal.iloc[i]["rawdata"])
        cdm = center_of_mass(raw_data, 31)
        centers.append(cdm)
        dist_offset = utils.tof_to_dist2(cdm, signal.iloc[i]["temperature"]) * 2000
        centers_dist.append(dist_offset)
        centers_weight.append(dist_to_volume(dist_offset / 1000, "allo", silo_data_17))
        if i in []:
            plt.plot(raw_data)
            plt.axvline(cdm, color="green")
            plt.axvline(signal.iloc[i]["CDM_index"], color="red")
            plt.axvline(signal.iloc[i]["PGA_index"] / 2, color="pink")
            plt.show()
    plt.plot(signal["CDM_index"], label="Current cdm")
    plt.plot(signal["PGA_index"] / 2, label="Current pga")
    plt.plot(centers, label="Computed cdm")
    plt.legend(loc="best")
    plt.show()
    plt.plot(signal["AcquisitionTime"], signal["CDM_distance"] * 2, label="Current cdm")
    plt.plot(signal["AcquisitionTime"], signal["PGA_distance"], label="Current pga")
    plt.plot(signal["AcquisitionTime"], centers_dist, label="Computed cdm")
    plt.legend(loc="best")
    plt.show()
    # plt.plot(data["AcquisitionTime"], (data["CDM_weight"] - 17.8) * 2, label="Current cdm")
    filt = rs.generic_iir_filter(signal["PGA_weight"].values, rs.spike_filter, {"maximum_change_perc": 5, "number_of_changes": 2, "count": 0, "bin_max": 40})
    filt2 = rs.generic_iir_filter(centers_weight, rs.spike_filter, {"maximum_change_perc": 5, "number_of_changes": 2, "count": 0, "bin_max": 40})

    loadcells = pd.read_csv("data/agco/v2/loadcells.csv", converters={"AcquisitionTime": pd.to_datetime})
    lc17 = loadcells[loadcells["LocationName"] == "MPass-10"]

    plt.plot(signal["AcquisitionTime"], filt, label="Current pga")
    plt.plot(signal["AcquisitionTime"], filt2, label="Computed cdm")
    plt.plot(lc17["AcquisitionTime"], lc17["LoadCellWeight_t"], label="loadcell")
    plt.legend(loc="best")
    plt.show()
    exit()

    signal = pd.read_csv("data/random/wrong_test.csv")
    raw_data = json.loads(signal["Data"][0])
    pc = signal["PulseCount"][0]
    sf = signal["SamplingFrequency"][0]
    temp = signal["Temperature"][0]

    plt.plot(raw_data)
    plt.show()
    print(pc, sf, temp)
    print(len(raw_data))

    bang_end = detect_main_bang_end(raw_data, pc, sf)
    wf = wavefront(raw_data, temp, 0.5, 1.5, pc, sf)
    print(bang_end, wf)

    plt.plot(raw_data)
    plt.axvline(wf)
    plt.axvline(bang_end)
    plt.show()
    exit()

    signal = pd.read_csv("data/random/ultrasound_echoes.csv")
    print(signal.columns)
    for i in range(len(signal)):
        print(signal.loc[i, "PulseCount"])
        raw_data = np.array(json.loads(signal.loc[i, "Data"]))
        bang_end = detect_main_bang_end(raw_data.copy(), signal.loc[i, "PulseCount"], signal.loc[i, "SamplingFrequency"])
        auto_gain = auto_gain_detection(raw_data.copy(), bang_end, signal.loc[i, "SamplingFrequency"])
        print(auto_gain)
        plt.subplot(2, 1, 1)
        plt.plot(raw_data)
        plt.axvline(signal.loc[i, "MainBangEnd"], color="red")
        plt.axvline(signal.loc[i, "WavefrontIndex"], color="green")

        raw_data_2 = np.array(json.loads(signal.loc[i, "Data"]))[::2]
        bang_end_2 = detect_main_bang_end(raw_data_2.copy(), signal.loc[i, "PulseCount"], signal.loc[i, "SamplingFrequency"] / 2)
        auto_gain_2 = auto_gain_detection(raw_data_2.copy(), bang_end_2, signal.loc[i, "SamplingFrequency"] / 2)

        print(auto_gain_2)
        plt.subplot(2, 1, 2)
        plt.plot(raw_data_2)
        plt.axvline(bang_end_2, color="red")
        plt.show()

    exit()
    wf = wavefront(signal["raw_data"].values, 0, 0.5, 1.5, 31, 500000)
    print(wf)

    plt.plot(signal["raw_data"].values)
    plt.axvline(wf)
    plt.show()
    exit()

    signal = np.genfromtxt("data/downsample_tests/test_input_pulsecount_31.csv", delimiter=",", skip_header=1)
    bang_end = detect_main_bang_end(signal, 20)
    auto_gain = auto_gain_detection(signal, bang_end, signal_range=(0, 255))
    wf = wavefront(signal, 0, 0.5, 1.0, 31, 500000)

    print(f"End of main bang -> {bang_end:5d}")
    print(f"Wavefront index -> {wf:5d}")
    print(f"Auto gain value  -> {auto_gain:5d}")
