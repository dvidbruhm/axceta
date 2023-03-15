import math
import numpy as np


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


def detect_main_bang_end(data, pulse_count, max_mainbang_index=3000, min_plateau_len=50, min_raw_data_len=10000) -> int:
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

    pulse_count_to_index = [
        (5, 1100),
        (10, 1300),
        (20, 1500),
        (31, 1700)
    ]

    closest_pulse_count = -1
    min_difference = np.inf
    for pc, i in pulse_count_to_index:
        diff = abs(pulse_count - pc)
        if diff < min_difference:
            closest_pulse_count = pc

    bang_index = [pc[1] for pc in pulse_count_to_index if pc[0] == closest_pulse_count][0]

    # Find the first minima following the last plateau
    max_value_in_bang = max(data[:max_mainbang_index])
    first_min_threshold = 0.7
    mins = find_local_minimas(data)
    first_min_index = 0
    for m in mins:
        if m > bang_index and data[m] < first_min_threshold * max_value_in_bang:
            first_min_index = m
            break

    return min(first_min_index, max_mainbang_index)

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
    last_max_index = -1
    for i, max_count in max_counts:
        if max_count > min_plateau_len:
            last_max_index = i

    if last_max_index == -1:
        return max_mainbang_index

    # Find the first minima following the last plateau
    first_min_threshold = 0.7
    mins = find_local_minimas(data)
    first_min_index = 0
    for m in mins:
        if m > last_max_index and data[m] < first_min_threshold * max_value:
            first_min_index = m
            break
    #first_min_index = min(first_min_index, max_mainbang_index)

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
    """


def wavefront(data, temperature, threshold, window_in_meters, pulse_count, bang_end, sample_rate=500000, max_distance=9):
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
    max_distance : float, optional
        Maximum distance for which to look for a signal

    Returns
    -------
    int
        Index of the wavefront, returns -1 if no wavefront index is found
    """
    def argmax(x):
        return max(range(len(x)), key=lambda i: x[i])

    # Error handling
    if math.isnan(temperature):
        return -1

    # Compute the threshold and the speed of sound based on the temperature
    threshold = threshold * max(data[bang_end:])
    sound_speed = 20.02 * math.sqrt(temperature + 273.15)

    # Compute the window to check for the threshold around the max index,
    # which depends on the frequency of the signal
    window_in_samples = int(window_in_meters / sound_speed * sample_rate) * 2
    max_distance_in_samples = int(max_distance / sound_speed * sample_rate) * 2

    # Find index of max value
    max_index = argmax(data[bang_end:max_distance_in_samples]) + bang_end

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
