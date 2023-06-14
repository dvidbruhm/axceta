import math
import numpy as np


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
        raise "Start index must be > 0."

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

    if end_peak_value > threshold and end_peak_value > start_peak_value:
        return False
    return True


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
        (31, int(4000 * conversion_factor))
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
    first_min_threshold = 0.7  # minima has to be less than this threshold to be considered as bang end
    first_min_index = find_next_minimum_below_threshold(data, bang_index, first_min_threshold * max_value_in_bang, max_bang_len)

    # Check if the signal is close to main bang or not
    if first_min_index < max_bang_len:
        threshold = max(data) * 0.4
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
                return i

    # If the bang end is further than the max bang len, return the max bang len
    return min(first_min_index, max_bang_len)


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


def wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate=500000):
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

    # Find end of bang
    bang_end = detect_main_bang_end(data, pulse_count, sample_rate)

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

    # Compute the normal wavefront
    wf = wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate=sample_rate)

    # Compute the lowpass filter
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, data)
    lowpass_bang_end = detect_main_bang_end(lowpass, pulse_count)
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


import scipy.signal as signal
from scipy import interpolate


def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
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


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    import us.utils as utils
    from rich import print
    from rich.progress import track
    import tools.smoothing as rs
    import timeit

    print("Loading data...")
    data = pd.read_csv("data/dashboard/p2c-8--MPass7.csv", converters={"AcquisitionTime": pd.to_datetime})
    print("Done.")
    print(data.columns)
    for i in track(range(0, len(data), 20)):
        raw = json.loads(data.iloc[i]["rawdata"])
        lowpass_wavefront(raw, 0, 0.5, 31)
    exit()
    centers = []
    centers_dist = []
    centers_weight = []
    for i in track(range(len(data)), ):
        raw_data = json.loads(data.iloc[i]["rawdata"])
        cdm = center_of_mass(raw_data, 31)
        centers.append(cdm)
        dist_offset = utils.tof_to_dist2(cdm, data.iloc[i]["temperature"]) * 2000
        centers_dist.append(dist_offset)
        centers_weight.append(dist_to_volume(dist_offset / 1000, "allo", silo_data_17))
        if i in []:
            plt.plot(raw_data)
            plt.axvline(cdm, color="green")
            plt.axvline(data.iloc[i]["CDM_index"], color="red")
            plt.axvline(data.iloc[i]["PGA_index"] / 2, color="pink")
            plt.show()
    plt.plot(data["CDM_index"], label="Current cdm")
    plt.plot(data["PGA_index"] / 2, label="Current pga")
    plt.plot(centers, label="Computed cdm")
    plt.legend(loc="best")
    plt.show()
    plt.plot(data["AcquisitionTime"], data["CDM_distance"] * 2, label="Current cdm")
    plt.plot(data["AcquisitionTime"], data["PGA_distance"], label="Current pga")
    plt.plot(data["AcquisitionTime"], centers_dist, label="Computed cdm")
    plt.legend(loc="best")
    plt.show()
    # plt.plot(data["AcquisitionTime"], (data["CDM_weight"] - 17.8) * 2, label="Current cdm")
    filt = rs.generic_iir_filter(data["PGA_weight"].values, rs.spike_filter, {
        "maximum_change_perc": 5, "number_of_changes": 2, "count": 0, "bin_max": 40})
    filt2 = rs.generic_iir_filter(centers_weight, rs.spike_filter, {
        "maximum_change_perc": 5, "number_of_changes": 2, "count": 0, "bin_max": 40})

    loadcells = pd.read_csv("data/agco/v2/loadcells.csv", converters={"AcquisitionTime": pd.to_datetime})
    lc17 = loadcells[loadcells["LocationName"] == "MPass-10"]

    plt.plot(data["AcquisitionTime"], filt, label="Current pga")
    plt.plot(data["AcquisitionTime"], filt2, label="Computed cdm")
    plt.plot(lc17["AcquisitionTime"], lc17["LoadCellWeight_t"], label="loadcell")
    plt.legend(loc="best")
    plt.show()
    exit()

    data = pd.read_csv("data/random/wrong_test.csv")
    raw_data = json.loads(data["Data"][0])
    pc = data["PulseCount"][0]
    sf = data["SamplingFrequency"][0]
    temp = data["Temperature"][0]

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

    data = pd.read_csv("data/random/ultrasound_echoes.csv")
    print(data.columns)
    for i in range(len(data)):
        print(data.loc[i, "PulseCount"])
        raw_data = np.array(json.loads(data.loc[i, "Data"]))
        bang_end = detect_main_bang_end(raw_data.copy(), data.loc[i, "PulseCount"], data.loc[i, "SamplingFrequency"])
        auto_gain = auto_gain_detection(raw_data.copy(), bang_end, data.loc[i, "SamplingFrequency"])
        print(auto_gain)
        plt.subplot(2, 1, 1)
        plt.plot(raw_data)
        plt.axvline(data.loc[i, "MainBangEnd"], color="red")
        plt.axvline(data.loc[i, "WavefrontIndex"], color="green")

        raw_data_2 = np.array(json.loads(data.loc[i, "Data"]))[::2]
        bang_end_2 = detect_main_bang_end(raw_data_2.copy(), data.loc[i, "PulseCount"], data.loc[i, "SamplingFrequency"] / 2)
        auto_gain_2 = auto_gain_detection(raw_data_2.copy(), bang_end_2, data.loc[i, "SamplingFrequency"] / 2)

        print(auto_gain_2)
        plt.subplot(2, 1, 2)
        plt.plot(raw_data_2)
        plt.axvline(bang_end_2, color="red")
        plt.show()

    exit()
    wf = wavefront(data["raw_data"].values, 0, 0.5, 1.5, 31, 500000)
    print(wf)

    plt.plot(data["raw_data"].values)
    plt.axvline(wf)
    plt.show()
    exit()

    signal = np.genfromtxt('data/downsample_tests/test_input_pulsecount_31.csv', delimiter=',', skip_header=1)
    bang_end = detect_main_bang_end(signal, 20)
    auto_gain = auto_gain_detection(signal, bang_end, signal_range=(0, 255))
    wf = wavefront(signal, 0, 0.5, 1.0, 31, 500000)

    print(f"End of main bang -> {bang_end:5d}")
    print(f"Wavefront index -> {wf:5d}")
    print(f"Auto gain value  -> {auto_gain:5d}")
