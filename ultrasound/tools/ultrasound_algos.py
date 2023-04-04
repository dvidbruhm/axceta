import math


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


def wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate=500000, max_distance=9):
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
    max_distance : float, optional
        Maximum distance (meters) for which to look for a signal, by default 9

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

    # Find end of bang
    bang_end = detect_main_bang_end(data, pulse_count, sample_rate)

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


if __name__ == "__main__":
    import numpy as np

    signal = np.genfromtxt('data/downsample_tests/test_input_pulsecount_31.csv', delimiter=',', skip_header=1)
    bang_end = detect_main_bang_end(signal, 20)
    auto_gain = auto_gain_detection(signal, bang_end, signal_range=(0, 255))
    wf = wavefront(signal, 0, 0.5, 1.0, 31, 500000)

    print(f"End of main bang -> {bang_end:5d}")
    print(f"Wavefront index -> {wf:5d}")
    print(f"Auto gain value  -> {auto_gain:5d}")
