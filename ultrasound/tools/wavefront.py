import math


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
