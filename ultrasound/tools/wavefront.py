import math


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
        Index of the wavefront, returns -1 if no wavefront index is found
    """
    def argmax(x):
        return max(range(len(x)), key=lambda i: x[i])

    # Index at which to cut the main bang
    bang_index = 2500

    # Compute the threshold and the speed of sound based on the temperature
    threshold = threshold * max(data[bang_index:])
    sound_speed = 20.02 * math.sqrt(temperature + 273.15)

    # Compute the window to check for the threshold around the max index,
    # which depends on the frequency of the signal
    window_in_samples = int(window_in_meters / sound_speed * freq) * 2

    # Find index of max value
    max_index = argmax(data[bang_index:]) + bang_index

    # Start and end index of the window in which to check for the threshold
    start = int(max_index - window_in_samples / 2)
    end = int(max_index + window_in_samples / 2)

    # Find the index at which the data is at the threshold inside the window
    wf_index = -1
    for i, d in enumerate(data[start:end], start=start):
        if d > threshold:
            wf_index = i
            break

    return wf_index
