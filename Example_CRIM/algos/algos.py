import math
import numpy as np
import statistics


# La taille de la fenêtre dans laquelle regarder autour du max du signal en mètre: par example
# 1 veux dire de regarder pour le wavefront à 1 mètre devant le max du peak, et donc tout ce
# qui sera à plus d'1 mètre avant le max sera ignoré (i.e. un plus petit pic)
WAVEFRONT_WINDOW_SIZE_METERS = 1

# Proportion de la hauteur du max qui correspond au wavefront: par example 0.5 veux dire que le
# wavefront est la valeur à mi-hauteur du pic
WAVEFRONT_THRESHOLD = 0.5

# La valeur pour laquelle si la somme du signal après le fond du silo est supérieure, le silo est considéré
# vide. La valeur par défaut est de 25 (empirique), donc si on veut que le silo soit considéré vide avec
# moins de signal après le "bin max" (donc plus facilement), on réduit la valeur (à 20 par exemple), experimental
# vice versa. [valeur d'origine : 25]
WAVEFRONT_EMPTY_DETECTION_SIGNAL_AFTER_BIN_MAX_THRESHOLD = 25

# Pour la détection de silo plein, on regarde la quantité de signal total qui se trouve avant et après la fin du bang,
# et on en prend le ratio : (signal après bang) / (signal avant bang). Si le ratio est supérieur à cette valeur de 0.18
# par défaut (donc qu'il y a beaucoup plus de signal avant la fin du bang que après), on considère que le silo est plein.
# Pour faire en sorte que le silo soit plus facilement considéré plein, augmenter cette valeur, et vice versa.
WAVEFRONT_FULL_DETECTION_SIGNAL_RATIO = 0.05  # 0.18 -> if we remove len()

# Facteur qui détermine en dessous de quel seuil le signal doit tomber pour être considéré comme la fin du bang. Pour
# trouver une fin de bang plus loin augmenter la valeur (à 3 ou 4 par exemple), et pour trouver une fin de bang plus tôt
# diminer la valeur (à 1 ou 0.5 par exemple).
BANG_END_THRESHOLD_FACTOR = 2
BANG_END_THRESHOLD_FACTOR_MIN = 5


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


def detect_main_bang_v2(data, sample_rate, window_size=2000, max_bang_len=20000):
    # data = data[:-3]
    conversion_factor = sample_rate / 1e6
    window_size = int(conversion_factor * window_size)
    max_bang_len = conversion_factor * max_bang_len
    step = window_size // 10

    current_threshold = max((sum(data) / len(data)) / BANG_END_THRESHOLD_FACTOR, BANG_END_THRESHOLD_FACTOR_MIN)
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


def is_silo_full(data, bang_end, ratio=0.05):  # 0.18 if we remove -> / len()
    signal_before_bang = sum(data[:bang_end]) / len(data[:bang_end])
    signal_after_bang = sum(data[bang_end:]) / len(data[bang_end:])
    return signal_after_bang / signal_before_bang < ratio or bang_end > 200


def detect_full_silo_bang_end(data, current_bang_end):
    bang_start_index = next(i for i, d in enumerate(data) if d > 200 or d >= max(data[:current_bang_end]))
    neg_data = abs(np.array(data[bang_start_index:current_bang_end]) - 255)
    peaks, proms = custom_find_peaks(neg_data)
    if len(peaks) == 0:
        return np.argmax(neg_data), 0, 127
    max_prom_index = np.argmax(proms)
    new_bang_end = peaks[max_prom_index] + bang_start_index
    return new_bang_end, bang_start_index, max(proms)


def wavefront_empty_and_full_detection(data, threshold, pulse_count, sample_rate, max_bin_index, bang_end=None, temperature=0, window_in_meters=100):
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
    if signal_after_max_bin >= WAVEFRONT_EMPTY_DETECTION_SIGNAL_AFTER_BIN_MAX_THRESHOLD or max_index >= max_bin_index:
        return max_bin_index

    # Full silo detection
    if is_silo_full(data, bang_end):
        new_bang_end, bang_start_index, _ = detect_full_silo_bang_end(data, bang_end)
        wf = wavefront(data[bang_start_index:bang_end], temperature, threshold, window_in_meters, pulse_count, sample_rate, bang_end=new_bang_end - bang_start_index)
        return wf

    return wavefront(data, temperature, threshold, window_in_meters, pulse_count, sample_rate)


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


def signal_quality(data, sampling_rate=20000):
    data = data[:-3]
    bang_end = detect_main_bang_v2(data, sampling_rate)
    conv = sampling_rate / 1e6
    stats = {}
    stats["max"] = round(int(max(data[bang_end:])), 2)
    normalized_data = [d / stats["max"] for d in data[bang_end:]]
    stats["mean"] = sum(normalized_data) / len(normalized_data)
    stats["area"] = sum(normalized_data) * conv  # normalized area under curve
    stats["stdev"] = statistics.pstdev(normalized_data)

    stats["quality"] = ((1 / 25.5) * stats["max"] - stats["area"] * 5 - (max((bang_end / (10000 * conv)) - 0.5, 0) * 20)) + 50
    return stats


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
