import numpy as np
import math
import statistics


def argmax(x):
    """Find the index of the max in a list of values"""
    return max(range(len(x)), key=lambda i: x[i])


def detect_main_bang_v2(data, sample_rate, window_size=1000, max_bang_len=10000):
    # data = data[:-3]
    conversion_factor = sample_rate / 1e6
    window_size = int(conversion_factor * window_size)
    max_bang_len = conversion_factor * max_bang_len
    step = window_size // 10

    current_threshold = (sum(data) / len(data)) / 2
    bang_end = -1

    for i in range(0, len(data) - 1 - window_size, step):
        window = data[i : i + 1 + window_size]
        s = sum(window)
        l = len(window)
        if (s / l) < current_threshold or i >= max_bang_len - window_size:
            bang_end = i + window_size
            break

    return bang_end


def wavefront(data, threshold, pulse_count, sample_rate=500000, bang_end=None):
    data = data[:-3]
    # Find end of bang
    if not bang_end:
        bang_end = detect_main_bang_v2(data, sample_rate)

    # Compute the threshold and the speed of sound based on the temperature
    threshold = threshold * max(data[bang_end:])

    # Find the index at which the data is at the threshold inside the window
    wf_index = -1
    for i, d in enumerate(data[bang_end:], start=bang_end):
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


def wavefront_empty_and_full_detection(data, threshold, pulse_count, sample_rate, max_bin_index):
    data = data[:-3]

    # Find end of bang
    bang_end = detect_main_bang_v2(data, sample_rate)

    start_index = int(max_bin_index) if max_bin_index <= len(data) - 1 else len(data) - 1
    end_index = int(max_bin_index + (6000 * sample_rate / 1e6))
    if end_index >= len(data) - 1:
        end_index = len(data) - 1
    signal_after_max_bin = sum(data[start_index:end_index]) / max(data[bang_end:])

    # Empty silo detection
    max_index = argmax(data[bang_end:]) + bang_end
    if signal_after_max_bin >= 20 or max_index >= max_bin_index:
        return max_bin_index

    # Full silo detection
    signal_before_bang = sum(data[:bang_end])
    signal_after_bang = sum(data[bang_end:])
    ratio = signal_after_bang / signal_before_bang

    if ratio < 0.18:
        peaks, proms = custom_find_peaks([abs(d - 255) for d in data])
        max_prom_index = np.argmax(proms)
        new_bang_end = peaks[max_prom_index]
        return wavefront(data, threshold, pulse_count, sample_rate, bang_end=new_bang_end)

    return wavefront(data, threshold, pulse_count, sample_rate)


def signal_quality(data, sampling_rate=20000):
    data = data[:-3]
    bang_end = detect_main_bang_v2(data, sampling_rate)
    conv = sampling_rate / 1e6

    stats = {}
    stats["max"] = round(max(data[bang_end:]), 2)
    normalized_data = [d / stats["max"] for d in data[bang_end:]]
    stats["mean"] = sum(normalized_data) / len(normalized_data)
    stats["area"] = sum(normalized_data) * conv  # normalized area under curve
    stats["stdev"] = statistics.pstdev(normalized_data)

    stats["quality"] = ((1 / 25.5) * stats["max"] - stats["area"] * 5) + 50

    return stats["max"], stats["mean"], stats["area"], stats["stdev"], stats["quality"]


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
