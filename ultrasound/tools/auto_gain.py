

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
    end_of_min_plateau = 0

    # Iterate on the signal until the signal stops decreasing
    for i in range(start_index, len(data)):
        val = data[i]
        if val < current_min_value:
            current_min_value = val
            current_min_index = i
        elif val > current_min_value:
            end_of_min_plateau = i
            break

    # Find the "middle point" of the minimum plateau
    current_min_index = (end_of_min_plateau + current_min_index) // 2
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
            current_index += 1
            continue

        if min_value < threshold:
            valid = True

        current_index += 1
        if current_index > max_index:
            min_index = max_index
            break

    return min_index


def detect_main_bang_end(data, pulse_count, sample_rate=500000, max_bang_len=6000) -> int:
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
    max_bang_len = conversion_factor * max_bang_len

    # Value pairs of experimental values to convert from PulseCount -> approx. of bang end
    # Tuple(PulseCount, Approximation of bang end)
    pulse_count_to_index = [
        (5, 2200 * conversion_factor),
        (10, 2600 * conversion_factor),
        (20, 3000 * conversion_factor),
        (31, 3400 * conversion_factor)
    ]

    # Find closest pulse count from dict
    closest_pulse_count = -1
    min_difference = float("inf")
    for pc, i in pulse_count_to_index:
        diff = abs(pulse_count - pc)
        if diff < min_difference:
            closest_pulse_count = pc

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
    print(conversion_factor)
    exit()

    max_value = max(data[bang_end:])
    mean_value = sum(data[bang_end:]) / len(data[bang_end:])

    # Remove small noise lower than the mean
    filt_data = data.copy()
    filt_data[filt_data < mean_value] = 0

    # Use the normalized signal for the area under curve
    norm_data = filt_data / max_value
    area_under_curve = sum(norm_data[bang_end:])

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


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import json
    import glob

    all_files = list(glob.glob("data/downsample_tests/*.csv"))[0]
    df_from_each_file = (pd.read_csv(f, converters={"ultrasound": json.loads}) for f in all_files)
    data = pd.concat(df_from_each_file, ignore_index=True)
    DOWNSAMPLE_FACTORS = [1, 5, 10, 20, 50, 100, 150, 200]
    for i in range(len(data)):
        for down_fac in DOWNSAMPLE_FACTORS:
            sample_rate = 500000 / down_fac
            pulse_count = data.loc[i, "pulseCount"]
            signal = np.array(data.loc[i, "ultrasound"])[::down_fac]

            bang_end = detect_main_bang_end(signal, pulse_count, sample_rate)
            auto_gain = auto_gain_detection(signal, bang_end, sample_rate)

            exit()

    exit()
    signal = np.genfromtxt('data/test/auc_tests/test_input_pulsecount_20.csv', delimiter=',', skip_header=1)
    bang_end = detect_main_bang_end(signal, 20)
    auto_gain = auto_gain_detection(signal, bang_end, signal_range=(0, 255))

    import matplotlib.pyplot as plt
    plt.plot(signal)
    plt.axvline(bang_end)
    plt.axvline(1778, color="red")
    plt.show()

    print(f"End of main bang -> {bang_end:5d}")
    print(f"Auto gain value  -> {auto_gain:5d}")
