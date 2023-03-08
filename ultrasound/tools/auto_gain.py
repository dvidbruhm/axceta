import numpy as np


def auto_gain_detection_v2(data, data_range=(0, 255), bang_end=2500):
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
    data = np.array(data)
    max_value = np.max(data[bang_end:])
    mean_value = np.mean(data[bang_end:])

    # Remove small noise lower than the mean
    filt_data = data.copy()
    filt_data[filt_data < mean_value] = 0

    # Use the normalized signal for the area under curve
    norm_data = filt_data / max_value
    area_under_curve = np.sum(norm_data[bang_end:])

    # The max of the signal is too low, need more signal
    if max_value < data_range[1] * 0.5:
        return 1

    # The max of the signal is too high, or there is too much signal (the area
    # under the curve is too high), need less signal
    if area_under_curve > 2500 or max_value == data_range[1]:
        return -1

    # Signal is ok: 1. max of the signal is ok, and
    #               2. area under the curve is not too high
    return 0


def auto_gain_detection(raw_signal, range=(0, 255)):
    """Analyses a raw ultrasound signal to determine if there need to be more or less gain applied

    Parameters
    ----------
    raw_signal : list[int/float]
        Raw ultrasound signal
    range : tuple, optional
        Max and min values of the signal, by default (0, 255)

    Returns
    -------
    int
        2 -> Gain needs to be a lot stronger
        1 -> Gain needs to be a little stronger
        0 -> Gain is OK
        -1 -> Gain needs to be a little lower
        -2 -> Gain needs to be a lot lower
    """
    max_value = max(raw_signal[3000:])   # remove bang
    if max_value < 0.4 * range[1]:
        # Signal is very low, we need to put a lot more gain
        return 2
    elif max_value < 0.8 * range[1]:
        # Signal is low, we need a little more gain
        return 1
    else:
        is_clipped = [1 if d > range[1] - 1 else 0 for d in raw_signal]
        nb_clipped_data = sum(is_clipped)
        clipped_percentage = (nb_clipped_data / len(raw_signal)) * 100
        print(clipped_percentage)

        if clipped_percentage > 8:  # empirical "magic value"
            # Signal is over saturated, we need to lower gain by a lot
            return -2
        elif clipped_percentage > 2:  # empirical "magic value"
            # Signal is saturated, we need to lower gain
            return -1

        # Signal is OK
        return 0


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob

    files = glob.glob("data/test/auc_tests/*.csv")
    for i, f in enumerate(files, start=1):
        plt.subplot(len(files), 1, i)
        data = pd.read_csv(f)
        raw = data["raw_data"].values
        a, b, filt = auto_gain_detection_v2(raw)
        print(i, a, b)
        plt.plot(filt, '.')
    plt.show()

    print(auto_gain_detection(raw))
