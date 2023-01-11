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
    import json
    data = pd.read_csv("data/FullDatasets/log-1/1485.csv")
    raw = json.loads(data.loc[0, "rawData"])
    data = pd.read_csv("data/raw2.csv")
    raw = list(data["raw_data"].values)
    print(raw)
    print(max(raw))
    plt.plot(raw, '.')
    plt.show()

    print(auto_gain_detection(raw))
