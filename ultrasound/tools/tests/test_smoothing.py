
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import tools.regression_smoothing as rs


def exp_filter(prev_filtered_value, current_value, a=0.9):
    # a between 0.8 and 0.99
    return prev_filtered_value + (1 - a) * (current_value - prev_filtered_value)


def nonlin_exp_filter(prev_filtered_value, current_value, R=3.5):
    # R between 3 and 5
    delta_xn = current_value - prev_filtered_value
    f = min(1, abs(delta_xn / R))
    return prev_filtered_value + f * delta_xn


def spike_filter(prev_filtered_value, current_value, count, maximum_change, number_of_changes=2):
    current_filtered_value = 0

    if abs(current_value - prev_filtered_value) > maximum_change and count < number_of_changes:
        count += 1
        current_filtered_value = prev_filtered_value
    else:
        count = 0
        current_filtered_value = current_value

    return current_filtered_value, count


def savgol_filter():
    # TODO
    return


if __name__ == "__main__":
    data_path = "data/test/test2.csv"

    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime})
    data = data[data["weightAlgo1_t"].notnull()]
    data["weightAlgo1_t"] = data.apply(lambda x: float(x["weightAlgo1_t"].split(" ")[0]), axis=1)
    data["smoothed"] = data.apply(lambda x: float(x["smoothed"].split(" ")[0]), axis=1)
    data = data.resample("1H", on="AcquisitionTime").mean()
    data["AcquisitionTime"] = data.index
    data = data[data["weightAlgo1_t"].notnull()]

    print(data.columns)
    print(data)
    print(data["AcquisitionTime"].diff().mean())

    print(len(data))

    sv = np.zeros(len(data))
    sv_exp = np.zeros(len(data))
    sv_nonlin_exp = np.zeros(len(data))
    sv_spike = np.zeros(len(data))
    spike_fil_count = 0

    for i in range(len(data)):
        current_val = data["weightAlgo1_t"].values[i]
        if i > 0:
            prev_val = data["weightAlgo1_t"].values[i - 1]
            exp_prev_filtered_val = sv_exp[i - 1]
            exp_nonlin_prev_filtered_val = sv_nonlin_exp[i - 1]
            spike_prev_filtered_val = sv_spike[i - 1]

            if current_val - prev_val > 2:
                exp_prev_filtered_val = current_val
                exp_nonlin_prev_filtered_val = current_val
        else:
            exp_prev_filtered_val = data["weightAlgo1_t"].values[i]
            exp_nonlin_prev_filtered_val = data["weightAlgo1_t"].values[i]
            spike_prev_filtered_val = data["weightAlgo1_t"].values[i]

        sv_exp[i] = exp_filter(exp_prev_filtered_val, current_val, a=0.8)
        sv_nonlin_exp[i] = max(0, nonlin_exp_filter(exp_nonlin_prev_filtered_val, current_val, R=3.5))
        sv_spike[i], spike_fil_count = spike_filter(spike_prev_filtered_val, current_val, spike_fil_count, 1, 2)

        if i > 48:
            sv[i] = rs.auto_regression_smoothing(data["AcquisitionTime"].values[i-48:i], data["weightAlgo1_t"].values[i-48:i])
        else:
            sv[i] = 0
    plt.plot(data["AcquisitionTime"], data["weightAlgo1_t"], '.')
    plt.plot(data["AcquisitionTime"], sv, label="regression")
    plt.plot(data["AcquisitionTime"], sv_exp, label="exp filter")
    plt.plot(data["AcquisitionTime"], sv_nonlin_exp, label="nonlin exp filter")
    plt.legend(loc="best")
    plt.show()

    plt.plot(data["AcquisitionTime"], data["weightAlgo1_t"], '.')
    plt.plot(data["AcquisitionTime"], sv_spike, label="spike filter")
    plt.legend(loc="best")
    plt.show()
