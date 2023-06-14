
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import tools.smoothing as rs


def exp_filter_old(prev_filtered_value, current_value, a=0.9):
    # a between 0.8 and 0.99
    return prev_filtered_value + (1 - a) * (current_value - prev_filtered_value)


def exp_filter(prev, current, params):
    # a between 0.8 and 0.99
    # params = {"tau": 0.8, "timestep": 1}
    if params["delta_value"] > 3:
        return current, params
    a = np.exp(-params["timestep"] / params["tau"])
    filtered = prev + (1 - a) * (current - prev)
    return filtered, params


def nonlin_exp_filter(prev, current, params):
    # R between 3 and 5
    # params = {"R": 3.5}
    if params["delta_value"] > 3:
        return current, params
    delta_xn = current - prev
    f = min(1, abs(delta_xn / params["R"]))
    filtered = prev + f * delta_xn
    return filtered, params


def generic_iir_filter(data, filter_func, params):
    data = np.array(data)
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    filtered_data_2 = np.zeros_like(data)
    filtered_data_2[0] = data[0]

    params["max_value"] = np.max(data)
    for i in range(1, len(data)):
        # if data[i] - data[i - 1] > 2:
        #    filtered_data[i] = data[i]
        # else:
        params["delta_value"] = data[i] - data[i - 1]
        filtered_data[i], params = filter_func(filtered_data[i - 1], data[i], params)

    return filtered_data


def nonlin_exp_filter_old(prev_filtered_value, current_value, R=3.5):
    # R between 3 and 5
    delta_xn = current_value - prev_filtered_value
    f = min(1, abs(delta_xn / R))
    return prev_filtered_value + f * delta_xn


def spike_filter(prev_filtered_value, current_value, params):
    # params = {"maximum_change_perc": 1, "number_of_changes": 2, "count": 0}

    # if params["delta_value"] > 3:
    #    params["count"] = 0
    #    return current_value, params

    current_filtered_value = 0

    change_perc = (abs(current_value - prev_filtered_value) / 25.6) * 100

    if change_perc > params["maximum_change_perc"] and params["count"] < params["number_of_changes"]:
        params["count"] += 1
        current_filtered_value = prev_filtered_value
    else:
        params["count"] = 0
        current_filtered_value = current_value

    return current_filtered_value, params


def spike_filter_old(prev_filtered_value, current_value, count, maximum_change, number_of_changes=2):
    current_filtered_value = 0

    if abs(current_value - prev_filtered_value) > maximum_change and count < number_of_changes:
        count += 1
        current_filtered_value = prev_filtered_value
    else:
        count = 0
        current_filtered_value = current_value

    return current_filtered_value, count


def savgol_filter(data, window_size, ):
    # TODO
    return


if __name__ == "__main__":
    from regressio.models import cubic_spline, linear_regression, exp_moving_average, isotonic_regression, knn_kernel
    import scipy
    from scipy.optimize import curve_fit
    data_path = "data/agco/v2/p2c-17.csv"
    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime})
    data = data[270:500]
    print(data.columns)
    pga = rs.generic_iir_filter(data["PGA_weight"].values, rs.spike_filter, {
                                "maximum_change_perc": 5, "number_of_changes": 2, "count": 0, "bin_max": 40})

    values = np.array(pga)
    time = np.array(data["AcquisitionTime"], dtype=np.datetime64)

    start_time, end_time = time[0], time[-1]

    time = time.astype(np.float64)
    time = (time - np.min(time)) / (np.max(time) - np.min(time))
    time_step = time[1] - time[0]
    # time = np.linspace(0, 10, 101)
    # values = -np.heaviside((time - 5), 0.)

    def sigmoid(x, x0, b, c, d):
        return c * scipy.special.expit((x - x0) * b) + d

    def multi_sigmoid(x, N, *args):
        x0s, a0s, b0s, d = list(args[0][:N]), list(args[0][N:2 * N]), list(args[0][2 * N:3 * N]), args[0][-1]
        s = 0
        for i in range(N):
            s += sigmoid(x, x0s[i], a0s[i], b0s[i], 0)
        s += d
        return s

    plt.plot(time, values)
    from lmfit import Model
    lmodel = Model(multi_sigmoid)
    params = lmodel.make_params(N=2, args=[0.2, 0.5, 30, 50, -1, -1, 5])
    result = lmodel.fit(values, params, x=time)
    print(result.params)
    plt.plot(time, result.best_fit)
    plt.show()
    exit()

    # plt.plot(time, multi_sigmoid(time, 3, [0.2, 0.5, 0.75, 30, 50, 200, -1, -1, -1, 5]))
    # plt.show()

    n = 2
    n_params = n * 3 + 1
    lower_bounds = []
    upper_bounds = []
    params_0 = []
    for i in range(n_params):
        if i < n:
            lower_bounds.append(0)
            upper_bounds.append(1)
            params_0.append((i / n) + (1 / n / 2))
        elif i < 2 * n:
            params_0.append(100)
            lower_bounds.append(50)
            upper_bounds.append(500)
        elif i < 3 * n:
            params_0.append(-3)
            lower_bounds.append(-30)
            upper_bounds.append(0)
        else:
            params_0.append(5)
            lower_bounds.append(0)
            upper_bounds.append(60)
    print(params_0)
    print(lower_bounds)
    print(upper_bounds)
    args, cov = curve_fit(lambda x, *params_0: multi_sigmoid(x, n, params_0), time, values, p0=params_0)
    # args, cov = curve_fit(
    #    multi_sigmoid, time, values,
    #    bounds=(lower_bounds, upper_bounds))
    print([round(elem, 2) for elem in args])
    plt.scatter(time, values)
    plt.plot(time, multi_sigmoid(time, n, *[args]))
    plt.plot(time, multi_sigmoid(time, n, [0.2, 0.5, 100, 100, -2, -3, 5]))
    plt.plot(time, multi_sigmoid(time, n, *[params_0]))
    plt.show()

    exit()

    cubic_splined = knn_kernel(n=10)
    cubic_splined.fit(data["AcquisitionTime"].values.astype(np.float64), pga, confidence_interval=0.99, plot=True)
    cubic_splined2 = knn_kernel(n=10)
    cubic_splined2.fit(data["AcquisitionTime"].values.astype(np.float64), pga, confidence_interval=0.7, plot=True)

    plt.plot(data["AcquisitionTime"], pga)
    plt.show()

    exit()

    data = data[data["latestAlgo_w_t"].notnull()]
    # data["weightAlgo1_t"] = data.apply(lambda x: float(x["weightAlgo1_t"].split(" ")[0]), axis=1)
    # data["smoothed"] = data.apply(lambda x: float(x["smoothed"].split(" ")[0]), axis=1)

    data = data.resample("1H", on="AcquisitionTime").mean(numeric_only=True)
    data["AcquisitionTime"] = data.index
    data = data[data["latestAlgo_w_t"].notnull()]

    plt.plot(data["AcquisitionTime"], data["latestAlgo_w_t"], '.')

    spike_filtered = generic_iir_filter(data["latestAlgo_w_t"].values, spike_filter, params={
        "maximum_change_perc": 5, "number_of_changes": 2, "count": 0})
    # plt.plot(data["AcquisitionTime"], spike_filtered)

    tau, R = 4, 3.5
    exp_filtered = generic_iir_filter(spike_filtered, exp_filter, params={"tau": tau, "timestep": 1})
    # plt.plot(data["AcquisitionTime"], exp_filtered, label=f"exp, tau={tau}")
    nonlin_exp_filtered = generic_iir_filter(spike_filtered, nonlin_exp_filter, params={"R": R})
    plt.plot(data["AcquisitionTime"], nonlin_exp_filtered, label=f"nonlin exp, R={R}")
    plt.legend(loc="best")
    plt.show()

    plt.plot(data["AcquisitionTime"], data["latestAlgo_w_t"], '.')
    for tau in [3]:
        exp_filtered = generic_iir_filter(spike_filtered, exp_filter, params={"tau": tau, "timestep": 1})
        plt.plot(data["AcquisitionTime"], exp_filtered, label=f"exp, tau={tau}")

    for R in [3, 5]:
        nonlin_exp_filtered = generic_iir_filter(spike_filtered, nonlin_exp_filter, params={"R": R})
        plt.plot(data["AcquisitionTime"], nonlin_exp_filtered, label=f"nonlin exp, R={R}")

    plt.legend(loc="best")
    plt.show()

    reg_smooth = np.zeros_like(spike_filtered)
    reg_smooth[0] = spike_filtered[0]
    for i in range(1, len(data["latestAlgo_w_t"].values)):
        if i >= 48:
            reg_smooth[i] = rs.auto_regression_smoothing(data["AcquisitionTime"].values[i - 48: i + 1], spike_filtered[i - 48: i + 1])
        else:
            reg_smooth[i] = spike_filtered[i]
        # reg_smooth[i] = generic_iir_filter(reg_smooth[max(0, i - 48):i], exp_filter, params={"tau": 2, "timestep": 1})[-1]
        # delta = float(spike_filtered[i] - spike_filtered[i-1])
        # reg_smooth[i], params = exp_filter(reg_smooth[i-1], spike_filtered[i], {"tau": 2, "timestep": 1, "delta_value": delta})
    filt = rs.generic_iir_filter(reg_smooth, rs.exp_filter, params={"tau": 2, "timestep": 1, "min_fill_value": 3})
    plt.plot(data["latestAlgo_w_t"].values)
    # plt.plot(data["AcquisitionTime"], reg_smooth, label="regression")
    plt.plot(filt, label="cheated")

    final_smoothed = np.zeros_like(data["latestAlgo_w_t"].values)
    final_smoothed[0] = data["latestAlgo_w_t"].values[0]
    for i in range(1, len(data["latestAlgo_w_t"].values)):
        final_smoothed[i] = rs.smoothing(
            data["AcquisitionTime"].values[max(0, i - 48): i],
            data["latestAlgo_w_t"].values[max(0, i - 48): i],
            final_smoothed[i - 1], 25)
    plt.plot(final_smoothed, label="final")
    plt.legend(loc="best")
    """
    data2 = data["latestAlgo_w_t"].values[0:200]
    time2 = data["AcquisitionTime"].values[0:200]
    spike_filtered2 = generic_iir_filter(data2, spike_filter, params={
        "maximum_change_perc": 5, "number_of_changes": 2, "count": 0})
    reg_smooth = np.zeros_like(data2)
    for i in range(48, len(data2)):
        reg_smooth[i] = rs.auto_regression_smoothing(time2[i - 48:i], spike_filtered2[i - 48:i])
        reg_smooth[i] = generic_iir_filter(reg_smooth[i - 48:i], exp_filter, params={"tau": 2, "timestep": 1})[-1]
    #filt = generic_iir_filter(reg_smooth, exp_filter, params={"tau": 2, "timestep": 1})
    plt.plot(time2, filt, "x")
    """
    plt.show()
