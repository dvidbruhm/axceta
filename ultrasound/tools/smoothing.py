import numpy as np


def r_squared(y, y_hat):
    """Computes the r squared metric to compare the data with the regression values

    Parameters
    ----------
    y : np.array
        original data values
    y_hat : np.array
        linear regression values

    Returns
    -------
    float
        r squared metric, 1 is perfect fit, 0 is very bad fit
    """
    y_bar = y.mean()
    ss_tot = ((y - y_bar)**2).sum()
    ss_res = ((y - y_hat)**2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
    return r2


def linear_regression(x, y):
    """Performs a linear regresion on a set of data points

    Parameters
    ----------
    x : np.array
        time values
    y : np.array
        actual values of the data points

    Returns
    -------
    (np.array, float)
        (regression values, r squared metric)
    """
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_reg = m * x + c
    r2 = r_squared(y, y_reg)
    return y_reg, r2, m, c


def auto_regression_smoothing(time, values, min_r2_score=0.9, max_regression_len=48, regression_weight=0.9):
    """Live smoothing of a dataset using linear regression

    Parameters
    ----------
    time : np.array or list
        timestamps of each data point
    values : np.array or list
        values of each data points
    min_r2_score : float, optional
        minimum r squared metric to keep the smoothed data point, by default 0.8
    max_regression_len : int, optional
        maximum number of previous data points to use for the linear regression, by default 48
    regression_weight : float, optional
        weight of the regression result to use compared to the original data value, by default 0.9

    Returns
    -------
    float
        smoothed data value
    """
    assert len(values) >= max_regression_len, "There must be more or equal previous data than the maximum regression length."
    assert len(values) == len(time), "There must be the same number of values and time index."

    values = np.array(values)
    time = np.array(time, dtype=np.datetime64)

    time = time.astype(np.float64)
    time = (time - np.min(time)) / (np.max(time) - np.min(time))

    # Sort arrays according to time
    sort_indices = time.argsort()
    time = time[sort_indices]
    values = values[sort_indices]

    best_r2 = -100
    best_time = []
    best_values = []
    best_values_reg = []
    for i in range(len(values) - max_regression_len, len(values) - 5, 2):
        x_slice = time[i:]
        y_slice = values[i:]

        y_reg, r2, _, _ = linear_regression(x_slice, y_slice)
        if r2 > best_r2:
            best_r2 = r2
            best_time = x_slice
            best_values = y_slice
            best_values_reg = y_reg

    smoothed_value = best_values[-1]
    if best_r2 > min_r2_score:
        smoothed_value = best_values_reg[-1] * regression_weight + best_values[-1] * (1 - regression_weight)
    smoothed_value = max(0, smoothed_value)

    return smoothed_value


def exp_filter(prev, current, params):
    # a between 0.8 and 0.99
    # params = {"tau": 0.8, "timestep": 1}
    if params["delta_value"] > params["min_fill_value"]:
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

    params["max_value"] = np.max(data)
    for i in range(1, len(data)):
        params["delta_value"] = data[i] - data[i - 1]
        filtered_data[i], params = filter_func(filtered_data[i - 1], data[i], params)

    return filtered_data


def spike_filter(prev_filtered_value, current_value, params):
    # params = {"maximum_change_perc": 1, "number_of_changes": 2, "count": 0}

    # if params["delta_value"] > 3:
    #    params["count"] = 0
    #    return current_value, params

    current_filtered_value = 0

    change_perc = (abs(current_value - prev_filtered_value) / params["bin_max"]) * 100

    if change_perc > params["maximum_change_perc"] and params["count"] < params["number_of_changes"]:
        params["count"] += 1
        current_filtered_value = prev_filtered_value
    else:
        params["count"] = 0
        current_filtered_value = current_value

    return current_filtered_value, params


def smoothing(times, values, prev_smoothed_value, bin_max, min_r2_score=0.9, max_regression_len=48, regression_weight=0.9, min_fill_value=3,
              exp_filter_tau=2, exp_filter_timestep=1, spike_filter_max_perc=5, spike_filter_num_change=2):

    assert len(values) == len(times), "There must be the same number of values and time index."
    assert len(values) >= 1, "There must be at least 1 value."
    if len(values) == 1:
        return values[0]

    spike_filtered = generic_iir_filter(
        values, spike_filter,
        {"maximum_change_perc": spike_filter_max_perc, "number_of_changes": spike_filter_num_change, "count": 0, "bin_max": bin_max})
    delta_value = spike_filtered[len(spike_filtered) - 1] - spike_filtered[len(spike_filtered) - 2]

    if len(values) < max_regression_len:
        smoothed_value, _ = exp_filter(prev_smoothed_value, spike_filtered[-1], {
            "tau": exp_filter_tau, "timestep": exp_filter_timestep, "min_fill_value": min_fill_value, "delta_value": delta_value})
        reg_smoothed_value = 0
    else:
        if delta_value > min_fill_value:
            reg_smoothed_value = spike_filtered[-1]
        else:
            reg_smoothed_value = auto_regression_smoothing(times, spike_filtered, min_r2_score, max_regression_len, regression_weight)
        smoothed_value, _ = exp_filter(prev_smoothed_value, reg_smoothed_value, {
            "tau": exp_filter_tau, "timestep": exp_filter_timestep, "min_fill_value": min_fill_value, "delta_value": delta_value})

    if smoothed_value - prev_smoothed_value > 0.5 and smoothed_value - prev_smoothed_value < min_fill_value:
        smoothed_value = prev_smoothed_value + 0.1

    return smoothed_value


def smooth_all(times, values, bin_max, min_r2_score=0.9, max_regression_len=48, regression_weight=0.9, min_fill_value=3,
               exp_filter_tau=2, exp_filter_timestep=1, spike_filter_max_perc=5, spike_filter_num_change=2):
    smoothed_values = []
    smoothed_values.append(values[0])
    for i in range(1, len(values)):
        time_slice = times[max(0, i - 48):min(i + 1, len(values) - 1)]
        value_slice = values[max(0, i - 48):min(i + 1, len(values) - 1)]
        smoothed_value = smoothing(time_slice, value_slice, smoothed_values[i - 1], bin_max, min_r2_score, max_regression_len, regression_weight,
                                   min_fill_value, exp_filter_tau, exp_filter_timestep, spike_filter_max_perc, spike_filter_num_change)
        smoothed_values.append(smoothed_value)
    return smoothed_values
