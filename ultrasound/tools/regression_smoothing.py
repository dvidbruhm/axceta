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
    return y_reg, r2


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

    time = time.astype(np.float)
    time = (time - np.min(time)) / (np.max(time) - np.min(time))

    # Sort arrays according to time
    sort_indices = time.argsort()
    time = time[sort_indices]
    values = values[sort_indices]

    best_r2 = -100
    best_time = []
    best_values = []
    best_values_reg = []
    for i in range(len(values) - max_regression_len, len(values) - 5, 3):
        x_slice = time[i:]
        y_slice = values[i:]

        y_reg, r2 = linear_regression(x_slice, y_slice)
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
