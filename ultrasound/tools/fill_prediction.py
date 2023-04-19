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


def find_fills(values, threshold):
    """Find indices of when the silo is filled

    Parameters
    ----------
    values : list[float]
        weight values
    threshold : float
        minimum delta between two values to be considered a fill

    Returns
    -------
    list[int]
        indices of when the islo was filled
    """
    fills = []
    values = np.array(values)
    for i in range(len(values)):
        next_v = values[min(len(values) - 1, i + 1)]
        v = values[i]
        prev_v = values[max(0, i - 1)]
        if v - prev_v > threshold:
            fills.append(i)
    return fills


def predict_next_fill(time, values, min_regression_len=15, max_regression_len=72, min_r2_score=0.8, max_prediction_distance=200):
    """Estimate when the silo will need to be filled

    Parameters
    ----------
    time : list[datetime]
        AcquisitionTime data
    values : list[float]
        weight values of the silo
    min_regression_len : int, optional
        minimum number of data for a regression to be done, by default 15
    max_regression_len : int, optional
        maximum number of previous data to use for a regression, by default 72
    min_r2_score: float, optional
        minimum score for the regression to be accepted
    max_prediction_distance: int, optional
        maximum number of hours to predict in the future

    Returns
    -------
    datetime
        the estimated prediction date of the next fill
    """

    if len(values) < min_regression_len:
        # No prediction, not enough data points
        return np.datetime64("nat")

    values = np.array(values)
    time = np.array(time, dtype=np.datetime64)

    start_time, end_time = time[0], time[-1]

    time = time.astype(np.float64)
    time = (time - np.min(time)) / (np.max(time) - np.min(time))
    time_step = time[1] - time[0]

    # Sort arrays according to time
    sort_indices = time.argsort()
    time = time[sort_indices]
    values = values[sort_indices]

    # Remove nan and inf
    valid_indices = np.argwhere(np.isfinite(values)).flatten()
    time = time[valid_indices]
    values = values[valid_indices]

    # Find indices when silo was filled
    fills = find_fills(values, 3)
    fills.insert(0, 0)

    # Values of the current fill
    current_fill_values = values[fills[-1]:]
    current_fill_time = time[fills[-1]:]

    # Not enough data in current fill,
    if len(current_fill_values) < min_regression_len:
        if len(fills) > 1:
            # Going backwards, find first fill that contains more than N data points
            found = False
            for i in range(len(fills) - 1):
                prev_values = values[fills[-i - 2]:fills[-i - 1]]
                prev_time = time[fills[-i - 2]:fills[-i - 1]]

                if len(prev_values) >= min_regression_len:
                    found = True
                    break
            if found is False:
                # No prediction, did not found previous fill with enough data
                return np.datetime64("nat")

            y_reg, r2, m, c = linear_regression(prev_time, prev_values)

            # Extrapolate to find when the regression crosses the zero
            next_xs = []
            next_ys = []
            for i in range(max_prediction_distance):
                next_x = (time[-1] + i * time_step)
                fill_level = m * next_x + c + values[-1]
                next_xs.append(next_x)
                next_ys.append(fill_level)
                if fill_level < 0:
                    break
            else:
                # No prediction, the linear regression did not go to zero on previous fill, probably flat or uphill
                return np.datetime64("nat")

            next_fill_hours = int((next_xs[-1] - 1) / time_step)
            next_fill_date = end_time + np.timedelta64(next_fill_hours, "h")
            return next_fill_date

        else:
            # No prediction, there is no previous fill, and not enough data in current fill
            return np.datetime64("nat")

    best_r2 = -100
    best_time = []
    best_values = []
    best_values_reg = []
    best_params = ()

    # Automatically find best data to use for regression in the previous values
    for i in range(max(0, len(current_fill_values) - max_regression_len), len(values) - 5, 2):
        x_slice = current_fill_time[i:]
        y_slice = current_fill_values[i:]

        y_reg, r2, m, c = linear_regression(x_slice, y_slice)
        if r2 > best_r2:
            best_r2 = r2
            best_time = x_slice
            best_values = y_slice
            best_values_reg = y_reg
            best_params = (m, c)
        if best_r2 > min_r2_score:
            break

    if not best_params:
        # No prediction, linear regression did not work
        return np.datetime64("nat")

    # Extrapolate to find when the regression crosses the zero
    next_xs = []
    next_ys = []
    fill_level = np.inf
    i = 0
    while fill_level > 0:
        next_x = 1 + i * time_step
        fill_level = best_params[0] * next_x + best_params[1]
        next_xs.append(next_x)
        next_ys.append(fill_level)
        i += 1

        if fill_level < 0:
            break

        if i > max_prediction_distance:
            next_xs = []
            next_ys = []
            break

    if len(next_xs) > 0:
        next_fill_hours = int((next_xs[-1] - 1) / time_step)
        next_fill_date = end_time + np.timedelta64(next_fill_hours, "h")
        if next_fill_date < end_time:
            next_fill_date = end_time
    else:
        # No prediction, the linear regression did not go to zero, probably flat or uphill
        return np.datetime64("nat")
    return next_fill_date
