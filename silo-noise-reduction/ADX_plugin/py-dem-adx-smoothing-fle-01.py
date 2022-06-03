from scipy import optimize

def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    px, py = func(r.x)
    Y_pred = np.interp(X, px, py)
    return Y_pred



def gram_poly(i, m, k, s):
    if k > 0:
        grampoly = (4 * k - 2) / (k * (2 * m - k + 1)) * (i * gram_poly(i, m, k - 1, s) + s * gram_poly(i, m, k - 1, s - 1)) - ((k - 1) * (2 * m + k)) / (k * (2 * m - k + 1)) * gram_poly(i, m, k - 2, s)
        return grampoly
    else:
        if k == 0 and s == 0:
            grampoly = 1
        else:
            grampoly = 0
        return grampoly

def gen_fact(a, b):
    gf = 1
    for j in range(a - b + 1, a + 1):
        gf = gf * j
    return gf

def coefficient(i, t, m, n, s):
    su = 0
    for k in range(0, n + 1):
        su = su + (2 * k + 1) * (gen_fact(2 * m, k) / gen_fact(2 * m + k + 1, k + 1)) * gram_poly(i, m, k, 0) * gram_poly(t, m, k, s)
    return su


def get_coefficients(smoothing, order, window_size, offset):
    # Compute half of window_size (rounded down)
    half_window = window_size // 2

    # Simple checks for input parameters
    assert half_window > 0, \
           "The window size has to be positive and greater than 0."
    assert offset <= half_window, \
           f"Offset parameter can't be higher than window_size / 2. Current offset value: {offset}, current window_size value: {window_size}."
    assert offset >= -half_window, \
           f"Offset parameter can't be lower than window_size / 2. Current offset value: {offset}, current window_size value: {window_size}."
    assert smoothing >= 0, \
           f"Smoothing parameter has to be positive. Current value: {smoothing}"
    assert order > 0, \
           "Order has to be 1 or higher."

    coeffs = []
    for i in range(-half_window, half_window + 1):
        coeff = coefficient(offset, i, half_window, order, smoothing)
        coeffs.append(coeff)
    return coeffs

def remove_spikes(values: np.ndarray, spike_size_percent: float):

    for i in range(1, len(values) - 1):
        prev = values[i-1]
        current = values[i]
        next_val = values[i+1]

        if abs(current - prev) > (spike_size_percent * prev) and \
           abs(current - next_val) > (spike_size_percent * prev):
            values[i] = prev

    # First and last values:
    if abs(values[0] - values[1]) > (spike_size_percent * max(values[0], values[1])):
        values[0] = values[1]

    if abs(values[-1] - values[-2]) > (spike_size_percent * max(values[-1], values[-2])):
        values[-1] = values[-2]

    return values

def split_fills(times: np.ndarray, values: np.ndarray, threshold_percent: float):
    nb_prev_data = 5
    split_points = [0]
    for i in range(nb_prev_data, len(values)):
        prev_data = values[max(i-nb_prev_data, split_points[-1]):i]
        mean_prev = np.mean(prev_data)
        current_data = values[i]

        if abs(current_data - mean_prev) > (threshold_percent * current_data):
            split_points.append(i)

    if len(split_points) > 0 and split_points[-1] != len(values) - 1:
        split_points.append(len(values) - 1)

    values_splits = []
    time_splits = []
    for i in range(1, len(split_points)):
        val_sp = values[split_points[i-1]:min(split_points[i], len(values))]
        time_sp = times[split_points[i-1]:min(split_points[i], len(values))]
        values_splits.append(np.array(val_sp))
        time_splits.append(np.array(time_sp))
    values_splits[-1] = np.append(values_splits[-1], values_splits[-1][-1])
    time_splits[-1] = np.append(time_splits[-1], time_splits[-1][-1])

    return time_splits, values_splits


def savgol(values: np.ndarray, window: int, order: int, offset: int, fixed_window: bool = False):
    """Function that computes a real time version of the Savitsky-Golay filter"""
    # Fix window size if not enough data
    if not fixed_window:
        if len(values) < window:
            window = len(values)
            offset = 2
        if window in [1, 2]:
            return values[-1], window
    else:
        if len(values) < window:
            return values[-1], 0

    # Force window size to be odd
    window = window if window % 2 == 1 else window - 1

    # Compute coefficients of the filter
    coeffs = get_coefficients(0, order, window, int((window - 1) / 2) - offset)

    # Run filter on all data
    filtered_y = np.zeros_like(values)
    for i in range(len(values)):
        filtered_value = 0
        for j, c in enumerate(coeffs):
            filtered_value += c * values[i - (window - j - 1)]
        filtered_y[i] = filtered_value if filtered_value > 0 else 0

    return filtered_y[-1], window


def algo_savgol(values_splits):
    """ Function that computes the sav gol algo on each split and returns
        an array of the same dimension as the input"""
    splits_smoothed_values = []
    splits_nb_points_used = []

    for values in values_splits:
        smoothed_values = np.zeros_like(values)
        nb_points_used = np.zeros_like(values, dtype=np.int32)

        for i in range(len(values)):
            smoothed_values[i], nb_points_used[i] = savgol(values[max(0, i-window+1):i+1], window, order, offset, fixed_window)

        splits_smoothed_values.append(smoothed_values)
        splits_nb_points_used.append(nb_points_used)

    total_smoothed_values = np.concatenate(splits_smoothed_values)
    total_nb_points_used = np.concatenate(splits_nb_points_used)

    return total_smoothed_values, total_nb_points_used


def algo_regressions(time_splits, values_splits):
    """ Function that computes the multiline (piecewise) regressions algo on
        each split and returns an array of the same dimension as the input"""
    splits_smoothed_values = []

    for time, values in zip(time_splits, values_splits):
        # If not enough data in split, do not smooth and return original values
        if len(values) < 5:
            splits_smoothed_values.append(values)
            continue

        # Transform time to float values to pass to regressor fit
        time = time.astype(np.float)
        time = (time - np.min(time)) / (np.max(time) - np.min(time))

        # Fit piecewise regression
        smoothed_values = segments_fit(time, values, 2)

        # Combine regression results with original data
        smoothed_values = (smoothed_values * 4 + values) / 5
        splits_smoothed_values.append(smoothed_values)

        #plt.plot(time, smoothed_values, color="green", linewidth=2)
        #plt.plot(time, values, ".", color="gray")
        #plt.show()

    total_smoothed_values = np.concatenate(splits_smoothed_values)

    return total_smoothed_values


# Reorder table by date to make sure it is always correct
df["AcquisitionTime"] = pd.to_datetime(df["AcquisitionTime"])
df = pd.DataFrame(df.sort_values(by="AcquisitionTime"))

# Get params
result = df
window, order, offset, savgol_weight = kargs["window"], kargs["order"], kargs["offset"], kargs["savgol_weight"]
col_name, fixed_window = kargs["col_to_smooth"], kargs["fixed_window"]
values_to_smooth = df[col_name].values
acquisition_time = df["AcquisitionTime"].values

# Prepare data : remove spikes and split silo fillings
spike_size_percent, fill_percent = 0.05, 0.2
values_to_smooth = remove_spikes(values_to_smooth, spike_size_percent)
time_splits, v_splits = split_fills(acquisition_time, values_to_smooth, fill_percent)
values_splits = []
for split in v_splits:
    values_splits.append(remove_spikes(split, spike_size_percent))

# Run savgol algo on each split
smoothed_values_savgol, nb_points_used = algo_savgol(values_splits)

# Run regression algo on each split
smoothed_values_regression = algo_regressions(time_splits, values_splits)

# Assign new columns for the smoothed data to the result DataFrame as output
result["smoothed_savgol"] = smoothed_values_savgol
result["nb_points_used"] = nb_points_used
result["smoothed_regression"] = smoothed_values_regression
result["smoothed_combined"] = (smoothed_values_regression * (1 - savgol_weight)) + (smoothed_values_savgol * savgol_weight)
