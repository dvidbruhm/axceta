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

def split_fills(values: np.ndarray, threshold_percent: float):
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

    splits = []
    for i in range(1, len(split_points)):
        sp = values[split_points[i-1]:min(split_points[i], len(values))]
        splits.append(np.array(sp))
    splits[-1] = np.append(splits[-1], splits[-1][-1])
    return splits


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


# Reorder table by date to make sure it is always correct
df["AcquisitionTime"] = pd.to_datetime(df["AcquisitionTime"])
df = pd.DataFrame(df.sort_values(by="AcquisitionTime"))

# Get params
result = df
window, order, offset = kargs["window"], kargs["order"], kargs["offset"]
col_name, output_col_name, fixed_window = kargs["col_to_smooth"], kargs["output_col_name"], kargs["fixed_window"]
values_to_smooth = df[col_name].values

# Prepare data : remove spikes and split silo fillings
spike_size_percent, fill_percent = 0.05, 0.2
values_to_smooth = remove_spikes(values_to_smooth, spike_size_percent)
temp_splits = split_fills(values_to_smooth, fill_percent)
splits = []
for split in temp_splits:
    splits.append(remove_spikes(split, spike_size_percent))

# Run algo on each split
splits_smoothed_values = []
splits_nb_points_used = []
for values in splits:
    smoothed_values = np.zeros_like(values)
    nb_points_used = np.zeros_like(values, dtype=np.int32)
    for i in range(len(values)):
        smoothed_values[i], nb_points_used[i] = savgol(values[max(0, i-window+1):i+1], window, order, offset, fixed_window)
    splits_smoothed_values.append(smoothed_values)
    splits_nb_points_used.append(nb_points_used)
smoothed_values = np.concatenate(splits_smoothed_values)
nb_points_used = np.concatenate(splits_nb_points_used)

result[output_col_name] = smoothed_values
result["nb_points_used"] = nb_points_used
