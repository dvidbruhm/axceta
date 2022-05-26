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


# Get params
result = df
window, order, offset = kargs["window"], kargs["order"], kargs["offset"]
col_name, output_col_name, fixed_window = kargs["col_to_smooth"], kargs["output_col_name"], kargs["fixed_window"]

# Run smoothing algo on all previous volume data
smoothed_values = np.zeros_like(df[col_name].values)
nb_points_used = np.zeros_like(df[col_name].values, dtype=np.int32)
values_to_smooth = df[col_name].values
for i in range(len(values_to_smooth)):
    smoothed_values[i], nb_points_used[i] = savgol(values_to_smooth[max(0, i-window+1):i+1], window, order, offset, fixed_window)
result[output_col_name] = smoothed_values
result["nb_points_used"] = nb_points_used
