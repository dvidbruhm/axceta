import pandas as pd
from SGCC.savgol import get_coefficients
from scipy.signal import savgol_filter
import numpy as np
from sklearn.linear_model import LinearRegression


def nonrealtime_savgol(x: np.ndarray, y: np.ndarray, window_size: int, order: int, smoothing: int) -> np.ndarray:
    xs, ys = split_silo_fill(x, y)

    filtered_ys = []
    for _, yt in zip(xs, ys):
        filtered_y = yt
        if len(yt) >= window_size:
            filtered_y = savgol_filter(yt, window_size, order, smoothing)
        filtered_y[filtered_y < 0] = 0
        filtered_ys.append(filtered_y)

    total_filtered_y = np.concatenate(filtered_ys)
    return total_filtered_y

def realtime_savgol(x: np.ndarray, y: np.ndarray, window_size: int, order: int, smoothing: int, offset: int, simple: bool = True) -> np.ndarray:
    coeffs = get_coefficients(smoothing, order, window_size, int((window_size - 1) / 2) - offset)

    xs, ys = split_silo_fill(x, y)
    filtered_ys = []
    for _, yt in zip(xs, ys):
        filtered_y = np.zeros_like(yt)
        for i in range(len(yt)):
            if i < window_size:
                filtered_y[i] = yt[i] if yt[i] > 0 else 0
                continue
            
            if not simple:
                filtered = savgol_filter(yt[i - window_size:i], window_size, order, deriv=smoothing)
                filtered_y[i] = filtered[-offset] if filtered[-offset] > 0 else 0
                continue

            filtered_value = 0
            #print("--------")
            for j, c in enumerate(coeffs):
                #print(filtered_data[window_size - j - 1])
                #print(i, window_size - j - 1, i - (window_size - j - 1), filtered_value, data[i])
                filtered_value += c * yt[i - (window_size - j - 1)]
            filtered_y[i] = filtered_value if filtered_value > 0 else 0
        filtered_ys.append(filtered_y)
    total_filtered_y = np.concatenate(filtered_ys)

    return total_filtered_y


def realtime_regression(x: np.ndarray, y: np.ndarray, window_size: int, show_plot: bool = False) -> np.ndarray:
    xs, ys = split_silo_fill(x, y)

    filtered_ys = []
    for xt, yt in zip(xs, ys):
        filtered_y = np.zeros_like(yt)

        model = LinearRegression()

        for i in range(len(xt)):
            if i < window_size:
                filtered_y[i] = yt[i] if yt[i] > 0 else 0
                continue
            new_x = xt[i - window_size:i]
            new_y = yt[i - window_size:i]
            model.fit(new_x.reshape((-1, 1)), new_y)
            filtered_values = model.predict(new_x.reshape((-1, 1)))
            filtered_value = filtered_values[-1]
            filtered_y[i] = filtered_value if filtered_value > 0 else 0

            if i == 280 and show_plot:
                plt.plot(xt, yt)
                plt.plot(new_x, new_y)
                plt.plot(xt, filtered_y)
                plt.plot(new_x.flatten(), filtered_values)
                plt.scatter([new_x[-1]], [filtered_value])
                plt.show()
        filtered_ys.append(filtered_y)

    total_filtered_y = np.concatenate(filtered_ys)
    
    return total_filtered_y


def split_silo_fill(x: np.ndarray, y: np.ndarray):
    smallest_y = np.inf
    split_points = [0]
    for i in range(len(x)):
        current_y = y[i]
        current_x = x[i]

        if current_y < smallest_y:
            smallest_y = current_y

        if current_y > smallest_y + 10:
            split_points.append(i)
            smallest_y = np.inf
    if len(split_points) > 0 and split_points[-1] != len(x) - 1:
        split_points.append(len(x) - 1)
    xs = []
    ys = []
    for i in range(1, len(split_points)):
        temp_x = x[split_points[i-1]:min(split_points[i], len(x))]
        temp_y = y[split_points[i-1]:min(split_points[i], len(x))]
        xs.append(np.array(temp_x))
        ys.append(np.array(temp_y))
    print(split_points)
    xs[-1] = np.append(xs[-1], xs[-1][-1])
    ys[-1] = np.append(ys[-1], ys[-1][-1])

    #plt.plot(x, y)
    #for xt, yt in zip(xs, ys):
    #    plt.plot(xt, yt)
    #plt.scatter([x[i] for i in split_points], [y[i] for i in split_points])
    #plt.show()

    return xs, ys

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    import snr.data as data
    import snr.utils as utils
    data_path = Path("data", "silo-data-3.csv")
    df = data.load_silo_data(file_path=data_path, scale_dist=True)
    silo_name = "Avinor-1486A"

    conversion_data = data.load_dist_to_volume_data(file_path=Path("data", "dist_to_volume.csv"))
    silo_data = utils.batch_dist_to_vol(df, conversion_data, silo_name)

    resampled_silo_data = silo_data[["AcquisitionTime", "perc_filled"]].resample('1H', on="AcquisitionTime").mean()
    resampled_silo_data["AcquisitionTime"] = resampled_silo_data.index
    resampled_silo_data["perc_filled"] = resampled_silo_data["perc_filled"].interpolate(method="nearest")

    resampled_silo_data["AcquisitionTime"] = resampled_silo_data["AcquisitionTime"].view('int64') // 10 ** 9
    #resampled_silo_data["AcquisitionTime"] = resampled_silo_data["AcquisitionTime"] - resampled_silo_data["AcquisitionTime"].values[0]

    #xs, ys = split_silo_fill(resampled_silo_data["AcquisitionTime"].values, resampled_silo_data["perc_filled"].values)

    x = resampled_silo_data["AcquisitionTime"].values
    y = resampled_silo_data["perc_filled"].values

    y_pred = realtime_regression(x, y, 20)
    #y_pred2 = realtime_regression(x.reshape((-1, 1)), y, 10)
    #y_pred3 = realtime_regression(x.reshape((-1, 1)), y, 30)
    y_savgol = realtime_savgol(x, y, 11, 2, 0, 3)


    print(x.shape)
    print(y.shape)
    print(y_pred.shape)
    plt.plot(x, y, label="data")
    plt.plot(x, y_pred, label="regression")
    plt.plot(x, y_savgol, label="savgol")
    #plt.plot(x, y_pred3, label="r3")
    plt.legend(loc="best")
    plt.show()
    exit()


    silo_data["filtered_perc"] = realtime_savgol(silo_data["perc_filled"].values, 11, 2, 0, 3)
    plt.plot(silo_data["AcquisitionTime"], silo_data["perc_filled"], label="data")
    plt.plot(silo_data["AcquisitionTime"], silo_data["filtered_perc"], label="savgol1")
    plt.plot(silo_data["AcquisitionTime"], silo_data["filtered_perc2"], label="savgol2")
    plt.legend(loc="best")
    plt.show()
