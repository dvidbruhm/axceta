import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from rich import print
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)


def create_dataset(path):
    data = pd.read_csv(path, converters={"raw_data": json.loads, "AcquisitionTime": pd.to_datetime})
    data["AcquisitionTimeDelta"] = data["AcquisitionTime"].diff().shift(-1)
    data = data[data["AcquisitionTimeDelta"] > pd.Timedelta(minutes=10)].reset_index()
    data["weight_t"] = [d[0] if not np.isnan(d[0]) else d[1] for d in zip(data["weight_latest_t"].values, data["weight_pga_t"].values)]
    data = data.drop("raw_data", axis=1)
    data = data.set_index("AcquisitionTime")
    data = data.resample("1H").mean(True)
    return data


def read_data(path):
    data = pd.read_csv(path, converters={"AcquisitionTime": pd.to_datetime})
    # data = data.set_index("AcquisitionTime")
    # data["weight_t"] = data["weight_t"].interpolate(method="linear")
    return data


def add_next_fill_index_to_dataset(data):
    temp = []
    temp2 = []
    for i in range(len(data)):
        next_fill_index = len(data) - 1
        for j in range(i, len(data) - 1):
            current_d = data.iloc[j]["weight_t"]
            next_d = data.iloc[j + 1]["weight_t"]
            if next_d - current_d > 3:
                next_fill_index = j
                break
        temp.append(next_fill_index)

        prev_fill_index = 0
        for j in range(i, 1, -1):
            current_d = data.iloc[j]["weight_t"]
            prev_d = data.iloc[j - 1]["weight_t"]
            if current_d - prev_d > 3:
                prev_fill_index = j
                break
        temp2.append(prev_fill_index)

    data["next_fill_index"] = temp
    data["prev_fill_index"] = temp2

    return data


def add_next_fill_to_dataset(data):

    import us.forecasting.algos as fore

    next_zero_dates = []
    for i in range(len(data)):
        if np.isnan(data.iloc[i]["weight_t"]):
            next_zero_dates.append(np.nan)
            continue
        prev_fill_index = int(data.iloc[i]["prev_fill_index"])
        next_fill_index = int(data.iloc[i]["next_fill_index"])
        points = data.iloc[prev_fill_index:next_fill_index]["weight_t"].interpolate(method="linear").values
        if len(points) < 2:
            next_zero_dates.append(np.nan)
            continue
        pred, regr = fore.regression(points)
        # plt.plot(range(len(data)), data["weight_t"], '.')
        # plt.plot(range(prev_fill_index, prev_fill_index + len(points)), points, "x")
        # plt.plot(range(prev_fill_index, prev_fill_index + len(points)), pred)
        current_fill = 10000
        empty_date = 0
        temp = []
        while current_fill > 0:
            current_fill = regr.predict(np.array([[empty_date]]))
            # print(current_fill)
            temp.append(current_fill[0])
            empty_date += 1

            if empty_date > i + 200:
                empty_date = -1
                break
        next_zero_date = data.index[prev_fill_index] + pd.Timedelta(hours=empty_date)
        next_zero_dates.append(next_zero_date)
        """
        print(i, prev_fill_index, next_fill_index)
        print(empty_date)
        plt.show()
        plt.plot(data["weight_t"], '.')
        plt.axvline(next_zero_date)
        plt.show()
        exit()
        plt.plot(temp, "-")
        plt.plot(range(len(data)), data["weight_t"], '.')
        plt.plot(pred, '.')
        plt.plot(points, 'x')
        plt.axvline(i)
        plt.axvline(empty_date)
        plt.show()
        break
    exit()
    """
    data["next_fill"] = next_zero_dates

    return data


def find_fills(values, threshold):
    fills = []
    values = np.array(values)
    for i in range(len(values)):
        next_v = values[min(len(values) - 1, i + 1)]
        v = values[i]
        prev_v = values[max(0, i - 1)]
        if v - prev_v > threshold:
            fills.append(i)
    return fills


from tools.smoothing import linear_regression


def predict_next_fill(time, values, max_regression_len=48):

    if len(values) < 7:
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

    # Find sharp augmentations in value: silo is filled
    fills = find_fills(values, 3)
    fills.insert(0, 0)
    # print(fills)
    # plt.clf()
    # plt.plot(values)
    # for fill in fills:
    #    plt.axvline(fill)
    # plt.show()
    # exit()
    current_fill_values = values[fills[-1]:]
    current_fill_time = time[fills[-1]:]

    if len(current_fill_values) < 7:
        if len(fills) > 1:
            # Going backwards, find first fill that contains more than N data points
            found = False
            for i in range(len(fills) - 1):
                prev_values = values[fills[-i - 2]:fills[-i - 1]]
                prev_time = time[fills[-i - 2]:fills[-i - 1]]
                print(len(values), fills[-i - 2], fills[-i - 1])

                if len(prev_values) >= 7:
                    found = True
                    break
            if found is False:
                print("[red]No prediction[/red], did not found previous fill with enough data")
                return np.datetime64("nat")

            y_reg, r2, m, c = linear_regression(prev_time, prev_values)

            next_xs = []
            next_ys = []
            for i in range(200):
                next_x = (time[-1] + i * time_step)
                fill_level = m * next_x + c + values[-1]
                next_xs.append(next_x)
                next_ys.append(fill_level)
                if fill_level < 0:
                    break
            else:
                print("[red]No prediction[/red], the linear regression did not go to zero on previous fill, probably flat or uphill")
                return np.datetime64("nat")

            next_fill_hours = int((next_xs[-1] - 1) / time_step)
            next_fill_date = end_time + np.timedelta64(next_fill_hours, "h")
            return next_fill_date

        else:
            print("[red]No prediction[/red], there is no previous fill, and not enough data in current fill")
            return np.datetime64("nat")

    best_r2 = -100
    best_time = []
    best_values = []
    best_values_reg = []
    best_params = ()

    for i in range(max(0, len(values) - max_regression_len), len(values) - 5, 2):
        x_slice = time[i:]
        y_slice = values[i:]

        y_reg, r2, m, c = linear_regression(x_slice, y_slice)
        if r2 > best_r2:
            best_r2 = r2
            best_time = x_slice
            best_values = y_slice
            best_values_reg = y_reg
            best_params = (m, c)
        if best_r2 > 0.8:
            break

    if not best_params:
        print("[red]No prediction[/red], linear regression did not work")
        return np.datetime64("nat")
    # plt.clf()
    # plt.plot(time, values, '.')
    # plt.plot(best_time, best_values, 'x')
    # plt.plot(best_time, best_values_reg)
    # print(best_params)
    # print(start_time, end_time, time_step)

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

        if len(next_ys) > 200:
            next_xs = []
            next_ys = []
            break

    # plt.plot(next_xs, next_ys, ".")
    # plt.show()
    # exit()

    if len(next_xs) > 0:
        next_fill_hours = int((next_xs[-1] - 1) / time_step)
        next_fill_date = end_time + np.timedelta64(next_fill_hours, "h")
        if next_fill_date < end_time:
            next_fill_date = end_time
    else:
        print("[red]No prediction[/red], the linear regression did not go to zero, probably flat or uphill")
        return np.datetime64("nat")
    return next_fill_date


if __name__ == "__main__":
    data = read_data("data/isoporc/dataset_fill.csv")
    data = data.set_index("AcquisitionTime")

    # data = add_next_fill_to_dataset(data)
    # data.to_csv("data/isoporc/dataset_fill.csv")

    """
    ind = 980
    print(data["next_fill_index"][ind], data["prev_fill_index"][ind])
    plt.plot(range(len(data)), data["weight_t"], '.')
    plt.show()

    print(data["next_fill"].unique())
    plt.plot(data["weight_t"], '.')
    plt.axhline(0, color="gray", linestyle="--")
    plt.plot(data.index[ind], data["weight_t"][ind], "x")
    plt.axvline(pd.to_datetime(data["next_fill"][ind]))

    plt.show()
    """
    indices = [100, 150]
    # indices = [750, 120, 160, 300, 360, 403, 410, 420, 500, 540, 750, len(data) - 1]

    """
    for i, ind in enumerate(indices):
        plt.subplot(len(indices), 1, i + 1)

        next_fill_date = predict_next_fill(data.index[:ind], data["weight_t"][:ind])
        plt.plot(data.index[:ind], data["weight_t"][:ind], ".")
        plt.plot(data.index[ind:], data["weight_t"][ind:], ".")
        plt.plot(data.index[ind - 1], data["weight_t"][ind - 1], "o", color="red")
        plt.axvline(next_fill_date)
        plt.axhline(0, color="gray", linestyle="--")
    plt.show()
    exit()
    """

    import tools.fill_prediction as fp
    preds = []
    for i in range(len(data)):
        if np.isnan(data["weight_t"][i]):
            preds.append(np.datetime64("nat"))
            continue
        next_fill_date = fp.predict_next_fill(data.index[:i + 1], data["weight_t"][:i + 1])
        print(i, next_fill_date)
        preds.append(next_fill_date)
    data["prediction"] = preds
    data.to_csv("data/isoporc/dataset_fill_preds.csv")
