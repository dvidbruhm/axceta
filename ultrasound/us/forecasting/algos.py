from typing import List, Tuple, NamedTuple, Any
from pathlib import Path
from rich import print

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_load_cell_data(data_path: Path, silo_name: str, version: str = "v1", col_name: str = "LoadcellWeight_t") -> pd.DataFrame:
    assert version in ["v1", "v2", "v3"]
    if version == "v1":
        df = pd.DataFrame(pd.read_csv(data_path, usecols=["AcquisitionDate", silo_name], converters={
                          silo_name: lambda x: float(x[:-2]) if len(x) > 0 else np.NaN}))
        time_col_name = "AcquisitionDate"
    elif version == "v2":
        df = pd.DataFrame(pd.read_csv(data_path))
        df = df.loc[df["LocationName"] == silo_name]
        time_col_name = "AcquisitionTime"
        df[silo_name] = df["FeedRemaining_kg"]
        df = df.drop(["FeedRemaining_kg", "LocationName"], axis=1)
    elif version == "v3":
        df = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime})
        df = df.loc[df["LocationName"] == silo_name]
        time_col_name = "AcquisitionTime"
        df[silo_name] = df[col_name]
        df = df.drop([col_name, "LocationName"], axis=1)
    df.set_index(pd.DatetimeIndex(df[time_col_name]), inplace=True)
    df = pd.DataFrame(df.drop([time_col_name], axis=1))
    df = df.resample("1H").mean()
    df.index = df.index.tz_localize(None)
    df[silo_name] = df[silo_name].interpolate(method='polynomial', order=3)
    return df


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def normalize_array(data: np.ndarray) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_currently_active(data1: pd.DataFrame, data2: pd.DataFrame, silo_name1: str, silo_name2: str, nb_prev_points: int = 5) -> pd.DataFrame:
    points1 = data1[silo_name1].values[-nb_prev_points:]
    points2 = data2[silo_name2].values[-nb_prev_points:]

    pred1, _ = regression(points1)
    pred2, _ = regression(points2)
    slope1 = (pred1[-1] - pred1[0]) / len(pred1)
    slope2 = (pred2[-1] - pred2[0]) / len(pred2)

    if slope1 < slope2:
        return data1
    return data2


def find_consecutive(data, min_len: int = 10):
    data = data.copy()                   # avoid mutating the original list
    counting = []                      # keep track of True indexes, to count them later
    for i in range(len(data)):          # cycle by index
        is_last = i + 1 >= len(data)    # True if this is the last index in the array
        if data[i] == True:
            counting.append(i)         # add value to list if True
        if is_last or data[i] == False:  # when we are at the last entry, or find a False
            if len(counting) < min_len:      # check the length of our True indexes, and if less than 6
                for j in counting:
                    data[j] = False     # make each False
            counting = []
    return data


def find_not_both_flat(data1: pd.DataFrame, data2: pd.DataFrame, min_len: int = 10, debug: bool = False):
    data1["is_flat"] = find_flat_signals(data1.iloc[:, 0].values, min_len)
    data2["is_flat"] = find_flat_signals(data2.iloc[:, 0].values, min_len)
    both_flat = [(flat1 and flat2) for flat1, flat2 in zip(data1["is_flat"], data2["is_flat"])]
    both_flat = find_consecutive(both_flat, min_len)
    not_both_flat = [not flat for flat in both_flat]
    not_both_flat = find_consecutive(not_both_flat, min_len)

    if debug:
        for i, data in enumerate([data2, data1]):
            plt.subplot(2, 1, i + 1)
            # plt.plot(data.iloc[:, 0])
            # plt.plot(data[np.array(both_flat) == True].iloc[:, 0], '.')
            plt.plot(data[np.array(not_both_flat) == True].iloc[:, 0], '.')
        plt.show()
        print(both_flat)
        print(data1)
        print(data2)
        exit()

    return not_both_flat


Cycle = NamedTuple("Cycle", [("start", int), ("end", int)])


def find_big_cycles(not_both_flat, start_at: str = "start", debug: bool = False):
    assert (start_at.lower() in ["start", "end"])

    data = np.argwhere(np.diff(not_both_flat)).squeeze()
    start_index = 0 if start_at == "start" else 1
    cycles = []
    if start_at == "end":
        cycles.append(Cycle(start=0, end=data[0]))
    for i in range(start_index, len(data) - 1, 2):
        cycles.append(Cycle(start=data[i], end=data[i + 1]))
    if (start_at == "end" and len(data) % 2 == 0) or (start_at == "start" and len(data) % 2 == 1):
        cycles.append(Cycle(start=data[-1], end=len(not_both_flat) - 1))

    if debug:
        print(list(range(start_index, len(data) - 1, 2)))
        print(data)
        print(cycles)

    return cycles


def find_flat_signals(data: np.ndarray, min_len: int = 10, debug: bool = False) -> List[Tuple[int, int]]:
    data = data / np.max(data)
    threshold = 0.0004
    is_flat = [grad > -threshold for grad in moving_average(np.gradient(data), 5)]
    count = 0
    flat_segments = []
    for i, flat in enumerate(is_flat):
        if flat:
            count += 1
            if count > min_len:
                if (i + 1) > len(is_flat) - 1 or not is_flat[i + 1]:
                    flat_segments.append((i - count, i))
        else:
            count = 0

    if debug:
        plt.clf()
        plt.plot(data, '.')
        plt.plot(moving_average(np.gradient(data), 5))
        for (start, end) in flat_segments:
            print(start, end)
            plt.axvline(start)
            plt.axvline(end)
        plt.show()
        exit()

    return list(is_flat)


def find_silo_fills(data: np.ndarray) -> List[int]:
    data = data / np.max(data)
    threshold = 0.05
    is_fill = [grad > threshold for grad in np.gradient(data)]
    fill_indices, = np.where(is_fill)
    fill_indices = list(fill_indices)
    fixed_fill_indices = []
    for i in range(len(fill_indices) - 1):
        current_index = fill_indices[i]
        next_index = fill_indices[i + 1]
        if next_index - current_index != 1:
            fixed_fill_indices.append(current_index)
    fixed_fill_indices.append(fill_indices[-1])
    """
    print(fill_indices)
    plt.plot(data, '.')
    plt.plot(np.gradient(data))
    print(fixed_fill_indices)
    for index in fixed_fill_indices:
        plt.axvline(index)
    plt.show()
    """
    return fixed_fill_indices


FillInfo = NamedTuple(
    "FillInfo",
    [
        ("start_date", Any),
        ("end_date", Any),
        ("nb_hours", int),
        ("start_value", float),
        ("end_value", float),
        ("rate_per_unit", float),
        ("big_cycle_nb", int)
    ]
)


def index_to_cycle(index: int, big_cycles: List[Cycle]) -> int:
    for i, cycle in enumerate(big_cycles):
        if cycle.start < index <= cycle.end:
            return i
    return -1


def find_start_end_of_fill(start_index: int, end_index: int, data: np.ndarray, threshold: float = 0.005):
    data = normalize_array(data)
    gradient = np.abs(moving_average(np.gradient(data), 2))
    start_index += 5
    current_grad = gradient[start_index]
    # find start
    while current_grad < threshold:
        current_grad = gradient[start_index]
        start_index += 1
    # find end
    end_index -= 5
    current_grad = gradient[end_index]
    while current_grad < threshold:
        current_grad = gradient[end_index]
        end_index -= 1
    return start_index, end_index, gradient


def get_fill_rate_info(fill_indices: List[int], data: pd.DataFrame, silo_name: str, big_cycles: List[Cycle]) -> List:
    np_data = data[silo_name].values
    fill_rate_info = []
    for i in range(1, len(fill_indices)):
        current_index = fill_indices[i]
        prev_index = fill_indices[i - 1]
        starti, endi, gradient = find_start_end_of_fill(prev_index, current_index, np_data)
        nb_hours = max(endi - starti, 1)
        info = FillInfo(
            start_date=data.index[starti],
            end_date=data.index[endi],
            nb_hours=nb_hours,
            start_value=np_data[starti],
            end_value=np_data[endi],
            rate_per_unit=(np_data[starti] - np_data[endi]) / (endi - starti),
            big_cycle_nb=index_to_cycle(endi, big_cycles)
        )
        if nb_hours > 5:
            fill_rate_info.append(info)
    return fill_rate_info


def predict_next_fill(fill_info: List[FillInfo], data: pd.DataFrame, silo_name: str, fill_value: float = 0.0, debug: bool = False) -> int:
    current_value = data[silo_name].iat[-1]
    next_rate = (fill_info[-1].rate_per_unit - fill_info[-2].rate_per_unit) + fill_info[-1].rate_per_unit
    nb_hours = (current_value - fill_value) / next_rate
    if debug:
        print(next_rate)
        print(nb_hours)
        print(data.index[-1])
        print(data.index[-1] + pd.Timedelta(hours=nb_hours))

    next_fill_pred = data.index[-1] + pd.Timedelta(hours=nb_hours)
    return next_fill_pred


def regression(points: np.ndarray):
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    X = np.array(range(1, len(points) + 1)).reshape(-1, 1)
    Y = points
    regr.fit(X, Y)
    pred = regr.predict(X)
    return pred, regr


def simple_regression_pred(data: np.ndarray, nb_points: int = 24, threshold: float = 0, debug: bool = False):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    points = data[-nb_points:]
    pred, regr = regression(points)
    X_test = np.array(range(1, nb_points + 1)).reshape(-1, 1)
    current_fill = 10000
    empty_date = nb_points
    pred = list(regr.predict(X_test))

    while current_fill > threshold:
        current_fill = regr.predict(np.array([[empty_date]]))
        if debug:
            print(current_fill)
        pred.append(current_fill[0])
        empty_date += 1

        if empty_date > nb_points + 200:
            empty_date = -1
            break

    r2 = r2_score(pred[:nb_points], points)
    print(f"R2 score [bold green]{r2}[/bold green]")
    if debug:
        plt.clf()
        plt.plot(points, '.')
        plt.plot(pred)
        plt.show()
        exit()

    return pred, empty_date
