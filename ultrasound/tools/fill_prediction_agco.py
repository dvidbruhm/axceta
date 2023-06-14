import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_mean_consumption(values, max_delta_tons=2):
    mean_consum = 0
    count = 0
    for i, d in enumerate(np.diff(values)):
        if d > 2:
            continue
        mean_consum += d
        count += 1
    mean_consum /= count
    return mean_consum


def is_silo_active(df, max_nb_hours=30):
    new_df = df[df.index > df.index[-1] - pd.Timedelta(hours=max_nb_hours)]
    values = new_df["values"].values
    if values[-1] < 0.1:
        return False

    mean_consum = get_mean_consumption(values)
    if mean_consum > -0.05:
        return False
    return True


def predict_next_fill_agco(times, values, resample_hours=4, tons_fill_threshold=0, max_days_before=60, nb_hours_silo_inactive=30):
    df = pd.DataFrame(data={"values": values}, index=times)
    df = df[df.index > df.index[-1] - pd.Timedelta(days=max_days_before)]
    resampled = df.resample(f"{resample_hours}H").mean()
    silo_active = is_silo_active(resampled, max_nb_hours=nb_hours_silo_inactive)
    if not silo_active:
        return None, None

    mean_consum = get_mean_consumption(resampled["values"].values)

    pred_times = [resampled.index[-1]]
    pred_values = [resampled["values"].values[-min(len(resampled["values"].values), 3):-1].mean()]
    while pred_values[-1] > tons_fill_threshold:
        pred_values.append(pred_values[-1] + mean_consum)
        pred_times.append(pred_times[-1] + pd.Timedelta(hours=resample_hours))

    return pred_times[-1], pred_values[-1]


if __name__ == "__main__":
    data = pd.read_csv("data/dashboard/output/Mpass7_best_weight.csv", converters={"AcquisitionTime": pd.to_datetime})
    lc_data = pd.read_csv("data/dashboard/loadcells/Mpass7.csv", converters={"AcquisitionTime": pd.to_datetime})
    resampled = data.set_index("AcquisitionTime").resample("4H").mean()
    values = data["best_weight"].values

    starts = [30, 50, 60, 74, 80, 90, 95, 100, 120, len(values) - 1]
    for i, start in enumerate(starts):
        plt.subplot(len(starts), 1, i + 1)
        pred_time, pred_value = predict_next_fill_agco(
            data["AcquisitionTime"].values[: start],
            data["best_weight"].values[: start],
            tons_fill_threshold=0)

        plt.plot(data["AcquisitionTime"], data["best_weight"])
        plt.plot(lc_data["AcquisitionTime"], lc_data["LoadCellWeight_t"], color="green")
        plt.plot(data["AcquisitionTime"].values[:start], data["best_weight"].values[:start], ".", color="red")
        if pred_time is not None:
            plt.axvline(pred_time)
    plt.show()
