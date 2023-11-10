import pandas as pd


def get_mean_consumption(values, max_pos_delta=2, max_neg_delta=-4):
    # Compute the mean consumption of the values, no library/functions used so the code is pretty straightforward
    mean_consum = 0
    count = 0
    weights = [w / len(values) for w in range(1, len(values) + 1)]
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        if d > max_pos_delta or d < max_neg_delta:
            continue
        mean_consum += d
        count += 1
    if count > 0:
        mean_consum /= count
    return mean_consum


def is_silo_active(values):
    # Check if the silo is active by checking if the mean consumption is too
    # small (which means the silo is stagnant)
    if values[-1] < 0.1:
        return False

    mean_consum = get_mean_consumption(values)
    if mean_consum > -0.05:
        return False
    return True


def predict_next_fill(times, values, resample_hours=4, tons_fill_threshold=0, max_days_before=60, nb_hours_silo_inactive=30):
    # Use a pandas DataFrame for easier resampling, but could do without by using another
    # way to resample and keeping the times and values as simple arrays
    df = pd.DataFrame(data={"values": values}, index=times)

    # Only keep data that is not older than max_days_before, example approx code without pandas:
    #   cut_index = 0
    #   current_time = times[-1]
    #   for i in range(len(times)):
    #       if times[i] > current_time - max_days_before:
    #           cut_index = i
    #           break
    #   times = times[cut_index:]
    #   values = values[cut_index:]
    #
    df = df[df.index > df.index[-1] - pd.Timedelta(days=max_days_before)]

    # Resample the data to resample_hours (4) instead of 1 hour by taking the mean
    # might need to find a function/library to resample, or implement it
    resampled = df.resample(f"{resample_hours}H").mean().dropna(subset=["values"])
    # Check if silo is active by passing only data the is not older than nb_hours_silo_inactive (similar as above)
    silo_active = is_silo_active(resampled[resampled.index > resampled.index[-1] - pd.Timedelta(hours=nb_hours_silo_inactive)].values)
    if not silo_active:
        print("Silo is inactive")
        return None, None

    mean_consum = get_mean_consumption(resampled["values"].values)

    if mean_consum > 0:
        print("Mean consum is positive instead of negative")
        return None, None

    # Using the mean consommation, predict the next value until the silo is empty (we hit 0 or tons_fill_threshold):
    #   start with the last value in the values array, add the mean consum (which is negative) and add the
    #   next time (which is the previous time + the resample_hours), and repeat
    pred_times = [resampled.index[-1]]
    pred_values = [resampled["values"].values[-min(len(resampled["values"].values), 3) :].mean()]
    print(resampled["values"].values)
    print(resampled["values"].values[-min(len(resampled["values"].values), 3) :])
    while pred_values[-1] > 0:
        pred_values.append(pred_values[-1] + mean_consum)
        pred_times.append(pred_times[-1] + pd.Timedelta(hours=resample_hours))

    # return the last predicted time which is the date that the silo is predicted to be empty
    return pred_times[-1], pred_values[-1]
