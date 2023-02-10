import pandas as pd
import numpy as np


def mae(true, measured):
    """Computes the mean absolute error between true and measured data

    Parameters
    ----------
    true : list
        true values
    measured : list
        measured data

    Returns
    -------
    float
        Computed mean absolute error
    """
    output = np.mean(abs(true - measured))
    return output


def rmse(true, measured):
    """Computes the root mean squared error between true and measured data

    Parameters
    ----------
    true : list
        true values
    measured : list
        measured data

    Returns
    -------
    float
        Computed root mean squared error
    """
    mse = np.mean((true - measured) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def mape(true, measured):
    """Computes the mean absolute percentage error between true and measured data

    Parameters
    ----------
    true : list
        true values
    measured : list
        measured data

    Returns
    -------
    float
        Computed mean absolute percentage error
    """
    mape = abs((true - measured) / true)
    mape = np.mean(mape) * 100
    return mape


def compute_metrics(true_values, measured_values, distances):
    """Compute different error metrics between true and measured values, separated by distances

    Parameters
    ----------
    true_values : list
        List of values that are considered true/reliable
    measured_values : list
        List of values that are measured to compare tot he true values
    distances : list
        List of distances

    Returns
    -------
    dist
        Dictionnary containing all the computed metrics
    """

    # Make sure inputs are as numpy array
    true_values = np.array(true_values)
    measured_values = np.array(measured_values)
    distances = np.array(distances)

    # Check that all input arrays are the same length
    assert len(true_values) == len(measured_values) == len(distances), "All input arrays must be of same length."
    assert len(true_values) > 0, "The input lists must contain data."

    # Create a pandas DataFrame for easier processing and filtering
    df = pd.DataFrame({"true_values": true_values, "measured_values": measured_values, "distance": distances})

    # Compute the distance steps to check the errors
    steps = range(int(min(distances)), np.ceil(max(distances)).astype("int") + 1)

    # Init arrays
    mae_errors_by_dist = []
    rmse_errors_by_dist = []
    mape_errors_by_dist = []
    counts_by_dist = []
    dist_names = []

    # Compute the errors by distance
    for i in range(1, len(steps)):
        step1 = steps[i - 1]
        step2 = steps[i]

        # Filter the data by distance
        filtered_data = df[(df["distance"] >= step1) & (df["distance"] < step2)]

        mae_val = mae(filtered_data["true_values"], filtered_data["measured_values"])
        rmse_val = rmse(filtered_data["true_values"], filtered_data["measured_values"])
        mape_val = mape(filtered_data["true_values"], filtered_data["measured_values"])

        mae_errors_by_dist.append(round(mae_val, 2))
        rmse_errors_by_dist.append(round(rmse_val, 2))
        mape_errors_by_dist.append(round(mape_val, 2))
        counts_by_dist.append(len(filtered_data))
        dist_names.append(f"{step1} to {step2} meters.")

    # Compute the errors on all the data
    mae_total = mae(df["true_values"], df["measured_values"])
    rmse_total = rmse(df["true_values"], df["measured_values"])
    mape_total = mape(df["true_values"], df["measured_values"])

    mae_errors_by_dist.append(round(mae_total, 2))
    rmse_errors_by_dist.append(round(rmse_total, 2))
    mape_errors_by_dist.append(round(mape_total, 2))
    counts_by_dist.append(len(df))
    dist_names.append("all data")

    # Construct the final dict to return
    output = {
        "Distance": dist_names,
        "Nb Data": counts_by_dist,
        "Mean Absolute Error": mae_errors_by_dist,
        "Root Mean Squared Error": rmse_errors_by_dist,
        r"Mean Absolute % Error": mape_errors_by_dist
    }

    return output
