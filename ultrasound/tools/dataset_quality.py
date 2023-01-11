from dataclasses import dataclass
from typing import Tuple, List
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import typer
from rich import print
from rich.console import Console
from rich.table import Table


app = typer.Typer(pretty_exceptions_show_locals=False)


def tof_to_dist(tof: float, temperature: float):
    """Computes the distance from the ToF

    Parameters
    ----------
    tof : float
        Time of flight in ms
    temperature : float
        Temperature in celsius

    Returns
    -------
    float
        Distance in meters
    """
    zero_c_kelvin = 273.15
    temp_kelvin = temperature + zero_c_kelvin
    sound_speed = 20.02 * np.sqrt(temp_kelvin)
    dist = sound_speed * tof * 10e-7 / 2
    return dist


def quality_color(quality: float) -> str:
    """Returns the color associated with the quality of the signal

    Parameters
    ----------
    quality : float
        Quality of the raw signal

    Returns
    -------
    str
        Name of the color
    """
    if quality < 1.5:
        return "red"
    if quality < 3.5:
        return "orange"
    return "green"


def raw_data_quality(data: np.ndarray) -> float:
    """
    Computes a quality index for a raw ultrasound reading

    Parameters
    ----------
    data : np.ndarray
        Raw ultrasound data

    Returns
    -------
    float quality:
        Quality index on a 1-5 scale. 5 means good quality, 1 means poor quality.
    """
    min_peak_height = 40

    # Remove main bang
    data = data[3000:]

    maxs, _ = find_peaks(data, height=min_peak_height, width=20)

    if len(maxs) == 0:
        return 1

    hist, _ = np.histogram(data[maxs], bins=[0, 80, 120, 160, 200, 256], density=True)
    hist = hist / np.max(hist)

    if hist[4] > 0:
        quality = 5
        if hist[4] > 3:
            quality -= 1
    elif hist[3] > 0:
        quality = 4
        if hist[3] > 4:
            quality -= 1
    elif hist[2] > 0:
        quality = 3
        if hist[2] > 5:
            quality -= 1
    elif hist[1] > 0:
        quality = 2
        if hist[1] > 6:
            quality -= 1
    else:
        quality = 1

    return quality


def compute_degradation_distance(distances: np.ndarray, qualities: np.ndarray, quality_threshold: float, nb_bad_data: int) -> float:
    """Computes the distance at which the raw signal starts to degrate

    Parameters
    ----------
    data : pd.DataFrame
        Data containing the distance and the raw ultrasound data

    Returns
    -------
    distance : float
        Distance at which the signal starts to degrade
    """
    start_degrading = 0
    start_degrading_dist = None
    for i, (d, q) in enumerate(zip(distances, qualities)):
        if q < quality_threshold:
            start_degrading += 1
        else:
            start_degrading = 0

        if start_degrading >= nb_bad_data:
            start_degrading_dist = d
            break

    return start_degrading_dist


@dataclass
class ErrorInfo:
    mae: float
    occurences: int


def compute_metrics(data: pd.DataFrame, by: str, steps: List) -> Tuple[List[ErrorInfo], ErrorInfo]:
    """Compute the error of a given dataset

    Parameters
    ----------
    data : pd.DataFrame
        data containing all the useful columns for computing the error
    by : str
        column name to compute metrics versus (distance or temperature)
    steps : list
        steps by which to compute the metrics

    Returns
    -------
    List[ErrorInfo], ErrorInfo
        All info of the computed errors
    """
    assert by in ["distance", "temperature"], f"Can only compute metrics of dataset by distance or temperature : by={by}"
    errors = []
    error_cdm_kg = abs(data["loadcell_weight"] - data["weight"])
    full_mae = error_cdm_kg.mean()
    full_info = ErrorInfo(mae=round(full_mae, 2), occurences=len(data))
    for i in range(1, len(steps)):
        t1 = steps[i - 1]
        t2 = steps[i]
        match by:
            case "distance":
                d = data[(data["loadcell_dist"] >= t1) & (data["loadcell_dist"] < t2)]
            case "temperature":
                d = data[(data["temperature"] >= t1) & (data["temperature"] < t2)]

        error_cdm = abs(d["loadcell_weight"] - d["weight"])
        d_mae = error_cdm.mean()
        error_info = ErrorInfo(mae=round(d_mae, 2), occurences=len(d))
        errors.append(error_info)

    return errors, full_info


def assert_col(df: pd.DataFrame, col_name: str, message: str):
    if col_name:
        assert col_name in df.columns, message


def create_table(errors: List[ErrorInfo], steps: List, silo: str, full_error: ErrorInfo, col_name: str, title: str):
    # Define and create a nice table with error infos
    table = Table(title=title)
    color = "cyan"
    table.add_column(col_name, justify="center", style="yellow")
    table.add_column("Nb data", justify="center", style=color)
    table.add_column("Mean error (ton)", justify="center", style=color)

    for i, (err, step) in enumerate(zip(errors[:-1], steps[:-1])):
        table.add_row(f"{round(step)} -> {round(steps[i + 1])}", f"{errors[i].occurences}", f"{errors[i].mae}")
    table.add_row("all", f"{full_error.occurences}", f"{full_error.mae}")

    console = Console()
    console.print(table)


@app.command()
def main(
        data_path: Path, tof_col_name: str = None, raw_data_col_name: str = None, temperature_col_name: str = None, silo: str = "all", silo_col_name: str = None,
        time_col_name: str = "AcquisitionTime", weight_col_name: str = None, loadcell_weight_col_name: str = None, loadcell_tof_col_name: str = None, config_col_name: str = None,
        config_name: str = None, plot: bool = False, output_dir: Path = Path("./outputs/")):

    # Load the DataFrame from the path
    df = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime})
    df = df.sort_values(by="AcquisitionTime")

    # Assert that the given column names exist
    assert_col(df, tof_col_name, f"Name of column for the ToF does not exist : {tof_col_name}")
    assert_col(df, raw_data_col_name, f"Name of column for the raw ultrasound data does not exist : {raw_data_col_name}")
    assert_col(df, weight_col_name, f"Name of column for the measured grain weight does not exist : {weight_col_name}")
    assert_col(df, loadcell_weight_col_name, f"Name of column for the weight from the Load Cell does not exist : {loadcell_weight_col_name}")
    assert_col(df, loadcell_tof_col_name, f"Name of column for the ToF from the Load Cell does not exist : {loadcell_tof_col_name}")
    assert_col(df, config_col_name, f"Name of column for the config name does not exist : {config_col_name}")
    assert_col(df, silo_col_name, f"Name of column for the silo name does not exist : {silo_col_name}")
    assert_col(df, time_col_name, f"Name of column for the time name does not exist : {time_col_name}")

    # Select the silo if given
    if silo != "all":
        assert silo in df[silo_col_name].unique(), f"Name of silo does not exist in this dataset : {silo}"
        df = df[df[silo_col_name] == silo].reset_index()

    # Select the config if given
    if config_name:
        assert config_name in df[config_col_name].unique(), f"Name of config does not exist in this dataset : {config_name}"
        df = df[df[config_col_name] == config_name].reset_index()

    # Compute the distance at which the signal degrades if all given columns are valid
    if raw_data_col_name and tof_col_name and temperature_col_name:
        print("[blue] ------------------------------ [/blue]")
        # Load the raw data column correctly
        df[raw_data_col_name] = df.apply(lambda x: json.loads(x[raw_data_col_name]), axis=1)
        if loadcell_tof_col_name:
            df["loadcell_dist"] = df.apply(lambda x: tof_to_dist(x[loadcell_tof_col_name], x[temperature_col_name]), axis=1)

        # Compute the distance and the quality of the raw data
        df["dist"] = df.apply(lambda x: tof_to_dist(x[tof_col_name], x[temperature_col_name]), axis=1)
        df["quality"] = df.apply(lambda x: raw_data_quality(np.array(x[raw_data_col_name])), axis=1)

        # Remove data that the distance is less than 1 meter, because we are blind anyways at this distance
        min_dist_df = df[df["dist"] > 1]

        # Compute distances at which the data starts getting worse, based on mean and std
        # Old method
        # start_degrading_dist = df[df["quality"] < 3.5]["dist"].mean() - 0.5 * df[df["quality"] < 3.5]["dist"].std()
        start_degrading_dist = min_dist_df.sort_values(by="dist")[min_dist_df.sort_values(by="dist")["quality"] < 3.5]["dist"].iloc[:10].mean()
        print(f"The data quality starts degrading at : [bold yellow]{round(start_degrading_dist, 2)} m[/bold yellow]")
        # bad_data_dist = df[df["quality"] < 1.5]["dist"].mean() - 0.5 * df[df["quality"] < 1.5]["dist"].std()
        # print(f"The data quality starts to become very bad at : [bold red]{round(bad_data_dist, 2)} m[/bold red]")

        if plot:
            plt.rcParams.update({'font.size': 22})
            custom_legend = [Line2D([0], [0], marker='o', color='w', label='Good data', markerfacecolor='green'),
                             Line2D([0], [0], marker='o', color='w', label='Medium data', markerfacecolor='orange'),
                             Line2D([0], [0], marker='o', color='w', label='Bad data', markerfacecolor='red')]
            df["quality_color"] = df.apply(lambda x: quality_color(x["quality"]), axis=1)
            if loadcell_tof_col_name:
                plt.scatter(df[time_col_name], df["loadcell_dist"], c=df["quality_color"])
            else:
                plt.scatter(df[time_col_name], df["dist"], c=df["quality_color"])
            if start_degrading_dist > 0:
                plt.axhline(start_degrading_dist, color="orange")
            # if bad_data_dist > 0:
                # plt.axhline(bad_data_dist, color="red")
            plt.xlabel("Time")
            plt.ylabel("Depth [m]")
            plt.legend(handles=custom_legend, loc="best")
            plt.show()

    else:
        print("[blue] ------------------------------ [/blue]")
        print("Could not compute the distance at which the raw signal starts to degrade because not enough valid columns where given. We need : ")
        print("\t [bold yellow]- Raw data column[/bold yellow]")
        print("\t [bold yellow]- Time of flight (ToF) column[/bold yellow]")
        print("\t [bold yellow]- Temperature column[/bold yellow]")

    if weight_col_name and loadcell_weight_col_name and tof_col_name and loadcell_tof_col_name and temperature_col_name:
        print("[blue] ------------------------------ [/blue]")

        # Compute distance and convert weight
        df["dist"] = df.apply(lambda x: tof_to_dist(x[tof_col_name], x[temperature_col_name]), axis=1)
        df["loadcell_dist"] = df.apply(lambda x: tof_to_dist(x[loadcell_tof_col_name], x[temperature_col_name]), axis=1)
        df["weight"] = df[weight_col_name] / 1000
        df["loadcell_weight"] = df[loadcell_weight_col_name] / 1000

        # Compute the error metrics by distance
        steps = range(0, 10)
        errors_by_dist, full_error = compute_metrics(df, by="distance", steps=steps)
        create_table(errors_by_dist, steps, silo, full_error, col_name="Depth (m)",
                     title=f"[green]Errors between measured data and loadcells for {silo} silo by distance.[/green]")

        # Compute the error metrics by temperature
        steps = np.linspace(min(df["temperature"].values), max(df["temperature"].values), 10)
        errors_by_temp, full_error = compute_metrics(df, by="temperature", steps=steps)
        create_table(errors_by_temp, steps, silo, full_error, col_name="Temperature (°C)",
                     title=f"[green]Errors between measured data and loadcells for {silo} silo by temperature.[/green]")

    else:
        print("[blue] ------------------------------ [/blue]")
        print("Could not compute the errors between the measures and the loadcell because not enough valid columns where given. We need : ")
        print("\t [bold yellow]- Measured weight column[/bold yellow]")
        print("\t [bold yellow]- Load Cell weight column[/bold yellow]")
        print("\t [bold yellow]- Measured time of flight (ToF) column[/bold yellow]")
        print("\t [bold yellow]- Load Cell time of flight (ToF) column[/bold yellow]")
        print("\t [bold yellow]- Temperature column[/bold yellow]")


if __name__ == "__main__":
    app()
