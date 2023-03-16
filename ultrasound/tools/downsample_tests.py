from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import us.utils as utils
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import tools.ultrasound_algos as ua

# Define custom progres
progress_bar = Progress(
    TextColumn("[green]{task.description}[/green][purple]{task.percentage:>3.0f}%[/purple]"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

DOWNSAMPLE_FACTORS = [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]


def compute_results(results):
    nb_samples = results["sample_nb"].max()
    downsample_factors = results["downsample_factor"].unique()

    column = "wavefront"
    multiply_factor = True
    errors = {}
    for downsample_factor in downsample_factors:
        errors[f"err_{column}_{downsample_factor}"] = []
    with progress_bar as p:
        for i in p.track(range(nb_samples + 1), description="Computing errors..."):
            for downsample_factor in downsample_factors:
                row_1 = results.loc[(results["sample_nb"] == i) & (results["downsample_factor"] == 1)]
                row_2 = results.loc[(results["sample_nb"] == i) & (results["downsample_factor"] == downsample_factor)]

                val_1 = row_1[column].values[0]
                val_2 = row_2[column].values[0]

                if multiply_factor:
                    val_2 = val_2 * downsample_factor

                err = abs(val_1 - val_2)
                errors[f"err_{column}_{downsample_factor}"].append(err)

    file_name = f"data/downsample_tests/errors-{column}.csv"
    print(f"Saving errors to file : {file_name}")

    df = pd.DataFrame.from_dict(errors)
    df.to_csv(file_name, index=None)


def display_errors(errors):
    print()
    downsample_factors = []
    for n in errors.columns:
        down_fac = int(str(n).split("_")[-1])
        downsample_factors.append(down_fac)
    downsample_factors = sorted(downsample_factors)

    errors_mean = []
    errors_max = []
    errors_argmax = []
    for k in errors.keys():
        mean = np.mean(errors[k])
        max = np.max(errors[k])
        arg_max = np.argmax(errors[k])
        errors_mean.append(mean)
        errors_max.append(max)
        errors_argmax.append(arg_max)

    print(errors_argmax)
    plt.plot(downsample_factors, errors_mean, "--", label="Mean of error")
    plt.plot(downsample_factors, errors_mean, ".")
    plt.plot(downsample_factors, errors_max, "--", label="Max of error")
    plt.plot(downsample_factors, errors_max, ".")
    plt.legend(loc="best")
    plt.show()


def filter_data(data):
    to_remove = []
    with progress_bar as p:
        for i in p.track(range(len(data)), description="Filtering..."):
            raw_data = data.loc[i, "ultrasound"]
            raw_data = np.array(raw_data)
            pulse_count = data.loc[i, "pulseCount"]
            bang_end = ua.detect_main_bang_end(raw_data, pulse_count, 500000)
            auto_gain = ua.auto_gain_detection(raw_data, bang_end)
            if auto_gain != 0:
                to_remove.append(i)
    data = data.drop(to_remove).reset_index()
    return data


def run_downsample_tests(data):

    results = {
        "sample_nb": [],
        "batch_id": [],
        "location_name": [],
        "downsample_factor": [],
        "bang_end": [],
        "bang_end_meters": [],
        "auto_gain": [],
        "wavefront": [],
        "wavefront_meters": []}
    with progress_bar as p:
        for i in p.track(range(len(data)), description="Downsampling..."):
            for j, down_fac in enumerate(DOWNSAMPLE_FACTORS):
                sample_rate = 500000 / down_fac
                pulse_count = data.loc[i, "pulseCount"]
                temperature = data.loc[i, "temperature"]
                signal = np.array(data.loc[i, "ultrasound"])[::down_fac]

                bang_end = ua.detect_main_bang_end(signal, pulse_count, sample_rate)
                auto_gain = ua.auto_gain_detection(signal, bang_end, sample_rate)
                wavefront = ua.wavefront(signal, temperature, 0.5, 0.8, pulse_count, sample_rate)

                wavefront_meters = utils.tof_to_dist2(wavefront * (1000000 / sample_rate), temperature)
                bang_end_meters = utils.tof_to_dist2(bang_end * (1000000 / sample_rate), temperature)

                batch_id = data.loc[i, "batchId"]
                location_name = data.loc[i, "LocationName"]
                results["sample_nb"].append(i)
                results["batch_id"].append(batch_id)
                results["location_name"].append(location_name)
                results["downsample_factor"].append(down_fac)
                results["bang_end"].append(bang_end)
                results["bang_end_meters"].append(bang_end_meters)
                results["auto_gain"].append(auto_gain)
                results["wavefront"].append(wavefront)
                results["wavefront_meters"].append(wavefront_meters)
                if i in [1433, 775, 777, 444, 783]:
                    # print(i, sample_rate, pulse_count, bang_end, wavefront)
                    plt.subplot(len(DOWNSAMPLE_FACTORS), 1, 1 + j)
                    plt.plot(signal, '.')
                    plt.axvline(bang_end, color="red")
                    plt.axvline(wavefront, color="green")
                    plt.axhline(np.max(signal[bang_end:]) * 0.5)
                    plt.ylabel(down_fac)
            if i in [1433, 775, 777, 444, 783]:
                plt.show()

    return results


if __name__ == "__main__":

    # all_files = list(glob.glob("data/downsample_tests/export*.csv"))
    # df_from_each_file = (pd.read_csv(f, converters={"ultrasound": json.loads}) for f in all_files)
    # data = pd.concat(df_from_each_file, ignore_index=True)

    # data = filter_data(data)
    # data.to_csv("data/downsample_tests/data-filtered.csv")
    # plt.plot(data.loc[829, "ultrasound"])
    # plt.show()
    # exit()

    print("Loading data...")
    data = pd.read_csv("data/downsample_tests/data-filtered.csv", converters={"ultrasound": json.loads})

    results = run_downsample_tests(data)
    df = pd.DataFrame.from_dict(results)
    df.to_csv("data/downsample_tests/results-new.csv", index=None)

    results = pd.read_csv("data/downsample_tests/results-new.csv")
    compute_results(results)

    errors = pd.read_csv("data/downsample_tests/errors-wavefront.csv")
    display_errors(errors)
