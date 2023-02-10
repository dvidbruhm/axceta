from dataclasses import dataclass
import datetime
from pathlib import Path
import itertools
import glob
from typing import List
import typer
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from scipy import signal

import us.data as data
import us.plot as plot
import us.utils as utils
import us.algos as algos
import us.ml.utils as ml_utils
import us.forecasting.algos as f_algos

import us.ml.ml as ml

plt.style.use("dark_background")
mpl.rcParams["figure.facecolor"] = '#282A36'
mpl.rcParams["axes.facecolor"] = '#282A36'
app = typer.Typer(pretty_exceptions_show_locals=False)

# question: l'allure des 2 raw ultrasons et où arrivent les résultats du excel

# INFO : enlever le x2 (ou plutôt faire * 500khz plutôt que 1000khz) a l'air de faire plus de sense
#        pour la transformer entre la distance et l'index

# IDÉE : - ML (conv 1d pour prédire le TOF)
#        - Enveloppe pour enlever les oscillations
#        - Comparer avec température extérieure


@app.command()
def csv_to_parquet(file: Path):
    data_paths = [Path(f) for f in glob.glob(str(file))]
    print("Converting these files : ")
    print(f"[bold green]{data_paths}[/bold green]")
    for f in data_paths:
        utils.assert_csv(f)
        output_file = f"{f}.parquet"

        print(
            f"Converting csv file to parquet... \n\t Input file : [bold red]{f}[/bold red] \n\t Output file : [bold green]{output_file}[/bold green]")
        df = pd.DataFrame(pd.read_csv(f, converters={
                          "sensor_raw_data": json.loads, "AcquisitionTime": pd.to_datetime, "LC_AcquisitionTime": pd.to_datetime}))
        df.to_parquet(output_file)
        print("Done.")


@app.command()
def combine_csvs(folder_path: Path):
    all_files = [name for name in glob.glob(str(folder_path) + "/*.csv")]
    combined_data = pd.concat([pd.read_csv(f) for f in all_files])
    combined_data.to_csv(str(folder_path) + "/combined.csv")


@app.command()
def plot_raw(file: Path, excel_file: Path, index: int = 0, by_index: bool = False):

    excel_data = data.load_excel_data(excel_file)
    if by_index:
        single_excel_data = excel_data.iloc[index]
        data_file = Path(single_excel_data["filename"].replace("./", "data/"))
    else:
        single_excel_data = excel_data.loc[excel_data["filename"] == str(
            file).replace("data", ".")].squeeze()
        data_file = file
    df = data.load_raw_ultrasound(data_file)

    plot.plot_raw_ultrasound(df, single_excel_data)


@app.command()
def plot_full_excel(file: Path):
    df = data.load_excel_data(file)
    plot.plot_full_excel(df)


@app.command()
def plot_compare_raws(file: Path, indices: List[int]):
    excel_data = data.load_excel_data(file)

    excel_lines = [excel_data.iloc[idx] for idx in indices]
    files = [line["filename"].replace(".", "data", 1) for line in excel_lines]
    raws = [data.load_raw_ultrasound(file) for file in files]

    plot.plot_compare_raw(raws, excel_lines)


@app.command()
def plot_dashboard_data(volumes_path: Path, temperatures_path: Path, temperatures_extern_path: Path):
    volumes = data.load_dashboard_data(volumes_path)
    temperatures = data.load_dashboard_data(temperatures_path)
    temperatures_extern = pd.DataFrame(pd.read_csv(temperatures_extern_path))

    plot.plot_dashboard_data(volumes, temperatures, temperatures_extern)


@app.command()
def raw_ultrasound_algo(data_path: Path):
    """Takes a parquet file as input (not a csv). Use csv_to_parquet command to convert csv to parquet file"""
    df = data.load_raw_LC_data(data_path)

    df_LC_sorted = pd.DataFrame(df.sort_values(by="LC_AcquisitionTime"))
    df_sorted = pd.DataFrame(df_LC_sorted.sort_values(by="AcquisitionTime"))
    df_sorted = pd.DataFrame(df_sorted.drop_duplicates(
        subset="AcquisitionTime", keep="last")).reset_index()

    df_sorted["output_TOF_v1"] = df_sorted.apply(
        lambda x: algos.algo_v1(x["sensor_raw_data"]), axis=1)
    df_sorted["output_TOF_v1"] = ml_utils.moving_average(
        df_sorted["output_TOF_v1"].values, 5)

    df_sorted["cdm_ToF2"] = df_sorted.apply(
        lambda x: x["cdm_ToF"] + 2000 * np.interp(x["cdm_ToF"], [5000, 30000], [-1, 1]), axis=1)

    plt.plot(range(len(df_sorted)),
             df_sorted["cdm_ToF"], color="orange", label="CDM")
    # plt.plot(range(len(df_sorted)), df_sorted["cdm_ToF2"], color="blue", label="CDM")
    plt.plot(range(len(df_sorted)),
             df_sorted["wf_ToF"], color="black", label="WF")
    plt.plot(range(len(df_sorted)),
             df_sorted["LC_ToF"], color="magenta", label="LC")
    plt.plot(range(len(df_sorted)),
             df_sorted["output_TOF_v1"], color="red", label="output v1")
    plt.legend(loc="best")
    plt.show()
    # plt.plot(df_LC_sorted["LC_AcquisitionTime"], df_LC_sorted["LC_FeedRemainingM3"])
    # plt.show()

    indices_to_plot = [10, 50]
    for plot_index, i in enumerate(indices_to_plot):
        series = df_sorted.loc[i]
        plt.subplot(len(indices_to_plot), 1, plot_index+1)
        algos.algo_v1(series["sensor_raw_data"], plot=False)
        # plt.show()

        output_TOF_v1 = algos.algo_v1(series["sensor_raw_data"], plot=False)
        plt.plot(series["sensor_raw_data"], color="blue")
        plt.axvline(output_TOF_v1 / 2, color="red", label="output TOF v1")
        plt.axvline(series["LC_ToF"] / 2, color="magenta", label="LC TOF")
        plt.axvline(series["wf_ToF"] / 2, color="black", label="WF TOF")
        plt.axvline(series["cdm_ToF"] / 2, color="orange", label="CDM TOF")
        plt.title(i)
    plt.show()


@app.command()
def compute_raw_data_quality(data_path: Path):
    plt.rcParams.update({'font.size': 22})
    df = data.load_raw_LC_data(data_path)

    df_LC_sorted = pd.DataFrame(df.sort_values(by="LC_AcquisitionTime"))
    df_sorted = pd.DataFrame(df_LC_sorted.sort_values(by="AcquisitionTime"))
    df_sorted = pd.DataFrame(df_sorted.drop_duplicates(
        subset="AcquisitionTime", keep="last")).reset_index()

    df_sorted["raw_data_quality"] = df_sorted.apply(
        lambda x: algos.raw_data_quality(x["sensor_raw_data"]), axis=1)
    df_sorted["quality_color"] = df_sorted.apply(
        lambda x: algos.quality_color(x["raw_data_quality"]), axis=1)

    plt.plot(df_sorted["LC_AcquisitionTime"],
             df_sorted["LC_distanceFromWeight"], color="gray")
    plt.scatter(df_sorted["LC_AcquisitionTime"],
                df_sorted["LC_distanceFromWeight"], s=50, c=df_sorted["quality_color"])
    plt.ylabel("LoadCell distance [mm]")
    plt.xlabel("Acquisition Time")
    plt.axhline(4, color="gray", linestyle="--")
    plt.title(data_path)
    plt.show()

    print(len(df_sorted))

    indices_to_plot = [60, 150, 280, 299]
    indices_to_plot = [300, 160, 37]
    for plot_index, i in enumerate(indices_to_plot):
        series = df_sorted.loc[i]
        plt.subplot(len(indices_to_plot), 1, plot_index + 1)
        algos.raw_data_quality(series["sensor_raw_data"], plot=True)
        plt.title(i)
    plt.show()


@app.command()
def compute_raw_data_quality_batch(data_path: Path):
    print(f"Loading data from [bold green]{data_path}[/bold green]")
    df = pd.DataFrame(pd.read_csv(data_path, converters={
                      "sensor_raw_data": json.loads, "AcquisitionTime": pd.to_datetime, "LC_AcquisitionTime": pd.to_datetime}))
    df = pd.DataFrame(df.dropna(subset=["frequency"]))
    df = pd.DataFrame(df.sort_values(by="AcquisitionTime"))

    df = pd.DataFrame(df.drop_duplicates(subset=["batchId", "TVG_T0", "TVG_T1", "TVG_T2", "TVG_T3", "TVG_T4", "TVG_T5", "TVG_G0", "TVG_G1",
                      "TVG_G2", "TVG_G3", "TVG_G4", "TVG_G5", "Count", "frequency", "pulseCount", "currentLimit"], keep="first")).reset_index()

    df["raw_data_quality"] = df.apply(
        lambda x: algos.raw_data_quality(np.array(x["sensor_raw_data"])), axis=1)
    df["quality_color"] = df.apply(
        lambda x: algos.quality_color(x["raw_data_quality"]), axis=1)
    print(f"Done. There are [bold green]{len(df)}[/bold green] data points")

    config_dfs = []
    for i in range(6):
        print(f"Processing config [bold green]#{i + 1}[/bold green]")
        config = df.loc[i][["TVG_T0", "TVG_T1", "TVG_T2", "TVG_T3", "TVG_T4", "TVG_T5", "TVG_G0", "TVG_G1",
                            "TVG_G2", "TVG_G3", "TVG_G4", "TVG_G5", "Count", "frequency", "pulseCount", "currentLimit"]]
        indices = []
        for j in range(len(df)):
            params = df.loc[j][["TVG_T0", "TVG_T1", "TVG_T2", "TVG_T3", "TVG_T4", "TVG_T5", "TVG_G0", "TVG_G1",
                                "TVG_G2", "TVG_G3", "TVG_G4", "TVG_G5", "Count", "frequency", "pulseCount", "currentLimit"]]
            if params.equals(config):
                indices.append(j)
        config_df = df.loc[indices].reset_index()
        config_dfs.append(config_df)
        print(
            f"There are [bold green]{len(indices)}[/bold green] readings with config [bold green]#{i + 1}[/bold green]")

    for i, c_df in enumerate(config_dfs):
        plt.subplot(len(config_dfs), 1, i + 1)
        plt.plot(c_df["AcquisitionTime"],
                 c_df["raw_data_quality"], label=f"Config #{i}")
        plt.legend(loc="best")

    plt.show()

    config_num = 4
    indices_to_plot = [5, 36, 51, 70, 90, 110, 130, 160, 177]
    for i, j in enumerate(indices_to_plot):
        plt.subplot(len(indices_to_plot), 1, i + 1)
        plt.plot(config_dfs[config_num].loc[int(j), "sensor_raw_data"])
    plt.show()


@app.command()
def compute_raw_data_quality_excel(xls_path: Path, data_dir: Path):
    from torchvision import transforms
    import us.ml.data as data

    full_silo_dataset = data.SiloFillDatasetExcel(
        xls_file=xls_path, root_dir=data_dir, transform=None, target_name="wavefront_distance_in_mm")

    qualities = []
    quality_colors = []
    targets = []
    for i in range(len(full_silo_dataset)):
        q = algos.raw_data_quality(full_silo_dataset[i][0])
        qualities.append(q)
        quality_colors.append(algos.quality_color(q))
        targets.append(full_silo_dataset[i][1])

    plt.plot(targets, color="gray")
    plt.scatter(range(len(full_silo_dataset)), targets, s=50, c=quality_colors)
    plt.ylabel("Wavefront distance [mm]")
    plt.xlabel("Acquisition Time")
    plt.axhline(4000, color="gray", linestyle="--")
    plt.title(xls_path)
    plt.show()

    indices_to_plot = [60, 150, 280, 299]
    indices_to_plot = [300, 160, 37]
    for plot_index, i in enumerate(indices_to_plot):
        series = full_silo_dataset[i][0]
        plt.subplot(len(indices_to_plot), 1, plot_index+1)
        algos.raw_data_quality(series, plot=True)
        plt.title(i)
    plt.show()


@app.command()
def compare_sensor_configs(data_path: Path):
    df = pd.DataFrame(pd.read_csv(data_path))
    df["Error_CDM"] = abs(df["LC_FeedRemaining_kg"] -
                          df["FeedRemaining_kg_CDM"])
    df["Error_PGA_WF"] = abs(
        df["LC_FeedRemaining_kg"] - df["FeedRemaining_kg_PGA_WF"])
    config_ids = df["configID"].unique()
    silos = df["LocationName"].unique()
    print(df)
    print(df.columns)
    print(config_ids)
    print(silos)
    print(df[df["LocationName"] == silos[0]]["configID"])

    errors = []

    for silo in sorted(silos):
        silo_data = df[df["LocationName"] == silo]
        for i, config_id in enumerate(config_ids):
            config_data = silo_data[silo_data["configID"] == config_id]
            mae_cdm = round(config_data["Error_CDM"].mean())
            mae_pga_wf = round(config_data["Error_PGA_WF"].mean())
            var_cdm = round(config_data["FeedRemaining_kg_CDM"].std())
            var_pga_wf = round(config_data["FeedRemaining_kg_PGA_WF"].std())
            errors.append((silo, config_id, mae_cdm,
                          mae_pga_wf, var_cdm, var_pga_wf))
            plt.subplot(len(config_ids), 1, i + 1)
            plt.plot(config_data["LC_FeedRemaining_kg"],
                     color="green", label="LoadCell")
            plt.plot(config_data["FeedRemaining_kg_CDM"],
                     color="orange", label="CDM")
            plt.plot(config_data["FeedRemaining_kg_PGA_WF"],
                     color="blue", label="PGA WF")
            plt.title(config_id)
        plt.legend(loc="best")
        # plt.show()
    # print(errors)
    for silo in sorted(silos):
        print(f"\n-------{silo}--------\n")
        err = list(filter(lambda er: er[0] == silo, errors))
        min_i, e = min(enumerate(err), key=lambda t: (t[1][2] + t[1][4]))
        [print(f"[bold yellow]{e}[/bold yellow]") if i ==
         min_i else print(e) for i, e in enumerate(err)]


@app.command()
def CDM_window_test(data_path: Path):

    def plot_with_window(data, window, color, y=(0, 1)):
        plt.plot(data, label="raw data")
        plt.axvline(window.start_index,
                    ymin=y[0], ymax=y[1], color=color, lw=3)
        plt.axvline(window.start_index + window.width,
                    ymin=y[0], ymax=y[1], color=color, lw=3)
        plt.legend(loc="best")

    """
    df = pd.DataFrame(pd.read_parquet(data_path))
    i = 2700
    test_data = df.loc[i, "sensor_raw_data"]
    cdm = df.loc[i, "cdm_ToF"]
    computed_cdm = algos.compute_cdm(test_data[2500:], 2500)
    plt.plot(test_data)
    plt.axvline(computed_cdm, color="red")
    plt.axvline(cdm/2, color="green")
    plt.show()
    """

    silo_nb = data_path.name.split(".")[0][:-1]
    silo_a_path = Path(f"{data_path.parent}/{silo_nb}A.csv")
    silo_b_path = Path(f"{data_path.parent}/{silo_nb}B.csv")
    silo_a = pd.DataFrame(pd.read_csv(silo_a_path, converters={"rawData": json.loads, "AcquisitionTime": pd.to_datetime}))
    silo_b = pd.DataFrame(pd.read_csv(silo_b_path, converters={"rawData": json.loads, "AcquisitionTime": pd.to_datetime}))
    # silo_a = df[df["LocationName"] == f"Avinor-{silo_nb}A"].reset_index()
    # silo_b = df[df["LocationName"] == f"Avinor-{silo_nb}B"].reset_index()
    # silo_a = silo_a.sort_values(by="AcquisitionTime")

    window_sizes = [8000, 5000, 4000, 3000]
    for ws in window_sizes:
        cdms = []
        for i in range(len(silo_a)):
            cdm, _ = algos.compute_cdm_in_window(
                np.array(silo_a.loc[i, "rawData"]), 2500, ws)
            cdms.append(cdm)
        cdms = f_algos.moving_average(cdms, 10)
        silo_a[f"ToF_computed_cdm_{ws}"] = cdms
        silo_a[f"Error_ToF_computed_cdm_{ws}"] = abs(
            silo_a[f"ToF_computed_cdm_{ws}"] - (silo_a["LoadCells_ToF"]/2))
        mae = silo_a[f"Error_ToF_computed_cdm_{ws}"].mean()
        print(f"CDM mae {ws} : {mae}")
    silo_a["Error_ToF_cdm"] = abs(
        (silo_a["cdm_ToF"]/2) - (silo_a["LoadCells_ToF"]/2))
    mae = silo_a[f"Error_ToF_cdm"].mean()
    print(f"CDM mae : {mae}")

    """
    cdms = []
    for i in range(len(silo_a)):
        cdm = algos.compute_cdm(np.array(silo_a.loc[i, "rawData"][2500:]), 2500)
        cdms.append(cdm)
    cdms = f_algos.moving_average(cdms, 10)
    silo_a["ToF_computed_cdm_full"] = cdms
    """

    plt.plot(silo_a["LoadCells_ToF"] / 2, label="LC")
    plt.plot(f_algos.moving_average(silo_a["cdm_ToF"], 10) / 2, label="CDM")
    for ws in window_sizes:
        plt.plot(silo_a[f"ToF_computed_cdm_{ws}"], label=f"CDM_{ws}")
    plt.legend(loc="best")
    plt.show()

    # test_data = np.array(silo_a.loc[210, "rawData"])
    # computed_cdm = algos.compute_cdm(test_data[2500:], 2500)

    colors = ["red", "orange", "green"]
    ys = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    plt.subplot(3, 1, 1)
    i = 430
    test_data = np.array(silo_a.loc[i, "rawData"])
    for ws, color, y in zip(window_sizes, colors, ys):
        cdm, w = algos.compute_cdm_in_window(test_data, 2500, ws)
        plot_with_window(test_data, w, color, y)
        plt.axvline(cdm, color=color)
    cdm_full = algos.compute_cdm(test_data[2500:], 2500)
    plt.axvline(cdm_full, color="black")
    plt.axvline(silo_a.loc[i, "LoadCells_ToF"] / 2, color="blue")

    plt.subplot(3, 1, 2)
    i = 450
    test_data = np.array(silo_a.loc[i, "rawData"])
    for ws, color, y in zip(window_sizes, colors, ys):
        cdm, w = algos.compute_cdm_in_window(test_data, 2500, ws)
        plot_with_window(test_data, w, color, y)
        plt.axvline(cdm, color=color)
    cdm_full = algos.compute_cdm(test_data[2500:], 2500)
    plt.axvline(cdm_full, color="black")
    plt.axvline(silo_a.loc[i, "LoadCells_ToF"] / 2, color="blue")

    plt.subplot(3, 1, 3)
    i = 465
    test_data = np.array(silo_a.loc[i, "rawData"])
    for ws, color, y in zip(window_sizes, colors, ys):
        cdm, w = algos.compute_cdm_in_window(test_data, 2500, ws)
        plot_with_window(test_data, w, color, y)
        plt.axvline(cdm, color=color)
    cdm_full = algos.compute_cdm(test_data[2500:], 2500)
    plt.axvline(cdm_full, color="black")
    plt.axvline(silo_a.loc[i, "LoadCells_ToF"] / 2, color="blue")
    plt.show()


@app.command()
def compare_error_metrics(data_path1: Path, data_path2: Path):

    from rich.console import Console
    from rich.table import Table

    full_data1 = pd.DataFrame(pd.read_csv(data_path1, converters={"AcquisitionTime": pd.to_datetime}))
    full_data2 = pd.DataFrame(pd.read_csv(data_path2, converters={"AcquisitionTime": pd.to_datetime}))

    min_peak_str = 0
    full_data1 = full_data1[full_data1["peakStrength"] > min_peak_str]

    @dataclass
    class ErrorInfo:
        mae: float
        occurences: int

    def compute_metrics(full_data):
        # if len(full_data["configID"].unique()) > 1:
        #    full_data = full_data[full_data["configID"] == "log-1"].reset_index()
        full_data["LC_FeedRemaining_kg"] = full_data["LC_FeedRemaining_kg"] / 1000
        full_data["FeedRemaining_kg_CDM"] = full_data["FeedRemaining_kg_CDM"] / 1000
        full_data["sound_speed"] = full_data.apply(lambda x: utils.temp_to_sound_speed(x["temperature"]), axis=1)
        full_data["dist_cdm"] = full_data.apply(lambda x: utils.tof_to_dist(x["cdm_ToF"], x["sound_speed"]), axis=1)
        full_data["dist_lc"] = full_data.apply(lambda x: utils.tof_to_dist(x["LoadCells_ToF"], x["sound_speed"]), axis=1)

        errors = []
        data = full_data
        error_cdm_kg = abs(data["LC_FeedRemaining_kg"] - data["FeedRemaining_kg_CDM"])
        full_mae = error_cdm_kg.mean()
        full_info = ErrorInfo(mae=round(full_mae, 2),
                              occurences=len(full_data))
        steps = range(0, 10)
        for i in range(1, len(steps)):
            t1 = steps[i - 1]
            t2 = steps[i]
            d = data[(data["dist_lc"] > t1) & (data["dist_lc"] < t2)]
            error_cdm = abs(d["LC_FeedRemaining_kg"] - d["FeedRemaining_kg_CDM"])
            d_mae = error_cdm.mean()
            error_info = ErrorInfo(mae=round(d_mae, 2), occurences=len(d))
            errors.append(error_info)
            # print(f"| {t1:1.0f} m - {t2:1.0f} m | mean error: {d_mae:5.2f} |")

        return errors, full_info

    errors1, full1 = compute_metrics(full_data1)
    all_configs = full_data2["configID"].unique()
    for conf in all_configs:
        config_data = full_data2[full_data2["configID"] == conf].reset_index()
        if len(config_data) < 1:
            continue
        errors2, full2 = compute_metrics(config_data)

        table = Table(
            title=f"Error comparison between old and new config ([green]{conf}[/green]), with min amplitude of [green]{min_peak_str}[/green]")
        color = "cyan"
        table.add_column("Depth (m)", justify="center", style="yellow")
        table.add_column("Nb data", justify="center", style=color)
        table.add_column("Old error (t)", justify="center", style=color)
        table.add_column("Nb data", justify="center", style=color)
        table.add_column("New error (t)", justify="center", style=color)
        table.add_column("Difference (t)", justify="center")
        for i in range(len(errors1)):
            e1 = errors1[i].mae
            e2 = errors2[i].mae
            diff = round(e1 - e2, 2)
            color = "red" if diff < 0 else "green"
            table.add_row(f"{i} -> {i+1}", f"{errors1[i].occurences}", f"{e1}",
                          f"{errors2[i].occurences}", f"{e2}", f"[bold {color}]{diff}[/bold {color}]")

        table.add_row("all", f"{full1.occurences}", f"{full1.mae}",
                      f"{full2.occurences}", f"{full2.mae}", f"{round(full1.mae - full2.mae, 2)}")
        console = Console()
        console.print(table)


@app.command()
def compare_wavefronts(data_path: Path):
    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime})
    data = data[data["LocationName"] == "Avinor-1485A"]

    data["sound_speed"] = data.apply(lambda x: utils.temp_to_sound_speed(x["temperature"]), axis=1)
    data["dist_pga_wf"] = utils.tof_to_dist(data["PGA_wf_ToF"], data["sound_speed"])
    data["dist_wf"] = utils.tof_to_dist(data["wf_ToF"], data["sound_speed"])
    data["dist_loadcell"] = utils.tof_to_dist(data["LoadCells_ToF"], data["sound_speed"])
    data["dist_cdm"] = utils.tof_to_dist(data["cdm_ToF"], data["sound_speed"])

    plt.plot(data["AcquisitionTime"], data["dist_pga_wf"], label="PGA_WF")
    plt.plot(data["AcquisitionTime"], data["dist_wf"], label="WF")
    plt.plot(data["AcquisitionTime"], data["dist_loadcell"], label="LoadCell")
    plt.legend(loc="best")
    plt.show()

    short_data = data[data["dist_loadcell"] < 2]
    print(short_data)
    plt.plot(short_data["AcquisitionTime"], short_data["dist_pga_wf"], label="PGA_WF")
    plt.plot(short_data["AcquisitionTime"], short_data["dist_wf"], label="WF")
    plt.plot(short_data["AcquisitionTime"], short_data["dist_loadcell"], label="LoadCell")
    plt.plot(short_data["AcquisitionTime"], short_data["dist_cdm"], label="CDM")
    plt.legend(loc="best")
    plt.show()

    plt.plot(short_data["AcquisitionTime"], short_data["FeedRemaining_kg_PGA_WF"], label="PGA_WF")
    plt.plot(short_data["AcquisitionTime"], short_data["FeedRemaining_kg_WF"], label="WF")
    plt.plot(short_data["AcquisitionTime"], short_data["LC_FeedRemaining_kg"], label="LoadCell")
    plt.plot(short_data["AcquisitionTime"], short_data["FeedRemaining_kg_CDM"], label="CDM")
    plt.legend(loc="best")
    plt.show()

    print(data.columns)


@app.command()
def signal_separation(data_path: Path):
    from sklearn.decomposition import FastICA
    indices = list(range(400, 430))
    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime, "rawData": json.loads})
    data = data[data["LocationName"] == "Avinor-1485B"].reset_index()
    raw_signals = [np.array(data.loc[i, "rawData"])[2500:30600] for i in indices]
    raw_signals = np.stack([signal.resample_poly(raw_signal.astype(float), 1, 10) for raw_signal in raw_signals], axis=1)
    print(raw_signals.shape)

    ica = FastICA(n_components=1, whiten="arbitrary-variance")
    raw_signals_ = ica.fit_transform(raw_signals)
    print(raw_signals_.shape)

    plt.subplot(3, 1, 1)
    plt.plot(raw_signals)
    plt.subplot(3, 1, 2)
    raw_signals_ = raw_signals_.squeeze() + 0.018
    plt.plot(raw_signals_)
    plt.subplot(3, 1, 3)
    sample = np.array(data.loc[60, "rawData"][2500:30600])
    sample = signal.resample_poly(sample.astype(float), 1, 10)
    print(raw_signals_)
    print(sample)
    plt.plot(sample)
    plt.plot(sample - raw_signals_)
    plt.show()


@app.command()
def test_custom_wavefront(data_path: Path):
    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime, "rawData": json.loads})
    data = data[data["LocationName"] == "Avinor-1483A"].reset_index()

    data = data
    data["custom_wf"] = data.apply(lambda x: algos.wavefront(x["rawData"], x["temperature"], 0.5, 0.5), axis=1)

    plt.plot(data["wf_ToF"] / 2, ".", label="WF")
    plt.plot(data["cdm_ToF"] / 2, ".", label="CDM")
    plt.plot(data["PGA_wf_ToF"] / 2, ".", label="PGA_WF")
    plt.plot(data["custom_wf"], "-", label="CustomWF")
    plt.legend(loc="best")
    plt.show()

    indices = [407, 408, 409]
    plt_count = 1
    for i in indices:
        plt.subplot(len(indices), 1, plt_count)
        plt.plot(data.loc[i, "rawData"])
        plt.axvline(data.loc[i, "PGA_wf_ToF"] / 2, color="blue", label="pga_wf")
        plt.axvline(data.loc[i, "wf_ToF"] / 2, color="red", label="wf")
        plt.axvline(data.loc[i, "custom_wf"], color="green", label="custom_wf")
        plt.axvline(data.loc[i, "cdm_ToF"] / 2, color="orange", label="cdm")
        plt.legend(loc="best")
        plt_count += 1
    plt.show()


@app.command()
def test_custom_wavefront_2(data_path: Path):
    import tools.wavefront as wf

    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime, "rawData": json.loads})
    data = data.dropna()
    data = data[data["LocationName"] == "Isoporc-6045"]
    data = data[data["config_id"].str.contains("log-1-")].reset_index()

    def find_pulse_count(x):
        i = x["config_id"].find("-p")
        if i == -1:
            return 31
        else:
            return int(str(x["config_id"])[i+2:])
    data["pulse_count"] = data.apply(lambda x: find_pulse_count(x), axis=1)
    print(data["config_id"].unique())
    print(data["pulse_count"].unique())
    print(data.columns)
    print(data["LocationName"].unique())

    #data = data[data["LocationName"] == "CDPQA-001"].reset_index()
    data["custom_wf"] = data.apply(lambda x: wf.wavefront(x["rawData"], x["temperature"], 0.5, 0.5, x["pulse_count"]), axis=1)
    data["mainbang_index"] = data.apply(lambda x: wf.detect_main_bang_end(x["rawData"], x["pulse_count"]), axis=1)

    plt.plot(data["CDMIndex"] / 2, ".", label="cdm")
    plt.plot(data["WFIndex"] / 2, ".", label="wf")
    plt.plot(data["custom_wf"], ".", label="custom wf")
    plt.legend(loc="best")
    plt.show()

    indices = [11, 1002, 1003, 1004, 1005, 1006]
    plt_count = 1
    for i in indices:
        print(data.loc[i, "LocationName"])
        print(data.loc[i, "WFIndex"] / 2)
        print(data.loc[i, "custom_wf"])
        d = data.loc[i]
        print(wf.wavefront(d["rawData"], d["temperature"], 0.5, 0.5, 31))
        plt.subplot(len(indices), 1, plt_count)
        plt.plot(data.loc[i, "rawData"])
        plt.axvline(data.loc[i, "WFIndex"] / 2, color="red", label="wf")
        plt.axvline(data.loc[i, "custom_wf"], color="green", label="custom_wf")
        plt.axvline(data.loc[i, "CDMIndex"] / 2, color="orange", label="cdm")
        plt.axvline(data.loc[i, "mainbang_index"], color="blue", label="mainbang_end")
        plt.legend(loc="best")
        plt_count += 1
    plt.show()

# ML ---------


@ app.command()
def train_excel(
        data_path: Path, xls_path: Path, epochs: int, learning_rate: float, batch_size: int, loss_fn: str, kernels: List[int],
        train_size: float, small_dataset: bool = False):
    ml.train_excel(data_path, xls_path, epochs, learning_rate,
                   batch_size, kernels, loss_fn, train_size, small_dataset)


@ app.command()
def train_parquets(
        data_path: str, epochs: int, learning_rate: float, batch_size: int, loss_fn: str, kernels: List[int],
        train_size: float, small_dataset: bool = False):
    data_paths = [Path(f) for f in glob.glob(str(data_path))]
    ml.train_parquet(data_paths, epochs, learning_rate,
                     batch_size, kernels, loss_fn, train_size, small_dataset)


@ app.command()
def hyperparam_search_parquets(data_path: str, train_size: float):
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    data_paths = [Path(f) for f in glob.glob(str(data_path))]
    silo_dataset, dataset_params = ml.pretrain_parquet(
        data_paths, train_size, False)

    lrs = [0.00001, 0.000001, 0.0000001]
    epochs = [10, 50]
    bs = [16, 64]
    kernels = [[1, 1, 1, 1, 1], [1, 2, 4, 8, 16],
               [1, 4, 8, 16, 32], [1, 8, 16, 32, 64]]
    loss_fns = ["mse", "rmsle"]

    num = 0
    for lr, epoch, batch_size, kernel, loss_fn in itertools.product(lrs, epochs, bs, kernels, loss_fns):
        if num > 68:
            ml.train(silo_dataset, dataset_params, epoch, lr, batch_size, kernel,
                     loss_fn, train_size, data_paths, "silo_tof_parquet_search", verbose=False)
        num += 1


@ app.command()
def viz_excel(xls_path: Path, data_path: Path, model_path: Path):
    ml.viz_excel(xls_path, data_path, model_path)


@ app.command()
def viz_parquets(model_path: Path = Path(), parquet_path: Path = Path()):
    ml.viz_parquets(model_path, parquet_path)


@ app.command()
def test_regression(data_path: Path):
    data = pd.read_csv(data_path, converters={"AcquisitionTime": pd.to_datetime, "rawData": json.loads})
    print(data.columns)
    data = data.drop_duplicates(subset='AcquisitionTime', keep="first")
    plt.plot(data["AcquisitionTime"], data["filledVolume_WF_Dave"])
    plt.show()

    values = data["filledVolume_WF_Dave"].values
    time = data["AcquisitionTime"].values

    v = []
    t = []
    for current_x in range(300, 500):
        values_prev = values[current_x-48:current_x+1]
        time_prev = time[current_x-48:current_x+1]
        smoothed_val = algos.auto_regression_smoothing(time_prev, values_prev)
        smoothed_time = time[current_x]
        v.append(smoothed_val)
        t.append(smoothed_time)
    plt.plot(time, values)
    plt.plot(t, v, ".")
    plt.show()

    time = time.astype(np.float)
    time = (time - np.min(time)) / (np.max(time) - np.min(time))

    def reg(x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        y_reg = m * x + c
        r2 = algos.r_squared(y, y_reg)
        return y_reg, r2

    max_reg_len = 48
    min_r2_score = 0.8

    x = time
    y = values

    x_smoothed = []
    y_smoothed = []
    for current_x in range(300, 500):
        print(current_x)
        best_r2 = 0
        best_xreg = []
        best_yreg = []
        best_y_reg = []
        for i in range(current_x - max_reg_len, current_x - 5):
            xreg = x[i:current_x]
            yreg = y[i:current_x]

            y_reg, r2 = reg(xreg, yreg)
            print(i, max_reg_len, current_x, r2, best_r2, len(xreg))
            if r2 > best_r2:
                print("best found")
                best_r2 = r2
                best_xreg = xreg
                best_yreg = yreg
                best_y_reg = y_reg

        x_smoothed.append(best_xreg[-1])
        smoothed_value = best_yreg[-1]
        if best_r2 > min_r2_score:
            smoothed_value = best_y_reg[-1] * 0.9 + best_yreg[-1] * 0.1
        smoothed_value = max(0, smoothed_value)
        y_smoothed.append(smoothed_value)

        print(best_r2)
        #plt.plot(x, y)
        #plt.plot(best_xreg, best_y_reg)
        #plt.plot(best_xreg, best_yreg, '.')
        # plt.show()
    plt.plot(x, y)
    plt.plot(x_smoothed, y_smoothed, '.')
    plt.show()

    xs, ys = algos.split_silo_fill(time, values, threshold=0.2)
    plt.subplot(2, 1, 1)
    for x, y in zip(xs, ys):
        plt.plot(x, y, ".")
        x = x[:int(len(x) * 0.5)]
        y = y[:int(len(y) * 0.5)]

        y_reg, r2 = reg(x, y)
        print(r2)

        plt.plot(x, y_reg)

    plt.subplot(2, 1, 2)
    plt.plot(data["AcquisitionTime"], data["filledVolume_WF_Dave"])
    plt.show()


# ---------------

# Timeseries Forecasting -> prediction of next silo fill date

@ app.command()
def predict_next_fill(silo_nb: str, data_path: Path = Path("data/Timeseries/Loadcell_v3.csv")):

    silo_name_a = f"Avinor-{silo_nb}A"
    silo_name_b = f"Avinor-{silo_nb}B"
    dfA = f_algos.read_load_cell_data(data_path, silo_name_a, version="v2")
    dfB = f_algos.read_load_cell_data(data_path, silo_name_b, version="v2")

    # split_date = "2022-10-05 12:00:00"
    # dfA = dfA.loc[dfA.index <= split_date]
    # dfB = dfB.loc[dfB.index <= split_date]

    not_both_flat = f_algos.find_not_both_flat(
        dfA, dfB, min_len=48, debug=False)
    big_cycles = f_algos.find_big_cycles(
        not_both_flat, start_at="end", debug=False)

    current_df = f_algos.get_currently_active(
        dfA, dfB, silo_name_a, silo_name_b)
    current_silo_name = current_df.columns[0]
    print(
        f"Currently active silo : [bold green]{current_silo_name}[/bold green]")

    fill_indices = f_algos.find_silo_fills(
        current_df[current_silo_name].values)
    fill_rate_info = f_algos.get_fill_rate_info(
        fill_indices, current_df, current_silo_name, big_cycles)
    print(fill_rate_info)
    for fri in fill_rate_info:
        print(fri.big_cycle_nb, fri.rate_per_unit)
    current_big_cycle = len(big_cycles) - 1
    if current_big_cycle == fill_rate_info[-1].big_cycle_nb:
        # regression
        pass
    elif current_big_cycle > fill_rate_info[-1].big_cycle_nb:
        bc = 0
        next_bc = 1
        rates = []
        for fri in fill_rate_info:
            if fri.big_cycle_nb == next_bc:
                rates.append(fri.rate_per_unit)
                next_bc += 1
        pred_rate = np.mean(rates)
        current_fill = current_df[current_silo_name].values[-1]
        nb_hours_before_empty = round(current_fill / pred_rate)
        pred_empty_date = current_df.index[-1] + \
            pd.Timedelta(hours=nb_hours_before_empty)
        print("\n[green]------------------------------------------------[/green]\n")
        print(
            f"Current date : [bold green]{current_df.index[-1]}[/bold green]")
        print(
            f"Predicted number hours before empty : [bold green]{nb_hours_before_empty}[/bold green]")
        print(
            f"Predicted empty date : [bold green]{pred_empty_date}[/bold green]")
        plt.plot(current_df[current_silo_name])
        plt.axvline(pred_empty_date)
        plt.show()


@ app.command()
def viz_loadcell_data(silo_nb: str, data_path: Path = Path("data/Timeseries/Loadcell_v3.csv")):

    train_size = 0.4

    silo_name_a = f"Avinor-{silo_nb}A"
    silo_name_b = f"Avinor-{silo_nb}B"

    dfA = f_algos.read_load_cell_data(data_path, silo_name_a, version="v2")
    dfB = f_algos.read_load_cell_data(data_path, silo_name_b, version="v2")
    not_both_flat = f_algos.find_not_both_flat(
        dfA, dfB, min_len=48, debug=False)
    big_cycles = f_algos.find_big_cycles(
        not_both_flat, start_at="end", debug=True)

    current_df = dfA
    current_silo_name = silo_name_a
    x_min = current_df.index[0]
    x_max = current_df.index[-1]

    for i, (silo_name, df) in enumerate([(silo_name_a, dfA), (silo_name_b, dfB)]):
        plt.subplot(3, 1, i+1)
        plt.xlim([x_min, x_max])
        plt.title(silo_name)
        plt.plot(df[silo_name])
        plt.plot(df[np.array(not_both_flat) == True].iloc[:, 0], '.')
        for cycle in big_cycles:
            plt.axvline(df.index[cycle.start+1], color="green")
            plt.axvline(df.index[cycle.end-1], color="orange")
        fill_indices = f_algos.find_silo_fills(df[silo_name].values)
        for index in fill_indices:
            plt.axvline(df.index[index], color="gray", linestyle="--")

    is_flat = f_algos.find_flat_signals(
        current_df[current_silo_name].values, debug=False)
    df = current_df[np.array(is_flat) == False]
    fill_indices = f_algos.find_silo_fills(df[current_silo_name].values)
    # fill_indices.insert(0, 50)
    print(fill_indices)
    fill_rate_info = f_algos.get_fill_rate_info(
        fill_indices, df, current_silo_name, big_cycles)
    print(fill_rate_info)

    plt.subplot(3, 1, 3)
    plt.xlim([x_min, x_max])

    plt.plot(df[current_silo_name], '.', color="blue")
    for index in fill_indices:
        plt.axvline(df.index[index])

    # TRAIN DATA
    """
    plt.subplot(4, 1, 4)
    plt.xlim([x_min, x_max])
    #plt.plot(train_df[silo_nb], '.', color="green", label="Train")
    #plt.plot(val_df[silo_nb], '.', color="orange", label="Val")
    plt.legend(loc="best")
    """
    # Simple regression
    """
    nb_points = 24

    for split in [700, 750, 800, 1025]:
        preds, empty_date = f_algos.simple_regression(current_df[silo_nb][:split].values, nb_points, debug=False)
        plt.plot(current_df[silo_nb][split-nb_points:split].index, preds[:nb_points], color="yellow")
        plt.plot(current_df[silo_nb][split-1:split-nb_points+len(preds)-1].index, preds[nb_points:], color="red")

    preds, empty_date = f_algos.simple_regression(train_df[silo_nb].values, nb_points, debug=False)
    plt.plot(current_df[silo_nb][len(train_df)-nb_points:len(train_df)].index, preds[:nb_points], color="yellow")
    plt.plot(current_df[silo_nb][len(train_df)-1:len(train_df)-nb_points+len(preds)-1].index, preds[nb_points:], color="red")

    # Algo based on previous rates
    is_flat = f_algos.find_flat_signals(train_df[silo_nb].values)
    train_df = train_df[np.array(is_flat) == False]
    fill_indices = f_algos.find_silo_fills(train_df[silo_nb].values)
    fill_indices.insert(0, 50)
    fill_rate_info = f_algos.get_fill_rate_info(fill_indices, train_df, silo_nb)

    print(fill_rate_info)
    pred = f_algos.predict_next_fill(fill_rate_info, train_df, silo_nb)
    plt.axvline(pred, color="yellow")

    """
    for index in fill_indices:
        plt.axvline(df.index[index])
    plt.show()


@ app.command()
def darts_lib_test(silo_name: str, data_path: Path = Path("data/Timeseries/Load_Cells_Levels.csv")):
    from darts.timeseries import TimeSeries
    from darts.models.forecasting.auto_arima import AutoARIMA
    from darts.models.forecasting.arima import ARIMA
    from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
    from darts.models.forecasting.prophet_model import Prophet
    from darts.models.forecasting.gradient_boosted_model import LightGBMModel
    from darts.models.filtering.kalman_filter import KalmanFilter
    from darts.models.forecasting.transformer_model import TransformerModel

    df = pd.DataFrame()
    for i, silo_name in enumerate(["Avinor-1485B", "Avinor-1485A"]):
        df = f_algos.read_load_cell_data(data_path, silo_name)
        plt.subplot(2, 1, i+1)
        plt.plot(df[silo_name], ".", color="gray")
        plt.plot(abs(df[silo_name].diff()))
        # df[abs(df[silo_name].diff()) < 0.0001] = np.nan
        series = TimeSeries.from_dataframe(df)
        series.plot(linewidth=3)

    plt.show()

    # series, _ = series.split_before(pd.Timestamp("20220730"))
    _, series = series.split_after(pd.Timestamp("20220715"))

    train, val = series.split_before(0.45)
    train.plot()
    val.plot()
    for index in fill_indices:
        plt.axvline(df.index[index], color="green")
    plt.show()

    exit()
    models = [ARIMA(), AutoARIMA(), ExponentialSmoothing(), Prophet(), LightGBMModel(
        lags=72), TransformerModel(input_chunk_length=24, output_chunk_length=12)]
    for i, model in enumerate(models):
        plt.subplot(len(models), 1, i+1)
        model.fit(train)
        preds = model.predict(len(val))
        series.plot(label="actual")
        preds.plot(label="predicted")
        plt.legend()
    plt.show()
