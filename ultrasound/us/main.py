from pathlib import Path
from typing import List
import typer
import pandas as pd
import json
import matplotlib.pyplot as plt
from rich import print

import us.data as data
import us.plot as plot
import us.utils as utils
import us.algos as algos
import us.ml.utils as ml_utils

import us.ml.ml as ml

app = typer.Typer(pretty_exceptions_show_locals=False)

# question: l'allure des 2 raw ultrasons et où arrivent les résultats du excel

# INFO : enlever le x2 (ou plutôt faire * 500khz plutôt que 1000khz) a l'air de faire plus de sense
#        pour la transformer entre la distance et l'index

# IDÉE : - ML (conv 1d pour prédire le TOF)
#        - Enveloppe pour enlever les oscillations
#        - Comparer avec température extérieure


@app.command()
def csv_to_parquet(file: Path):
    utils.assert_csv(file)
    output_file = f"{file}.parquet"

    print(f"Converting csv file to parquet... \n\t Input file : [bold red]{file}[/bold red] \n\t Output file : [bold green]{output_file}[/bold green]")
    df = pd.DataFrame(pd.read_csv(file, converters={"sensor_raw_data": json.loads, "AcquisitionTime": pd.to_datetime, "LC_AcquisitionTime": pd.to_datetime}))
    df.to_parquet(output_file)
    print("Done.")


@app.command()
def plot_raw(file: Path, excel_file: Path):
    df = data.load_raw_ultrasound(file)

    excel_data = data.load_excel_data(excel_file)
    single_excel_data = excel_data.loc[excel_data["filename"] == str(file).replace("data", ".")].squeeze()

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


import numpy as np
@app.command()
def raw_ultrasound_algo(data_path: Path):
    """Takes a parquet file as input (not a csv). Use csv_to_parquet command to convert csv to parquet file"""
    df = data.load_raw_LC_data(data_path)

    df_LC_sorted = pd.DataFrame(df.sort_values(by="LC_AcquisitionTime"))
    df_sorted = pd.DataFrame(df_LC_sorted.sort_values(by="AcquisitionTime"))
    df_sorted = pd.DataFrame(df_sorted.drop_duplicates(subset="AcquisitionTime", keep="last")).reset_index()
    
    df_sorted["output_TOF_v1"] = df_sorted.apply(lambda x: algos.algo_v1(x["sensor_raw_data"]), axis=1)
    df_sorted["output_TOF_v1"] = ml_utils.moving_average(df_sorted["output_TOF_v1"].values, 5)
    
    df_sorted["cdm_ToF2"] = df_sorted.apply(lambda x: x["cdm_ToF"] + 2000 * np.interp(x["cdm_ToF"], [5000, 30000], [-1, 1]), axis=1)

    plt.plot(range(len(df_sorted)), df_sorted["cdm_ToF"], color="orange", label="CDM")
    plt.plot(range(len(df_sorted)), df_sorted["cdm_ToF2"], color="blue", label="CDM")
    plt.plot(range(len(df_sorted)), df_sorted["wf_ToF"], color="black", label="WF")
    plt.plot(range(len(df_sorted)), df_sorted["LC_ToF"], color="magenta", label="LC")
    plt.plot(range(len(df_sorted)), df_sorted["output_TOF_v1"], color="red", label="output v1")
    plt.legend(loc="best")
    plt.show()
    #plt.plot(df_LC_sorted["LC_AcquisitionTime"], df_LC_sorted["LC_FeedRemainingM3"])
    #plt.show()

    

    indices_to_plot = [10, 50]
    for plot_index, i in enumerate(indices_to_plot):
        series = df_sorted.loc[i]
        plt.subplot(len(indices_to_plot), 1, plot_index+1)
        algos.algo_v1(series["sensor_raw_data"], plot=False)
        #plt.show()

        
        output_TOF_v1 = algos.algo_v1(series["sensor_raw_data"], plot=False)
        plt.plot(series["sensor_raw_data"], color="blue")
        plt.axvline(output_TOF_v1 / 2, color="red", label="output TOF v1")
        plt.axvline(series["LC_ToF"] / 2, color="magenta", label="LC TOF")
        plt.axvline(series["wf_ToF"] / 2, color="black", label="WF TOF")
        plt.axvline(series["cdm_ToF"] / 2, color="orange", label="CDM TOF")
        plt.title(i)
    plt.show()





### ML

@app.command()
def train(data_path: Path, xls_path: Path, epochs: int, learning_rate: float, batch_size: int, kernels: List[int], train_size: float, small_dataset: bool = False):
    ml.train(data_path, xls_path, epochs, learning_rate, batch_size, kernels, train_size, small_dataset)

@app.command()
def viz(model_path: Path = Path()):
    ml.viz(model_path)
