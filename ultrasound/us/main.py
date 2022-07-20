from pathlib import Path
from typing import List
import typer
import pandas as pd

import us.data as data
import us.plot as plot

app = typer.Typer(pretty_exceptions_show_locals=False)


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
