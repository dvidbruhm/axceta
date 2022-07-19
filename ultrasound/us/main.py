from pathlib import Path
import typer

import us.data as data
import us.plot as plot


app = typer.Typer()


@app.command()
def plot_raw(file: Path):
    df = data.load_raw_ultrasound(file)
    plot.plot_raw_ultrasound(df)
