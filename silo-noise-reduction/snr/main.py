import typer
from pathlib import Path

#from snr import data, plot, filters
import snr.data as data
import snr.plot as plot
import snr.filters as filters
import snr.utils as utils


app = typer.Typer()


@app.command()
def viz():
    global max_silo_index, current_silo_index
    df = data.load_silo_data(file_path=Path("data", "silo-data-2.csv"))
    conversion_df = data.load_dist_to_volume_data(file_path=Path("data", "dist_to_volume.csv"))
    plot.plot_data_streamlit(df)
    utils.dist_to_volume(5.34, "Avinor-1483A", conversion_df)


viz()
