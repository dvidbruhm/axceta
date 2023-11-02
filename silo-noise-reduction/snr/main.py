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
    data_path = Path("data", "silo-data-3.csv")
    df = data.load_silo_data(file_path=data_path, scale_dist=True)
    #plot.plot_silo_matplotlib(df, "CDPQA-024")

    conversion_df = data.load_dist_to_volume_data(file_path=Path("data", "dist_to_volume.csv"))
    plot.plot_data_streamlit(df, conversion_df)

@app.command()
def dist_to_volume():
    data_path = Path("data", "silo-data-3.csv")
    conversion_df = data.load_dist_to_volume_data(file_path=Path("data", "dist_to_volume.csv"))
    data.add_percent_filled_to_data(data_path, conversion_df)

viz()
