from pathlib import Path
import pandas as pd


def load_excel_data(file: Path) -> pd.DataFrame:
    # TODO: load data from csv with all manual readings
    df = pd.DataFrame(pd.read_excel(file, sheet_name="data"))
    return df


def load_raw_ultrasound(file: Path) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv(file))
    return df
