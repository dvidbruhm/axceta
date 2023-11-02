from pathlib import Path
import pandas as pd
import numpy as np
import ast
import json


def load_excel_data(file: Path) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_excel(file, sheet_name="data"))
    return df


def load_raw_ultrasound(file: Path) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv(file))
    return df


def load_dashboard_data(file: Path) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv(file))
    df["AcquisitionTime"] = pd.to_datetime(df["AcquisitionTime"])
    return df


def load_raw_LC_data(file: Path) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_parquet(file))
    return df
    
