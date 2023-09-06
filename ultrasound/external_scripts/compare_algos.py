import pandas as pd
import matplotlib.pyplot as plt

import tools.utils as utils


def compute(file_name):
    df = pd.read_csv(file_name)

    for i in range(len(df)):
        row = df.loc[i]
        raw = utils.str_raw_data_to_list(row["raw_data"])



if __name__ == "__main__":
    
