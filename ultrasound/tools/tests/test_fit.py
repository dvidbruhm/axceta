import pandas as pd
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
import tools.ultrasound_algos as ua


def str_raw_data_to_list(raw_data):
    return list(map(int, raw_data.strip('][').replace('"', '').split(',')))


if __name__ == "__main__":
    df = pd.read_csv("data/paquette/Paquette5-16 _06-28_to_07-20.csv")
    print(df.columns)

    i = 10
    row = df.iloc[i]
    raw = str_raw_data_to_list(row["rawdata"])
    pc = row["pulseCount"]
    sr = 20000

    bang_end = ua.detect_main_bang_end(raw, pc, sr)
    peak_max = max(raw[bang_end:])

    plt.plot(raw)
    plt.axvline(bang_end)
    plt.show()
