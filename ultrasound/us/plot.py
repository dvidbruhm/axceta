import matplotlib.pyplot as plt
import pandas as pd

import us.algos as algos


def plot_raw_ultrasound(df: pd.DataFrame, manual_data: pd.DataFrame):
    values = df["ultrasons_data"]
    plt.plot(values)

    noise_threshold = algos.NoiseThresholdv1().process(values)
    results1 = algos.MainBangDetectorv1().process(values, noise_threshold)
    plt.axvline(results1["main_bang_start"], color="black")
    plt.axvline(results1["main_bang_end"], color="black")

    results2 = algos.CenterOfMassLin().process(values, noise_threshold)
    plt.axvline(results2, color="red")

    results3 = algos.CenterOfMassQuad().process(values, noise_threshold)

    print(noise_threshold)
    print(results1)
    print(results2)
    print(results3)


    plt.show()
