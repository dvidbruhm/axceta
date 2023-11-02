import pandas as pd
import matplotlib.pyplot as plt
import tools.ultrasound_algos as algos
import tools.utils as utils


def test():
    df = pd.read_csv("data/random/export-2.csv")
    print(df)
    df["raw_data"] = df.apply(lambda row: utils.str_raw_data_to_list(row["raw_data"]), axis=1)
    print(df)
    row = df.loc[0]
    raw = row["raw_data"]
    be = algos.detect_main_bang_v2(raw, row["samplingFrequency"])
    wf = algos.wavefront_empty_and_full_detection(raw, 0.5, row["pulseCount"], row["samplingFrequency"], 29834 / 50)
    print(f"wf: {wf}, bang_end: {be}")
    print(raw)

    plt.plot(raw)
    plt.axvline(be, color="orange")
    plt.axvline(wf, color="pink")
    plt.axvline(610)
    plt.show()


if __name__ == "__main__":
    test()
