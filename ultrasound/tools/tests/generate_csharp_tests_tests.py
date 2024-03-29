import pandas as pd
import matplotlib.pyplot as plt
import tools.ultrasound_algos as ua
import tools.utils as utils

import sys

sys.path.insert(0, "/home/david/Projects/experiments-api/app/experiments/")
import algos


def test_quality():
    raw = [
        64,
        0,
        0,
        0,
        23,
        87,
        152,
        202,
        236,
        223,
        146,
        147,
        175,
        200,
        216,
        222,
        221,
        216,
        217,
        224,
        234,
        245,
        254,
        255,
        255,
        255,
        255,
        250,
        230,
        195,
        157,
        126,
        103,
        104,
        122,
        137,
        134,
        109,
        85,
        109,
        151,
        190,
        221,
        243,
        255,
        255,
        255,
        255,
        255,
        255,
        249,
        223,
        191,
        163,
        139,
        124,
        113,
        98,
        86,
        72,
        54,
        43,
        49,
        64,
        76,
        78,
        77,
        73,
        81,
        92,
        99,
        96,
        81,
        54,
        44,
        55,
        73,
        88,
        96,
        95,
        85,
        69,
        52,
        37,
        32,
        33,
        37,
        41,
        42,
        42,
        40,
        40,
        41,
        43,
        45,
        46,
        45,
        44,
        43,
        42,
        42,
        41,
        41,
        40,
        40,
        39,
        37,
        34,
        31,
        29,
        28,
        27,
        27,
        27,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        34,
        35,
        35,
        35,
        35,
        35,
        34,
        33,
        32,
        31,
        29,
        27,
        25,
        22,
        20,
        18,
        16,
        15,
        13,
        12,
        12,
        11,
        10,
        9,
        8,
        8,
        8,
        7,
        7,
        7,
        6,
        6,
        5,
        5,
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        6,
        18,
        19,
        29,
        41,
        32,
        24,
        31,
        27,
        35,
        43,
        49,
        44,
        43,
        48,
        48,
        52,
        45,
        36,
        35,
        28,
        23,
        40,
        53,
        49,
        45,
        33,
        40,
        34,
        41,
        35,
        34,
        31,
        27,
        28,
        27,
        33,
        41,
        35,
        37,
        42,
        34,
        25,
        18,
        12,
        9,
        6,
        4,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        1,
        2,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        1,
        1,
        1,
        2,
        2,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
    ]

    sampling_freq = 20000
    quality = algos.signal_quality(raw, sampling_freq)
    print(quality)


def test_wavefront():
    df = pd.read_csv("data/csharp_tests/DistanceComputerEmptySiloTestData.csv")
    df = df[df["Unnamed: 0"] == 1]
    for i in range(len(df)):
        print(i)
        row = df.iloc[i]
        raw = utils.str_raw_data_to_list(row["raw_data"], ";")
        wf1 = round(ua.wavefront_empty_and_full_detection(raw, row["threshold"], row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), 2)
        wf2 = round(algos.wavefront_empty_and_full_detection(raw, row["threshold"], row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"]), 2)

        wf3 = round(row["wavefront"], 2)
        if not wf1 == wf2 == wf3:
            print(f"{i} -> {wf1} - {wf2} - {wf3}")

        plt.plot(raw)
        plt.axvline(wf3)
        plt.axvline(509.6)
        print(wf3)
        print(row["maxBinIndex"])
        plt.show()


if __name__ == "__main__":
    test_wavefront()
    test_quality()
