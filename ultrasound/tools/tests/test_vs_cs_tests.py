import pandas as pd
import tools.prod_algos as prod


def str_raw_data_to_list(raw_data, sep=","):
    return list(map(int, raw_data.strip("][").replace('"', "").split(sep)))


def run_tests():
    file = "/home/david/Projects/demeter-v2/tests/shared/Motif.Demeter.Shared.Helpers.Tests/DistanceComputerEmptySiloTestData.csv"
    df = pd.read_csv(file)
    df["raw_data"] = df.apply(lambda row: str_raw_data_to_list(row["raw_data"], sep=";"), axis=1)

    err_count = 0
    for i in range(len(df)):
        row = df.loc[i]
        wf = prod.wavefront_empty_and_full_detection(row["raw_data"], row["threshold"], row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"])

        err = abs(wf - row["wavefront"])
        if err > 1:
            err_count += 1
    print(f"{err_count} errors for {len(df)} tests.")


run_tests()
