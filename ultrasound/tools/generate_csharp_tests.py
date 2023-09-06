import pandas as pd
import matplotlib.pyplot as plt

import tools.utils as utils
import tools.ultrasound_algos as ua


def generate_bang_detection_tests(file):
    df = pd.read_csv(file)
    df = df[["raw_data", "samplingFrequency"]]
    df["raw_data"] = df.apply(lambda x: utils.str_raw_data_to_list(x["raw_data"]), axis=1)
    df["minThreshold"] = df.apply(lambda x: sum(x["raw_data"]) / len(x["raw_data"]), axis=1)
    df["windowSize"] = 1000
    df["maxBangLen"] = 10000
    df["bangEnd"] = df.apply(lambda x: ua.detect_main_bang_v2(x["raw_data"], x["minThreshold"], x["windowSize"], x["maxBangLen"], x["samplingFrequency"]), axis=1)
    df["raw_data"] = df.apply(lambda x: str(x["raw_data"]).replace(",", ";"), axis=1)
    df.to_csv("data/tests/BangEndTestData.csv", index=False)
    return df


def generate_wavefront_tests(file):
    df = pd.read_csv(file)
    df = df[["raw_data", "samplingFrequency", "pulseCount"]]
    df["raw_data"] = df.apply(lambda x: utils.str_raw_data_to_list(x["raw_data"]), axis=1)
    df["threshold"] = 0.5
    df["cutoffFreq"] = 150
    df["noDataThreshold"] = 20
    df["wavefront"] = df.apply(lambda x: ua.wavefront(x["raw_data"], 0, x["threshold"], 20, x["pulseCount"], x["samplingFrequency"]), axis=1)
    df["wavefrontLowpass"] = df.apply(lambda x: ua.lowpass_wavefront(x["raw_data"], 0, x["threshold"], x["pulseCount"], 20, x["samplingFrequency"], x["cutoffFreq"], x["noDataThreshold"]), axis=1)
    df["raw_data"] = df.apply(lambda x: str(x["raw_data"]).replace(",", ";"), axis=1)
    df = df[["raw_data", "threshold", "pulseCount", "samplingFrequency", "cutoffFreq", "noDataThreshold", "wavefront", "wavefrontLowpass"]]
    return df


def generate_wavefront_with_empty_detection_tests(file):
    df = pd.read_csv(file)
    df = df[["raw_data", "samplingFrequency", "pulseCount", "maxBinIndex"]]
    df["raw_data"] = df.apply(lambda x: utils.str_raw_data_to_list(x["raw_data"]), axis=1)
    df["threshold"] = 0.5
    df["cutoffFreq"] = 150
    df["noDataThreshold"] = 20
    df["wavefront"] = df.apply(lambda x: ua.wavefront_empty_detection(x["raw_data"], x["threshold"], x["pulseCount"], x["samplingFrequency"], x["maxBinIndex"]), axis=1)
    df["raw_data"] = df.apply(lambda x: str(x["raw_data"]).replace(",", ";"), axis=1)
    df = df[["raw_data", "threshold", "pulseCount", "samplingFrequency", "maxBinIndex", "wavefront"]]
    return df


def multiple_files_tests(files, generate_func, output_file, index=False):
    dfs = []
    for file in files:
        dfs.append(generate_func(file))
    total_df = pd.concat(dfs).reset_index(drop=True)
    if index:
        total_df = total_df.drop([750, 751, 752, 753, 754, 1473, 1477, 1481, 1542, 1547, 1561, 1573, 1598])
    total_df.to_csv(output_file, index=index)



if __name__ == "__main__":
    # Bang end detection test data
    multiple_files_tests(["data/random/Paquette7-2.csv", "data/random/beaudry-1-4.csv", "data/random/small_silo_full.csv", "data/random/Paquette11-6310.csv", "data/random/Paquette4-9.csv"], generate_bang_detection_tests, "data/csharp_tests/BangEndTestData.csv")

    # Wavefront tests
    multiple_files_tests(["data/random/small_silo_full.csv", "data/random/Paquette11-6310.csv", "data/random/Paquette4-9.csv"], generate_wavefront_tests, "data/csharp_tests/DistanceComputerTestData.csv", index=True)

    # Wavefront with empty detection tests
    multiple_files_tests(["data/paquette/Paquette2-6.csv"], generate_wavefront_with_empty_detection_tests, "data/csharp_tests/DistanceComputerEmptySiloTestData.csv")
    
